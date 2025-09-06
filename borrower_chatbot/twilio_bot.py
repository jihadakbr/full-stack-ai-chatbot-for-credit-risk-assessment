from pathlib import Path # top
import os
import json
import logging
import requests
import tiktoken
from dotenv import load_dotenv
from flask import Flask, request
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from twilio.twiml.messaging_response import MessagingResponse
from borrower_chatbot.credit_questionnaire import CreditQuestionnaire

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

app = Flask(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # Goes up 2 levels
env_path = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=env_path, override=True)

vectorstore_path = PROJECT_ROOT / "rag_vectorstore"
keywords_path = PROJECT_ROOT / "info" / "keywords.json"

ngrok1_url = os.getenv("NGROK1_URL", "https://a4f971f433ad.ngrok-free.app") # Docker or Dev mode

PREDICT_URL = "http://localhost:5000/predict"
MEDIA_URL = f"{ngrok1_url}/pdfs/credit_report.pdf"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# GPT API Pricing (as of 2025)
MODEL_NAME = "gpt-3.5-turbo"
PRICE_PER_1K_INPUT_TOKENS = 0.0005
PRICE_PER_1K_OUTPUT_TOKENS = 0.0015
MAX_OUTPUT_TOKENS = 100

# Load FAISS vectorstore and prepare the retriever
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=EMBED_MODEL)
vectorstore = FAISS.load_local(
    vectorstore_path,
    embeddings,
    index_name="index",
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.5},
)

# # for testing, run once
# res = retriever.get_relevant_documents("company name")
# for i, d in enumerate(res):
#     print(f"\n--- {i} --- {d.metadata}")
#     print(d.page_content[:300])

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are FlexiLoan’s assistant (Flexi). Answer as the company in first-person plural "
        "(we/our/us). Do not use third person (no they/their/them). "
        "Base the answer ONLY on the context. If the answer isn’t in the context, say: "
        "\"I don’t have that information yet.\" Use concise bullets when appropriate.\n\n"
        "Question: {question}\n\nContext:\n{context}\n\nAnswer:"
    ),
)

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        temperature=0,
        max_tokens=max(256, MAX_OUTPUT_TOKENS),
        model_name=MODEL_NAME,
    ),
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": QA_PROMPT},
    return_source_documents=False,   # Turn on (TRUE) while testing. True/False
)

with open(keywords_path, "r") as f:
    keywords = json.load(f)

# Normalize to lowercase
LOAN_KEYWORDS = [kw.lower() for kw in keywords["loan_keywords"]]
REGISTRATION_KEYWORDS = [kw.lower() for kw in keywords["registration_keywords"]]

def count_tokens_from_string(text, model=MODEL_NAME):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def estimate_cost(input_tokens, output_tokens):
    cost = (
        (input_tokens / 1000) * PRICE_PER_1K_INPUT_TOKENS +
        (output_tokens / 1000) * PRICE_PER_1K_OUTPUT_TOKENS
    )
    return round(cost, 6)

def contains_keyword(message, keywords):
    return any(word in message.lower() for word in keywords)

def send_telegram_message(sender_phone_raw: str, user_text: str):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        display_phone = _format_id_phone(sender_phone_raw)

        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": (
                "📥 Loan Registration Interest\n\n"
                f"From: {display_phone}\n"
                "Via: WhatsApp\n"
                f"Message: “{user_text.strip()}”"
            )
        }
        response = requests.post(url, json=payload)
        print(f"✅ Telegram sent: {response.status_code}")
    except Exception as e:
        print(f"❌ Telegram send failed: {e}")

# simple in-memory per-sender state
SESSIONS: dict[str, CreditQuestionnaire] = {}

# helpers for credit intent & cancel
CREDIT_INTENT_PHRASES = {
    "credit", "credit check", "risk check", "credit risk", "predict", "prediction",
    "loan risk", "start credit", "check credit", "credit scoring"
}
CANCEL_PHRASES = {"cancel", "stop", "quit", "exit"}

def _is_credit_intent(text: str):
    t = text.strip().lower()
    return t in CREDIT_INTENT_PHRASES or t.startswith("credit ") or "credit check" in t or "risk" in t and "credit" in t

def _start_credit_flow(sender: str):
    q = CreditQuestionnaire()
    SESSIONS[sender] = q
    return q.start()

def _handle_credit_flow(sender: str, user_text: str):
    q = SESSIONS.get(sender)
    if not q:
        q = CreditQuestionnaire()
        SESSIONS[sender] = q
        return True, q.start(), False

    ok, reply = q.handle(user_text)
    if q.is_complete():
        try:
            features = q.build_features()
            print("🔧 Built features:", features)
            result = requests.post(PREDICT_URL, json=features, timeout=30).json()
            msg = result.get("message", "Prediction completed.")
            SESSIONS.pop(sender, None)
            return True, msg, True
        except Exception as e:
            SESSIONS.pop(sender, None)
            return True, f"⚠️ Error during prediction: {e}", False
    else:
        return ok, reply, False

def _format_id_phone(raw: str) -> str:
    """
    Format Indonesian numbers to a readable display like: +62 813-3232-6785
    Accepts values like 'whatsapp:+62813...', '+62813...', '0813...', '813...'
    """
    if not raw:
        return "Unknown"

    s = raw.strip().replace("whatsapp:", "")
    digits = "".join(ch for ch in s if ch.isdigit())

    # Derive national number
    if digits.startswith("62"):
        national = digits[2:]
    elif digits.startswith("0"):
        national = digits[1:]
    elif digits.startswith("8"):
        national = digits
    else:
        return f"+{digits}" if digits else raw

    # Build display: +62 {first3}-{next up to 4}-{rest}
    e164 = "+62" + national
    if len(national) <= 3:
        return e164

    op = national[:3]
    rest = national[3:]
    if len(rest) <= 4:
        return f"+62 {op}-{rest}"
    else:
        group1 = rest[:4]
        group2 = rest[4:]
        return f"+62 {op}-{group1}-{group2}"

@app.route("/whatsapp", methods=["POST"])
def whatsapp_bot():
    incoming_msg = request.values.get('Body', '').strip()
    sender = request.values.get('From')

    print(f"📨 Received WhatsApp message: '{incoming_msg}' from {sender}")

    resp = MessagingResponse()
    msg = resp.message()

    lower_msg = incoming_msg.lower()

    # Allow users to cancel questionnaire
    if lower_msg in CANCEL_PHRASES:
        if sender in SESSIONS:
            SESSIONS.pop(sender, None)
            msg.body("✅ Cancelled. You can type 'credit check' anytime to restart.")
        else:
            msg.body("Nothing to cancel. Type 'credit check' to begin.")
        return str(resp)

    # Credit intent - start or continue flow
    if _is_credit_intent(lower_msg) or sender in SESSIONS:
        ok, reply, attach_pdf = _handle_credit_flow(sender, incoming_msg)
        msg.body(reply)
        if attach_pdf:
            msg.media(MEDIA_URL)
        return str(resp)

    if lower_msg == "start":
        msg.body(
            "👋 Welcome to FlexiLoan! I’m Flexi, your AI assistant.\n\n"
            "You can:\n"
            "• Have a question? Ask me anything about loans, applications, or our services.\n"
            "• Type *credit check* to answer quick questions and get a risk prediction.\n"
            "• Ready to apply for a loan? I can connect you with a FlexiLoan officer who will guide you through the registration process.\n\n"
            "What would you like to do?"
        )
        return str(resp)

    # Raw JSON prediction, keep for devs
    if incoming_msg.startswith("{") and incoming_msg.endswith("}"):
        try:
            user_data = json.loads(incoming_msg)
            result = requests.post(PREDICT_URL, json=user_data, timeout=30).json()
            reply = result.get("message", "Prediction completed.")
            msg.body(reply)
            msg.media(MEDIA_URL)
        except Exception as e:
            msg.body(f"⚠️ Error: {str(e)}")
        return str(resp)

    # Registration Intent
    elif contains_keyword(lower_msg, REGISTRATION_KEYWORDS):
        send_telegram_message(sender, incoming_msg)
        msg.body("✅ Thanks! We've informed our loan officer. They’ll contact you shortly.")
        return str(resp)

    # Non-Loan Topics
    elif not contains_keyword(lower_msg, LOAN_KEYWORDS):
        msg.body("❌ Sorry, I can only answer loan-related questions. Please ask something about your credit or loan application.")
        return str(resp)

    # Handle general questions using RAG
    else:
        try:
            print("💬 Passing to RAG (whatsapp)...")

            try:
                docs = retriever.invoke(lower_msg)
                print(f"\n==== docs: {docs}\n")
            except AttributeError:
                docs = retriever.get_relevant_documents(lower_msg)
                print(f"===== Attribute missing: {e}")
                # print(f"\n==== docs: {docs}\n")

            input_text = lower_msg + "\n\n" + "\n\n".join([doc.page_content for doc in docs])
            input_tokens = count_tokens_from_string(input_text)

            qa_out = qa_chain.invoke({"query": lower_msg})
            rag_answer = qa_out.get("result", qa_out if isinstance(qa_out, str) else "")

            output_tokens = count_tokens_from_string(rag_answer or "")
            total_cost = estimate_cost(input_tokens, output_tokens)
            print(f"🔢 [RAG] Input tokens: {input_tokens}, Output tokens: {output_tokens}, Cost: ${total_cost}")

            if not rag_answer or len(rag_answer.strip()) < 20:
                raise ValueError("RAG gave a weak or empty response")

            msg.body(rag_answer)
            return str(resp)
        except Exception as e:
            print(f"⚠️ RAG failed: {e}")
            msg.body("❌ Sorry, I couldn’t answer that right now.")
            return str(resp)


if __name__ == "__main__":
    app.run(port=5001)

