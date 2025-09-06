import logging
import pandas as pd
from telegram import Update, InputFile
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from airflow.dags.db import query_db # module
from officer_chatbot.nl2sql import natural_to_sql # module
from officer_chatbot.telegram_utils import (generate_bar, generate_pdf, generate_pie_chart_target, 
                                            generate_histogram, generate_scatter, OPENAI_API_KEY, 
                                            TELEGRAM_TOKEN, MODEL_NAME, EMBED_MODEL, MAX_OUTPUT_TOKENS, 
                                            VECTORSTORE_PATH, contains_loan_keyword, count_tokens, 
                                            estimate_cost) # module
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes


# Logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


# RAG
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=EMBED_MODEL)
vectorstore = FAISS.load_local(
    VECTORSTORE_PATH,
    embeddings,
    index_name="index",
    allow_dangerous_deserialization=True,
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


# Helpers
def _format_result(result, colnames):
    if not result:
        return "ℹ️ No data."
    df = pd.DataFrame(result, columns=colnames)
    if len(df) > 15:
        df = df.head(15)
    return f"```markdown\n{df.to_markdown(index=False)}\n```"


# /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_message = (
        "👋 Welcome, Officer! I’m Flexi, your AI assistant.\n\n"
        "You can interact with me in two ways:\n"
        "1) Database queries → `/sql <your request>`\n"
        "    Examples:\n"
        "       • `/sql show all loan volumes`\n"
        "       • `/sql show the borrower profile with ID 100142`\n"
        "       • `/sql get a chart of defaults vs non-defaults`\n"
        "       • `/sql which job types have the highest average default rates? provide a chart`\n"
        "       • `/sql show the profiles of the top 10 borrowers with the highest credit amounts in a PDF`\n\n"
        "2) General loan questions →  Just type them (e.g., our loan process, eligibility, policies, column descriptions).\n\n"
        "3) You will receive a notification here if a borrower wants to register via WhatsApp. Please respond to them as soon as possible.\n\n"
        "4) You can view live predictions of credit risk for real-time borrowers in the “FlexiLoan Live Prediction” bot.\n\n"
        "Notes:\n"
        "   • No `/sql` = no SQL call or interpreter.\n"
        "   • A chart or PDF is provided only if you ask.\n"
    )
    await update.message.reply_text(welcome_message, parse_mode="Markdown")


# /sql
async def sql_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    raw_text = update.message.text or ""
    parts = raw_text.split(" ", 1)
    if len(parts) < 2 or not parts[1].strip():
        await update.message.reply_text(
            "⚠️ Usage: `/sql <your request>`\n"
            "Example: `/sql show today’s loan volume`",
            parse_mode="Markdown",
        )
        return

    text = parts[1].strip()
    logging.info("🧾 /sql request: %s", text)

    try:
        # NL → SQL (explicit by /sql)
        response = natural_to_sql(text)
        sql = response["sql"]

        await update.message.reply_text(f"🧠 Interpreted SQL:\n```sql\n{sql}\n```", parse_mode="Markdown")

        # log GPT usage to console
        if response.get("used_gpt"):
            input_tokens = response.get("input_tokens", 0)
            output_tokens = response.get("output_tokens", 0)
            total = input_tokens + output_tokens
            cost = response.get("cost_usd", 0)
            logging.info(
                "[GPT] Input: %s | Output: %s | Total: %s tokens | Cost: $%.6f",
                input_tokens, output_tokens, total, cost,
            )

        # Run SQL
        result, colnames = query_db(sql)
        if not result:
            await update.message.reply_text("ℹ️ No results found.")
            return

        lowered = text.lower()
        wants_pdf = any(w in lowered for w in ["pdf", "report", "export", "download"])
        wants_chart = any(w in lowered for w in ["chart", "plot", "figure", "visualization", "distribution"])

        # Charts (when asked)
        if wants_chart:
            df = pd.DataFrame(result, columns=colnames)

            NUM_COLS = {
                "AMT_INCOME_TOTAL",
                "AMT_CREDIT",
                "AMT_ANNUITY",
                "DAYS_BIRTH",
                "DAYS_EMPLOYED",
                "TARGET",
                "avg_default_rate",
                "count",
            }
            CAT_COLS = {
                "NAME_INCOME_TYPE",
                "NAME_EDUCATION_TYPE",
                "NAME_FAMILY_STATUS",
                "OCCUPATION_TYPE",
            }

            def _is_binary_series(s: pd.Series) -> bool:
                try:
                    vals = pd.to_numeric(s, errors="coerce").dropna().astype(int).unique().tolist()
                    u = set(vals)
                    return len(u) > 0 and u.issubset({0, 1})
                except Exception:
                    return False

            def _looks_like_sk_id_curr(col_name: str) -> bool:
                name = str(col_name).strip().lower()
                return ("sk_id_curr" in name) or (name == "count")

            # 2-col special pie: TARGET 0/1 + counts
            if df.shape[1] == 2:
                label_col, value_col = colnames[0], colnames[1]
                values_num = pd.to_numeric(df[value_col], errors="coerce")
                if _is_binary_series(df[label_col]) and values_num.notna().any():
                    tmp = pd.DataFrame({label_col: df[label_col], value_col: values_num}).dropna(subset=[value_col])
                    buf = generate_pie_chart_target(tmp[[label_col, value_col]])
                    if buf:
                        await update.message.reply_photo(photo=InputFile(buf, filename="defaults_pie.png"))
                        buf.close()
                        return
                    await update.message.reply_text("❗ Could not generate the defaults pie chart.")
                    return

            # 1 column → histogram (only numeric)
            if len(colnames) == 1:
                col = colnames[0]
                if col in NUM_COLS:
                    buf = generate_histogram(df[[col]])
                    if buf:
                        await update.message.reply_photo(photo=InputFile(buf, filename="histogram.png"))
                        buf.close()
                    else:
                        await update.message.reply_text("❗ Could not generate the chart for this column right now.")
                else:
                    await update.message.reply_text("❗ We cannot provide the chart for this column right now.")
                return

            # 2 columns → histogram / scatter / bar (numeric, categorical)
            if len(colnames) == 2:
                c1, c2 = colnames[0], colnames[1]

                # numeric + (SK_ID_CURR or 'count') → histogram
                if (c1 in NUM_COLS and _looks_like_sk_id_curr(c2)) or (c2 in NUM_COLS and _looks_like_sk_id_curr(c1)):
                    num_col = c1 if c1 in NUM_COLS else c2
                    id_col = c2 if c1 in NUM_COLS else c1

                    counts = pd.to_numeric(df[id_col], errors="coerce")
                    if counts.notna().any():
                        try:
                            w = counts.fillna(0).astype(int).clip(lower=0)
                            expanded = df[[num_col]].loc[df.index.repeat(w)].reset_index(drop=True)
                        except Exception:
                            expanded = df[[num_col]].copy()
                    else:
                        expanded = df[[num_col]].copy()

                    buf = generate_histogram(expanded[[num_col]])
                    if buf:
                        await update.message.reply_photo(photo=InputFile(buf, filename="histogram.png"))
                        buf.close()
                    else:
                        await update.message.reply_text("❗ Could not generate the chart for these columns right now.")
                    return

                # numeric + numeric → scatter
                if c1 in NUM_COLS and c2 in NUM_COLS:
                    buf = generate_scatter(df[[c1, c2]])
                    if buf:
                        await update.message.reply_photo(photo=InputFile(buf, filename="scatter.png"))
                        buf.close()
                    else:
                        await update.message.reply_text("❗ Could not generate the chart for these columns right now.")
                    return

                # categorical + categorical → unsupported
                if c1 in CAT_COLS and c2 in CAT_COLS:
                    await update.message.reply_text("❗ We cannot provide the chart for these columns right now.")
                    return

                # numeric + categorical → bar
                if (c1 in NUM_COLS and c2 in CAT_COLS) or (c1 in CAT_COLS and c2 in NUM_COLS):
                    cat_col = c1 if c1 in CAT_COLS else c2
                    num_col = c2 if c1 in CAT_COLS else c1
                    buf = generate_bar(df[[cat_col, num_col]])
                    if buf:
                        await update.message.reply_photo(photo=InputFile(buf, filename="chart.png"))
                        buf.close()
                    else:
                        await update.message.reply_text("❗ Could not generate the chart for these columns right now.")
                    return

                await update.message.reply_text("❗ We cannot provide the chart for these columns right now.")
                return

            # > 2 columns → not supported
            await update.message.reply_text("❗ We cannot provide the chart for these columns right now.")
            return

        # PDF export (when asked)
        if wants_pdf:
            df = pd.DataFrame(result, columns=colnames)
            pdf_buffer = generate_pdf(df)
            await update.message.reply_document(document=InputFile(pdf_buffer, filename="result.pdf"))
            pdf_buffer.close()
            return

        # Default: compact markdown table
        await update.message.reply_text(_format_result(result, colnames), parse_mode="Markdown")

    except Exception as e:
        await update.message.reply_text(f"❌ SQL Error:\n{e}")


# General (non-command) messages → RAG on loan-only topics
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    if not text or text.startswith("/"):
        return

    if not contains_loan_keyword(text):
        await update.message.reply_text(
            "❌ I can only answer loan-related questions here.\n"
            "• For data queries, use `/sql <your request>`\n"
            "• For general loan topics (process, privacy, columns, credit), ask in plain text.",
            parse_mode="Markdown",
        )
        return

    try:
        logging.info("💬 Passing to RAG (telegram)...")

        try:
            docs = retriever.invoke(text.lower())
            print(f"\n==== docs: {docs}\n")
        except AttributeError:
            docs = retriever.get_relevant_documents(text.lower())
            print(f"\n==== docs: {docs}\n")

        context_blob = text + "\n\n" + "\n\n".join([d.page_content for d in docs])
        input_tokens = count_tokens(context_blob)

        qa_out = qa_chain.invoke({"query": text})
        rag_answer = qa_out.get("result", "")

        output_tokens = count_tokens(rag_answer or "")
        total_cost = estimate_cost(input_tokens, output_tokens)
        logging.info("🔢 [RAG] Input: %s, Output: %s, Cost: $%s", input_tokens, output_tokens, total_cost)

        if not rag_answer or len(rag_answer.strip()) < 20:
            raise ValueError("RAG gave a weak or empty response")

        await update.message.reply_text(rag_answer)

    except Exception as e:
        logging.warning("⚠️ RAG failed: %s", e)
        await update.message.reply_text("❌ Sorry, I couldn’t answer that right now.")


if __name__ == "__main__":
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("sql", sql_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("🚀 Officer Bot is running...")
    app.run_polling()
