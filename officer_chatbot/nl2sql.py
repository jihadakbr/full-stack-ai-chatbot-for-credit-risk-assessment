import re
from langchain_openai import ChatOpenAI
from officer_chatbot.telegram_utils import (OPENAI_API_KEY, MODEL_NAME, MAX_OUTPUT_TOKENS, 
                                            contains_loan_keyword, count_tokens) # module

COLUMNS = [
    "SK_ID_CURR",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "OCCUPATION_TYPE",
]

PROMPT_TEMPLATE = (
    """
You are a SQL assistant for PostgreSQL.
Translate the following natural language question into a syntactically correct SQL query.
Follow these rules carefully:
1. Use only the table 'clean_data' unless the question refers to real-time new applicants/borrowers data, in which case use 'new_applicants_clean_data'. Both tables have the same columns.
2. Only use these columns and their data types (and ignore any others even if mentioned in the question):
   - "SK_ID_CURR" -> text
   - "AMT_INCOME_TOTAL" -> float
   - "AMT_CREDIT" -> float
   - "AMT_ANNUITY" -> float
   - "DAYS_BIRTH" -> int
   - "DAYS_EMPLOYED" -> int
   - "NAME_INCOME_TYPE" -> text
   - "NAME_EDUCATION_TYPE" -> text
   - "NAME_FAMILY_STATUS" -> text
   - "OCCUPATION_TYPE" -> text
   - "TARGET" -> int (1 if defaulted, 0 if not)
3. Always enclose column names in double quotes (e.g., "COLUMN_NAME") and use UPPERCASE for column names.
4. For text columns, wrap values in single quotes (e.g., WHERE "SK_ID_CURR" = '100001').
5. For int or float columns, do NOT wrap values in quotes (e.g., WHERE "AMT_INCOME_TOTAL" > 10000).
6. If the question refers to columns that are not in the provided list, respond with:
   "Only ask about the following columns: {columns}."
7. Whatever the question is, only generate the SQL query using the provided columns: {columns}. Do not include or infer any other columns.
8. Generate only the SQL query as the output, without explanation.

Question: {question}
SQL Query:
"""
)

# Init LLM once
_llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    temperature=0,
    max_tokens=MAX_OUTPUT_TOKENS,
    model_name=MODEL_NAME,
)


def _quote_columns(sql: str) -> str:
    # Ensure all known columns are quoted
    out = sql
    for col in COLUMNS + ["TARGET"]:
        pattern = rf'(?<!\")\b{col}\b(?!\")'
        out = re.sub(pattern, f'"{col}"', out)
    return out.strip()


def natural_to_sql(question: str, allow_gpt_fallback: bool = True) -> dict:
    q = (question or "").strip()
    if not contains_loan_keyword(q):
        raise Exception(
            "❌ Sorry, I can only answer loan-related questions. Please ask something about your credit or loan application."
        )

    lowered = q.lower()

    # Fast paths
    if "loan" in lowered and "volume" in lowered:
        sql = (
            'SELECT '
            '(SELECT COUNT(*) FROM clean_data) AS "Existing Borrowers", '
            '(SELECT COUNT(*) FROM new_applicants_clean_data) AS "New Borrowers", '
            '((SELECT COUNT(*) FROM clean_data) + (SELECT COUNT(*) FROM new_applicants_clean_data)) AS "Total"'
        )
        return {"sql": sql, "used_gpt": False, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}

    if "defaults" in lowered and "non-defaults" in lowered:
        sql = (
            'SELECT "TARGET", COUNT(*) AS count\n'
            'FROM clean_data\n'
            'WHERE "TARGET" IS NOT NULL\n'
            'GROUP BY "TARGET"\n'
            'ORDER BY "TARGET"'
        )
        return {"sql": sql, "used_gpt": False, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}

    if not allow_gpt_fallback:
        raise Exception("🔍 GPT fallback disabled. No matching logic.")

    # Prompt
    prompt = PROMPT_TEMPLATE.format(columns=", ".join(COLUMNS), question=q)
    input_tokens = count_tokens(prompt)

    response_msg = _llm.invoke(prompt)
    sql = (response_msg.content or "").strip()
    output_tokens = count_tokens(sql)

    sql = _quote_columns(sql)

    return {
        "sql": sql,
        "used_gpt": True,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": 0.0,
    }
