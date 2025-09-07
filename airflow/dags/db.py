import os
from pathlib import Path  # top

import psycopg2
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Goes up 3 levels
env_path = PROJECT_ROOT / ".env"
# print(f"env_path: {env_path}")

load_dotenv(dotenv_path=env_path, override=True)


def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_DATABASE"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )


def query_db(sql):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(sql)
    result = cursor.fetchall()
    colnames = [desc[0] for desc in cursor.description]
    cursor.close()
    conn.close()
    return result, colnames


def get_db_url():
    """Return SQLAlchemy-compatible PostgreSQL URL using environment variables."""
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    db_name = os.getenv("DB_DATABASE")

    return (
        f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    )


def get_db_new_table():
    """Return the new table name from environment."""
    return os.getenv("DB_NEW_TABLE")
