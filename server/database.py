import psycopg2
from config import *


def create_database():
    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        dbname=POSTGRES_DB,
    )
    conn.autocommit = True
    cur = conn.cursor()

    # Create table to store data
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS data (
            id SERIAL PRIMARY KEY,
            filename VARCHAR(255)  NULL,
            embeddings BYTEA  NULL,
            question TEXT  NULL,
            response TEXT  NULL
        );
    """
    )

    cur.close()
    conn.close()


def insert_data(filename=None, embeddings=None, question=None, response=None):
    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        dbname=POSTGRES_DB,
    )
    conn.autocommit = True
    cur = conn.cursor()

    # Create a dictionary to store the available parameters and their values
    params_dict = {}

    # Add filename to params_dict
    params_dict["filename"] = filename

    # Add embeddings to params_dict if available
    if embeddings is not None:
        params_dict["embeddings"] = psycopg2.Binary(embeddings)

    # Add question to params_dict if available
    if question is not None:
        params_dict["question"] = question

    # Add response to params_dict if available
    if response is not None:
        params_dict["response"] = response

    # Use the keys and values from the params_dict to create the SQL query
    columns = ", ".join(params_dict.keys())
    placeholders = ", ".join(["%s"] * len(params_dict))
    values = tuple(params_dict.values())

    # Insert data into table
    cur.execute(
        f"""
        INSERT INTO data ({columns}) 
        VALUES ({placeholders});
        """,
        values,
    )

    cur.close()
    conn.close()


# once the database is created, the following line can be commented out


# create_database()
