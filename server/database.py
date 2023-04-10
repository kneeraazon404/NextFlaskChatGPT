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

    # Create table for filenames and embeddings
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS embeddings (
            id SERIAL PRIMARY KEY,
            filename VARCHAR(255)  NULL,
            embedding BYTEA  NULL
        );
        """
    )

    # Create table for questions and answers
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS qa (
            id SERIAL PRIMARY KEY,
            question TEXT  NULL,
            answer TEXT  NULL
        );
        """
    )

    cur.close()
    conn.close()


def insert_embedding_data(filename=None, embedding=None):
    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        dbname=POSTGRES_DB,
    )
    conn.autocommit = True
    cur = conn.cursor()

    # Insert data into embeddings table
    cur.execute(
        """
        INSERT INTO embeddings (filename, embedding)
        VALUES (%s, %s);
        """,
        (filename, psycopg2.Binary(embedding)),
    )

    cur.close()
    conn.close()


def insert_qa_data(question=None, answer=None):
    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        dbname=POSTGRES_DB,
    )
    conn.autocommit = True
    cur = conn.cursor()

    # Insert data into qa table
    cur.execute(
        """
        INSERT INTO qa (question, answer)
        VALUES (%s, %s);
        """,
        (question, answer),
    )

    cur.close()
    conn.close()


# create database tables if they don't exist
create_database()
