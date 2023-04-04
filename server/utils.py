import openai
import logging
import sys
import time
import psycopg2

from config import *

openai.api_key = "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"
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("debug.log"), logging.StreamHandler(sys.stdout)],
)


def get_pinecone_id_for_file_chunk(session_id, filename, chunk_index):
    """
    Helper function to generate a unique ID for a file chunk using the session ID, filename, and chunk index.
    """
    return str(session_id + "-!" + filename + "-!" + str(chunk_index))


def get_embedding(text, engine):
    """
    Gets the embedding for a given text using the OpenAI API.
    """
    return openai.Engine(id=engine).embeddings(input=[text])["data"][0]["embedding"]


def get_embeddings(text_array, engine):
    """
    Gets embeddings for an array of text using the OpenAI API.
    Implements exponential backoff in case of network errors.
    """
    # Parameters for exponential backoff
    max_retries = 5  # Maximum number of retries
    base_delay = 1  # Base delay in seconds
    factor = 2  # Factor to multiply the delay by after each retry
    while True:
        try:
            return openai.Engine(id=engine).embeddings(input=text_array)["data"]
        except Exception as e:
            if max_retries > 0:
                logging.info(f"Request failed. Retrying in {base_delay} seconds.")
                time.sleep(base_delay)
                max_retries -= 1
                base_delay *= factor
            else:
                raise e
