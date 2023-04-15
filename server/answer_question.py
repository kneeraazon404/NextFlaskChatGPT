from utils import get_embedding
from flask import jsonify
from config import *
from flask import current_app

import openai

from config import *
from database import insert_qa_data

TOP_K = 10


def get_answer_from_files(question, session_id, pinecone_index):
    logging.info(f"Getting answer for question: {question}")

    search_query_embedding = get_embedding(question, EMBEDDINGS_MODEL)

    try:
        query_response = pinecone_index.query(
            namespace=session_id,
            top_k=TOP_K,
            include_values=False,
            include_metadata=True,
            vector=search_query_embedding,
        )
        logging.info(
            f"[get_answer_from_files] received query response from Pinecone: {query_response}"
        )

        files_string = ""
        file_text_dict = current_app.config["file_text_dict"]

        for i in range(len(query_response.matches)):
            result = query_response.matches[i]
            file_chunk_id = result.id
            score = result.score
            filename = result.metadata["filename"]
            file_text = file_text_dict.get(file_chunk_id)
            file_string = f'###\n"{filename}"\n{file_text}\n'
            if score < COSINE_SIM_THRESHOLD and i > 0:
                logging.info(
                    f"[get_answer_from_files] score {score} is below threshold {COSINE_SIM_THRESHOLD} and i is {i}, breaking"
                )
                break
            files_string += file_string

        prompt = (
            f"Given a question, try to answer it using the content of the file extracts below, and if you cannot answer, or find "
            f'a relevant file, just output "I couldn\'t find the answer to that question in your files.".\n\n'
            f"If the answer is not contained in the files or if there are no file extracts, respond with \"I couldn't find the answer "
            f'to that question in your files." If the question is not actually a question, respond with "That\'s not a valid question."\n\n'
            f"In the cases where you can find the answer, first give the answer. Then explain how you found the answer from the source or sources, "
            f"and use the exact filenames of the source files you mention. Do not make up the names of any other files other than those mentioned "
            f"in the files context. Give the answer in markdown format."
            f'Use the following format:\n\nQuestion: <question>\n\nFiles:\n<###\n"filename 1"\nfile text>\n<###\n"filename 2"\nfile text>...\n\n'
            f'Answer: <answer or "I couldn\'t find the answer to that question in your files" or "That\'s not a valid question.">\n\n'
            f"Question: {question}\n\n"
            f"Files:\n{files_string}\n\n"
        )

        logging.info(f"[get_answer_from_files] prompt: {prompt}")

        response = openai.Completion.create(
            prompt=prompt,
            temperature=0,
            max_tokens=1000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            engine=GENERATIVE_MODEL,
        )

        pinecone_answer = response.choices[0].text.strip()
        insert_qa_data(question, pinecone_answer)
        logging.info(f"[get_answer_from_files] answer: {pinecone_answer}")

    except Exception as e:
        logging.error(f"[get_answers] Error with ChatGPT: {e}")
        chatgpt_answer = "I'm sorry, I couldn't generate an answer for your question."
    # return jsonify({"answer": answer})
    # Get answer from ChatGPT
    try:
        chatgpt_prompt = f"Q: {question}\nA: "
        chatgpt_response = openai.Completion.create(
            prompt=chatgpt_prompt,
            temperature=0.7,
            max_tokens=1000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            engine=GENERATIVE_MODEL,
        )

        chatgpt_answer = chatgpt_response.choices[0].text.strip()
        insert_qa_data(question, chatgpt_answer)
        logging.info(f"[get_answers] ChatGPT answer: {chatgpt_answer}")

    except Exception as e:
        logging.error(f"[get_answers] Error with ChatGPT: {e}")
        chatgpt_answer = "I'm sorry, I couldn't generate an answer for your question."
    return jsonify({"pinecone_answer": pinecone_answer, "answer": chatgpt_answer})
