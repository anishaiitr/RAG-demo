from build_index import create_faiss_index
from find_context import find_context
from promptflow.core import tool
import os

@tool
def get_response(question: str, openai: str):
    index_persistent_path = 'index/faiss_index'
    if not os.path.exists(index_persistent_path):
        index_persistent_path = create_faiss_index()
    print('index created')
    response = find_context(question, index_persistent_path, openai)
    return response