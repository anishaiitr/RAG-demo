from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.document_loaders import PyPDFDirectoryLoader


def split_documents():
    loader = PyPDFDirectoryLoader("data")
    docs = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    chunked_documents = text_splitter.split_documents(docs)
    return chunked_documents

def create_faiss_index():
    chunked_documents = split_documents()
    index_persistent_path = 'index/faiss_index'
    db = FAISS.from_documents(chunked_documents, 
                            HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))
    db.save_local(index_persistent_path)
    return index_persistent_path, chunked_documents
