import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

DB_DIR = "db"
FAISS_INDEX_PATH = os.path.join(DB_DIR, "faiss_index")

def get_embeddings_model():
    """Returns the HuggingFace embeddings model."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

def create_and_save_vector_store(documents: list[Document]) -> FAISS:
    """Creates a new FAISS vector store from documents and saves it locally."""
    os.makedirs(DB_DIR, exist_ok=True)
    embeddings = get_embeddings_model()
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(FAISS_INDEX_PATH)
    return vector_store

def load_vector_store() -> FAISS:
    """Loads an existing FAISS vector store from the local directory."""
    if not os.path.exists(FAISS_INDEX_PATH):
        return None
    embeddings = get_embeddings_model()
    # allow_dangerous_deserialization is required for FAISS local loading in newer LangChain versions
    return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

def clear_vector_store():
    """Deletes the local FAISS index files."""
    if os.path.exists(FAISS_INDEX_PATH):
        for file in os.listdir(FAISS_INDEX_PATH):
            os.remove(os.path.join(FAISS_INDEX_PATH, file))
        os.rmdir(FAISS_INDEX_PATH)