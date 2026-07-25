#from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

#def get_llm():
#    """Initializes the Gemini model."""
#    return ChatGoogleGenerativeAI(
#        model="gemini-flash-latest",
#        temperature=0.0 # Low temperature for factual RAG
#    )

def get_llm():
    """Initializes the Gemini model."""
    return ChatOllama(
        model="mistral",
        temperature=0.0 # Low temperature for factual RAG
    )

def answer_question(question: str, vector_store, top_k: int = 10):
    """
    Retrieves context from FAISS and generates an answer using Gemini.
    Returns the answer and the source chunks with similarity scores.
    """
    # Retrieve top K chunks with L2 distance scores
    results = vector_store.similarity_search_with_score(question, k=top_k)
    
    
    if not results:
        return "The uploaded documents do not contain sufficient information to answer this question.", []

    context_text = ""
    sources = []
    
    for doc, distance in results:
        context_text += f"\n\n---\nDocument: {doc.metadata.get('source')}, Page: {doc.metadata.get('page')}\nContent: {doc.page_content}"
        # Convert FAISS L2 distance to a simple 0-1 similarity score (lower distance = higher similarity)
        similarity_score = round(1 / (1 + distance), 4)
        
        sources.append({
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", "N/A"),
            "chunk_id": doc.metadata.get("chunk_id", "N/A"),
            "similarity_score": similarity_score,
            "content": doc.page_content
        })

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are an expert AI document assistant. Answer the question based ONLY on the provided context.
If the context does not contain enough information to fully answer the question, state exactly: 
"The uploaded documents do not contain sufficient information to answer this question." Do not hallucinate or use outside knowledge.

Context:
{context}

Question:
{question}

Answer:"""
    )
    
    llm = get_llm()
    prompt = prompt_template.format(context=context_text, question=question)
    
    response = llm.invoke(prompt)
    
    return response.content, sources