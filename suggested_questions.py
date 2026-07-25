#from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

def generate_suggested_questions(full_text: str) -> list[str]:
    """Generates 5 to 10 intelligent questions a user might ask about the document."""
    safe_text = full_text[:15000]
    
    prompt_template = PromptTemplate(
        input_variables=["text"],
        template="""Based on the following document text, generate 5 to 8 intelligent, analytical questions that a user could ask to deeply understand the content. 
The questions should cover main findings, risks, numerical comparisons (if any), and conclusions.
Output ONLY the questions, one per line, with no numbers, bullet points, or extra formatting.

Document Text:
{text}"""
    )
    
    #llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.5)
    llm = ChatOllama(model="mistral", temperature=0.5)
    prompt = prompt_template.format(text=safe_text)
    
    response = llm.invoke(prompt)
    
    # Clean up the output into a Python list
    questions = [q.strip("- *1234567890. \n") for q in response.content.split("\n") if q.strip()]
    return [q for q in questions if q][:8] # Return max 8 questions