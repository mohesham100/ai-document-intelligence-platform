#from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

def generate_summary(full_text: str) -> str:
    """Generates an Executive Summary, Key Findings, Important Topics, and Action Items."""
    # Truncate text to roughly 20,000 characters to ensure it fits safely and processes fast
    safe_text = full_text[:20000]
    
    prompt_template = PromptTemplate(
        input_variables=["text"],
        template="""You are a professional business analyst. Read the following text and generate a structured summary with exactly these four sections:

1. Executive Summary: A concise overview of the entire content (1-2 paragraphs).
2. Key Findings: The most important insights, data points, or conclusions extracted (bullet points).
3. Important Topics: The major themes or subjects discussed (bullet points).
4. Action Items: Any recommendations, tasks, next steps, or warnings mentioned (bullet points).

Text to summarize:
{text}

Provide the output in clean Markdown format."""
    )
    
    #llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.2)
    llm = ChatOllama(model="mistral", temperature=0.2)
    prompt = prompt_template.format(text=safe_text)
    
    response = llm.invoke(prompt)
    return response.content