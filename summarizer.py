#from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

def generate_summary(full_text: str) -> str:
    """Generates an Executive Summary, Key Findings, Important Topics, and Action Items."""
    # Truncate text to roughly 100,000 characters to ensure it fits safely and processes fast
    safe_text = full_text[:100000]
    
    prompt_template = PromptTemplate(
        input_variables=["text"],
        template="""
You are a senior business analyst and document intelligence expert.

Analyze the provided document thoroughly and create a detailed Executive Summary.

Requirements:
- Focus on synthesizing and explaining the content, not just listing topics.
- The Executive Summary section should be the longest and most detailed section.
- Capture the main ideas, arguments, concepts, findings, and conclusions.
- Do NOT simply repeat headings from the document.
- Avoid generic statements.
- Avoid excessive bullet-point lists.
- If the document is long, summarize all major sections proportionally.

Output format:

# Executive Summary
Write a comprehensive executive summary covering all major sections of the document.

This section must contain at least:
- 8 to 12 detailed paragraphs
- Key concepts
- Important explanations
- Major arguments
- Critical technical details
- Final conclusions

This section should represent approximately 70% of the entire output.

# Key Findings
(List the most important findings and insights.)

# Important Topics
(Only 5-10 topics maximum.)

# Recommendations / Action Items
(Practical actions or conclusions derived from the document.)

Document:
{text}
"""
    )
    
    #llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.2)
    llm = ChatOllama(model="mistral", temperature=0.2)
    prompt = prompt_template.format(text=safe_text)
    
    response = llm.invoke(prompt)
    return response.content