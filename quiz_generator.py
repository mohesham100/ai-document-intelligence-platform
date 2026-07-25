#from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

def generate_quiz(full_text: str) -> str:
    """Generates a 20-question quiz (10 MCQ, 5 T/F, 5 Short Answer) from the text."""
    safe_text = full_text[:20000]
    
    prompt_template = PromptTemplate(
        input_variables=["text"],
        template="""You are an expert educator.

Create a comprehensive quiz based STRICTLY on the provided document text.

Do NOT add information that does not exist in the document.

Use the EXACT formatting below.

# Multiple Choice Questions (10 Questions)

Question 1:
[Question]

A) Option A

B) Option B

C) Option C

D) Option D

Correct Answer:
A

--------------------------------------------------

Question 2:
...

Continue the same format for all 10 questions.

# True / False Questions (5 Questions)

Question 1:
[Statement]

Correct Answer:
True

--------------------------------------------------

Question 2:
...

Continue the same format for all 5 questions.

# Short Answer Questions (5 Questions)

Question 1:
[Question]

Correct Answer:
[Short Answer]

--------------------------------------------------

Question 2:
...

Continue the same format for all 5 questions.

IMPORTANT:

You MUST generate the quiz exactly in the format above.

Do NOT summarize the document.
Do NOT provide notes.
Do NOT provide outlines.
Do NOT provide study material.


Document Text:
{text}

Output clean Markdown only.
"""
    )
    
    #llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.3)

    llm = ChatOllama(model="mistral", temperature=0.3)
    prompt = prompt_template.format(text=safe_text)
    
    response = llm.invoke(prompt)
    return response.content