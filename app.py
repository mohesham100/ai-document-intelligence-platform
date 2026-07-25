import streamlit as st
import base64
import time
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO

# Internal Modules
from document_loader import process_uploaded_files
from vector_store import create_and_save_vector_store, load_vector_store, clear_vector_store
from rag_pipeline import answer_question
from summarizer import generate_summary
from quiz_generator import generate_quiz
from suggested_questions import generate_suggested_questions

# Configure Streamlit Page
st.set_page_config(page_title="AI Document Intelligence Platform", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>

[data-testid="stChatInput"] {
    position: fixed;
    bottom: 20px;
    left: 380px;
    right: 30px;
    z-index: 9999;
}

.main .block-container {
    padding-bottom: 100px;
}

/* Tabs */

button[data-baseweb="tab"]{
    font-size:22px !important;
    font-weight:bold !important;
    min-height:60px !important;
    padding:15px 30px !important;
}

</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>

/* Sub Headers */
h3 {
    color: #4FC3F7 !important;

}


</style>
""", unsafe_allow_html=True)


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(
            rgba(0,0,0,0.55),
            rgba(0,0,0,0.55)
        ), url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            
        }}

        .main {{
            background-color: rgba(0,0,0,0.35);
            border-radius: 10px;
            padding: 10px;
        }}

        [data-testid="stSidebar"] {{
            background-color: rgba(20,20,30,0.85);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("images/background1.png")

if "num_docs" not in st.session_state:
    st.session_state.num_docs = 0

if "num_chunks" not in st.session_state:
    st.session_state.num_chunks = 0


# Initialize Session States
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "full_raw_text" not in st.session_state:
    st.session_state.full_raw_text = ""
if "summary" not in st.session_state:
    st.session_state.summary = None
if "suggested_qs" not in st.session_state:
    st.session_state.suggested_qs = []
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None
if "current_input" not in st.session_state:
    st.session_state.current_input = ""

# --- SIDEBAR ---
with st.sidebar:
    st.header("📂 Document Management")

    uploaded_files = st.file_uploader(
        "Upload Documents (PDF, DOCX, TXT, CSV)",
        type=["pdf", "docx", "txt", "csv"],
        accept_multiple_files=True
    )

    # Process Documents
    if st.button("Process Documents", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one document.")
        else:
            with st.spinner("Processing and splitting documents..."):
                
                try:
                    docs = process_uploaded_files(uploaded_files)
                    st.session_state.num_docs = len(uploaded_files)
                    st.session_state.num_chunks = len(docs)

                    # Store raw text for later features
                    st.session_state.full_raw_text = "\n\n".join(
                        [d.page_content for d in docs]
                    )

                    with st.spinner("Generating embeddings (this may take a moment)..."):
                        st.session_state.vector_store = create_and_save_vector_store(docs)

                    st.success("✅ Documents processed successfully!")

                except Exception as e:
                    st.error(f"Error during processing: {e}")

    st.divider()

    # Generate Summary
    st.header("📊 Executive Summary")

    if st.button("Generate Summary", use_container_width=True):
        if not st.session_state.full_raw_text:
            st.warning("Please process documents first.")
        else:
            try:
                with st.spinner("Generating Executive Summary..."):
                    st.session_state.summary = generate_summary(
                        st.session_state.full_raw_text
                    )

                st.success("✅ Summary generated!")

            except Exception as e:
                st.error(f"Failed to generate summary: {e}")

    st.divider()

    # Quiz Generator
    st.header("📝 Quiz Generator")

    if st.button("Generate Quiz", use_container_width=True):
        if not st.session_state.full_raw_text:
            st.warning("Please process documents first.")
        else:
            try:
                with st.spinner("Generating comprehensive quiz..."):
                    st.session_state.quiz_data = generate_quiz(
                        st.session_state.full_raw_text
                    )

                st.success("✅ Quiz generated!")

            except Exception as e:
                st.error(f"Failed to generate quiz: {e}")

    st.divider()

    # Suggested Questions
    st.header("💡 Suggested Questions")

    if st.button("Generate Suggested Questions", use_container_width=True):
        if not st.session_state.full_raw_text:
            st.warning("Please process documents first.")
        else:
            try:
                with st.spinner("Generating Suggested Questions..."):
                    st.session_state.suggested_qs = generate_suggested_questions(
                        st.session_state.full_raw_text
                    )

                st.success("✅ Suggested questions generated!")

            except Exception as e:
                st.error(f"Failed to generate suggested questions: {e}")

    st.divider()

    # System Status
    st.header("⚙️ System Status")

    if st.session_state.vector_store is not None:
        st.success("Database Status: Loaded 🟢")

        if st.button("Clear Database", use_container_width=True):
            clear_vector_store()

            st.session_state.vector_store = None
            st.session_state.chat_history = []
            st.session_state.summary = None
            st.session_state.suggested_qs = []
            st.session_state.full_raw_text = ""
            st.session_state.quiz_data = None
            st.session_state.num_docs = 0
            st.session_state.num_chunks = 0

            st.rerun()

    else:
        st.error("Database Status: Empty 🔴")

    # -------------------
    # Clear Chat
    # -------------------

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    # -------------------
    # Export Chat TXT
    # -------------------

    export_text = ""

    for item in st.session_state.chat_history:
        export_text += f"""
    Question:
    {item['question']}

    Answer:
    {item['answer']}

    {'='*60}
    """

    

    # -------------------
    # Export Chat PDF
    # -------------------

    pdf_buffer = BytesIO()

    doc = SimpleDocTemplate(pdf_buffer)

    styles = getSampleStyleSheet()

    content = []

    for item in st.session_state.chat_history:

        content.append(
            Paragraph(
                f"<b>Question:</b> {item['question']}",
                styles["Normal"]
            )
        )

        content.append(Spacer(1, 10))

        content.append(
            Paragraph(
                f"<b>Answer:</b> {item['answer']}",
                styles["Normal"]
            )
        )

        content.append(Spacer(1, 20))

    doc.build(content)

    pdf_buffer.seek(0)

      

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            "📥 Export TXT",
            export_text,
            file_name="chat_history.txt",
            mime="text/plain",
            use_container_width=True,
            disabled=(len(st.session_state.chat_history) == 0)
        )

    with col2:
        st.download_button(
            "📄 Export PDF",
            pdf_buffer,
            file_name="chat_history.pdf",
            mime="application/pdf",
            use_container_width=True,
            disabled=(len(st.session_state.chat_history) == 0)
        )

# --- MAIN PAGE ---
st.title("🧠 AI Document Intelligence Platform")

st.markdown("""
### Powered by RAG + FAISS + Mistral

Upload multiple documents, generate summaries, create quizzes,
and chat with your data using Retrieval-Augmented Generation.
""")

st.divider()
col1, col2, col3 = st.columns(3)

col1.metric("Documents", st.session_state.num_docs)
col2.metric("Chunks", st.session_state.num_chunks)
col3.metric("Model", "Mistral")

st.markdown("""
<style>

/* Tabs container */
.stTabs [role="tablist"]{
    display:flex;
    width:100%;
    justify-content:space-between;
    gap:20px;
}

/* Each tab */
button[data-baseweb="tab"]{
    flex:1;
    font-size:22px !important;
    font-weight:700 !important;
    min-height:70px !important;

    background:rgba(255,255,255,0.08) !important;

    border:1px solid rgba(255,255,255,0.15) !important;
    border-radius:15px !important;

    backdrop-filter:blur(8px);

    transition:0.3s;
}

/* Hover */
button[data-baseweb="tab"]:hover{
    transform:translateY(-2px);
}

/* Active tab */
button[aria-selected="true"]{
    background:rgba(255,255,255,0.15) !important;
    border:1px solid rgba(255,255,255,0.3) !important;
}

</style>
""", unsafe_allow_html=True)

tabs = st.tabs(["💬 Chat & QA", "📊 Executive Summary", "🎓 Quiz Results"])

# TAB 1: Chat & QA
with tabs[0]:
    if st.session_state.suggested_qs:
        st.subheader("💡 Suggested Questions")
        # Layout buttons in columns
        cols = st.columns(min(len(st.session_state.suggested_qs), 3))
        for i, q in enumerate(st.session_state.suggested_qs):
            col = cols[i % 3]
            if col.button(q, key=f"sq_{i}"):
                st.session_state.current_input = q
    
    st.divider()
    
    # Display Chat History
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.write(chat["answer"])
            if chat["sources"]:
                with st.expander("🔍 View Source Citations"):
                    for idx, src in enumerate(chat["sources"]):
                        st.markdown(f"**Source {idx + 1}:** `{src['source']}` (Page {src['page']})")
                        st.markdown(f"*Similarity Score:* `{src['similarity_score']}`")
                        st.caption(f"_{src['content'][:250]}..._")
                        st.divider()

            if "response_time" in chat:
                st.caption(
                    f"⏱ Response Time: {chat['response_time']:.2f} sec"
                )            

    # Chat Input
    query = st.chat_input("Ask a question about your documents...", key="chat_input")
    
    # Check if a suggested question button populated the session state input
    if st.session_state.current_input != "":
        query = st.session_state.current_input
        st.session_state.current_input = "" # reset
    
    if query:
        if st.session_state.vector_store is None:
            st.warning("Please upload and process documents before asking questions.")
        else:
            # Display user message instantly
            with st.chat_message("user"):
                st.write(query)
                
            with st.chat_message("assistant"):
                with st.spinner("Searching documents and formulating answer..."):
                    try:
                        history_context = ""

                        for chat in st.session_state.chat_history[-3:]:
                            history_context += f"""
                        User: {chat['question']}
                        Assistant: {chat['answer']}
                        """
                        
                        enhanced_query = f"""
                        Previous conversation:

                        {history_context}

                        Current Question:
                        {query}
                        """    

                        start_time = time.time()

                        answer, sources = answer_question(
                            enhanced_query,
                            st.session_state.vector_store
                        )

                        response_time = round(time.time() - start_time, 2)
                        st.write(answer)
                        st.markdown("---")
                        st.caption(f"⏱ Response Time: {response_time    :.2f} sec")
                        
                        if sources:
                            with st.expander("🔍 View Source Citations"):
                                for idx, src in enumerate(sources):
                                    st.markdown(f"**Source {idx + 1}:** `{src['source']}` (Page {src['page']})")
                                    st.markdown(f"*Similarity Score:* `{src['similarity_score']}`")
                                    st.caption(f"_{src['content'][:250]}..._")
                                    st.divider()
                                    
                        # Save to history
                        st.session_state.chat_history.append({
                            "question": query,
                            "answer": answer,
                            "sources": sources,
                            "response_time": response_time
                        })
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error querying Gemini: {e}")

def create_pdf(title, content_text):

    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()

    elements = [
        Paragraph(title, styles["Title"]),
        Spacer(1, 20),
        Paragraph(
            content_text.replace("\n", "<br/>"),
            styles["BodyText"]
        )
    ]

    doc.build(elements)

    buffer.seek(0)

    return buffer

# TAB 2: Executive Summary
with tabs[1]:

    if st.session_state.summary:

        with st.container(border=True):

            st.subheader("📊 Executive Summary")

            st.markdown(st.session_state.summary)

        summary_pdf = create_pdf(
            "Executive Summary",
            st.session_state.summary
        )

        st.download_button(
            "📕 Download Summary PDF",
            summary_pdf,
            file_name="Executive_Summary.pdf",
            mime="application/pdf"
        )

    else:
        st.info(
            "Upload and process documents to generate a structured summary automatically."
        )

# TAB 3: Quiz Results
with tabs[2]:

    if st.session_state.quiz_data:
        formatted_quiz = st.session_state.quiz_data.replace(
            "A)", "\n\nA)"
        ).replace(
            "B)", "\n\nB)"
        ).replace(
            "C)", "\n\nC)"
        ).replace(
            "D)", "\n\nD)"
        ).replace(
            "Answer:", "\n\n### Answer:\n"
        )

        st.markdown(formatted_quiz)

        quiz_pdf = create_pdf(
            "Generated Quiz",
            st.session_state.quiz_data
        )

        st.download_button(
            "📕 Download Quiz PDF",
            quiz_pdf,
            file_name="Document_Quiz.pdf",
            mime="application/pdf"
        )

    else:
        st.info(
            "Click 'Generate Quiz' in the sidebar to create a test based on your documents."
        )