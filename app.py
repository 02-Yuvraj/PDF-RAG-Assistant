
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from src.pdf_processor import extract_text_from_pdf
from src.text_chunker import chunk_pages
from src.embedding_generator import create_embedding_model
from src.vector_store import create_vector_store
from src.answer_generator import generate_answer


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="PDF-RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# 2. SESSION STATE
# --------------------------------------------------

defaults = {
    "retriever": None,
    "pdf_name": None,
    "page_count": 0,
    "chunk_count": 0,
    "chat_history": [],
    "pending_question": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --------------------------------------------------
# 3. PROCESS THE PDF
# --------------------------------------------------

def process_pdf(uploaded_file):
    """Extract, chunk, embed, and index an uploaded PDF."""

    if not os.getenv("GROQ_API_KEY"):
        st.error(
            "Groq API key not found. "
            "Please check your project-root .env file."
        )
        return False

    temporary_path = None

    try:
        with st.spinner(
            "📖 Reading your PDF and preparing the AI search..."
        ):
            with tempfile.NamedTemporaryFile(
                suffix=".pdf",
                delete=False
            ) as temporary_file:
                temporary_file.write(uploaded_file.getvalue())
                temporary_path = temporary_file.name

            # Extract text and original PDF page numbers.
            pages = extract_text_from_pdf(temporary_path)

            if not pages:
                raise ValueError(
                    "No extractable text was found. "
                    "This PDF may be scanned or image-only."
                )

            # Split text into page-aware chunks.
            chunks = chunk_pages(pages)

            if not chunks:
                raise ValueError(
                    "No text chunks could be created."
                )

            # Build the semantic search index.
            embedding_model = create_embedding_model()

            vector_store = create_vector_store(
                chunks,
                embedding_model
            )

            # Build keyword-search documents.
            documents = [
                Document(
                    page_content=chunk["text"],
                    metadata={
                        "page_number": chunk["page_number"]
                    }
                )
                for chunk in chunks
            ]

            keyword_retriever = BM25Retriever.from_documents(
                documents,
                k=3
            )

            semantic_retriever = vector_store.as_retriever(
                search_kwargs={"k": 3}
            )

            # Combine keyword and semantic retrieval.
            hybrid_retriever = EnsembleRetriever(
                retrievers=[
                    keyword_retriever,
                    semantic_retriever
                ],
                weights=[0.5, 0.5]
            )

        # Save the completed pipeline only after processing succeeds.
        st.session_state.retriever = hybrid_retriever
        st.session_state.pdf_name = uploaded_file.name
        st.session_state.page_count = len(pages)
        st.session_state.chunk_count = len(chunks)
        st.session_state.chat_history = []
        st.session_state.pending_question = None

        st.success("🎉 Your PDF is ready!")

        return True

    except Exception as error:
        st.error(f"Could not process this PDF: {error}")
        return False

    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)


# --------------------------------------------------
# 4. GENERATE AN ANSWER
# --------------------------------------------------

def answer_question(question):
    """Retrieve source passages and generate a grounded answer."""

    if st.session_state.retriever is None:
        st.warning("Please upload and process a PDF first.")
        return False

    try:
        with st.spinner("🔎 Searching your document..."):
            retrieved_documents = (
                st.session_state.retriever.invoke(question)
            )[:3]

            answer = generate_answer(
                question,
                retrieved_documents
            )

        sources = []

        for document in retrieved_documents:
            sources.append({
                "text": document.page_content,
                "pages": document.metadata.get(
                    "page_number", []
                )
            })

        st.session_state.chat_history.append({
            "question": question,
            "answer": answer,
            "sources": sources
        })

        return True

    except Exception as error:
        st.error(f"Could not generate an answer: {error}")
        return False


# --------------------------------------------------
# 5. WELCOME HEADER
# --------------------------------------------------

st.title("📚 PDF-RAG Assistant")

st.markdown(
    "### 👋 Welcome! Let's explore your documents together."
)

st.write(
    "Upload a PDF, ask questions in your own words, "
    "and discover answers supported by your document."
)

st.caption(
    "🔍 Smart search  ·  🤖 AI-powered answers  ·  📌 Page references"
)

st.divider()


# --------------------------------------------------
# 6. CENTERED UPLOAD SCREEN
# --------------------------------------------------

if st.session_state.retriever is None:

    left_space, center, right_space = st.columns([1, 2, 1])

    with center:
        with st.container(border=True):
            st.subheader("✨ Start with your PDF")

            st.write(
                "Your document is the starting point. "
                "Upload it below to begin asking questions."
            )

            st.markdown("## 📤")

            uploaded_file = st.file_uploader(
                "Choose a PDF from your computer",
                type=["pdf"],
                help="Select a text-based PDF to get started.",
                key="welcome_pdf"
            )

            if uploaded_file is not None:
                st.info(f"📄 Selected: {uploaded_file.name}")

            if st.button(
                "🚀 Upload & Process PDF",
                type="primary",
                use_container_width=True
            ):
                if uploaded_file is None:
                    st.warning("Please choose a PDF first.")
                else:
                    if process_pdf(uploaded_file):
                        st.rerun()

    st.write("")

    st.subheader("💫 What you can do")

    feature1, feature2, feature3 = st.columns(3)

    with feature1:
        with st.container(border=True):
            st.markdown("### 📖")
            st.markdown("**Understand**")
            st.write(
                "Explore key ideas and find information "
                "without reading every page."
            )

    with feature2:
        with st.container(border=True):
            st.markdown("### 💬")
            st.markdown("**Ask questions**")
            st.write(
                "Ask questions naturally and get answers "
                "based on your uploaded document."
            )

    with feature3:
        with st.container(border=True):
            st.markdown("### 📌")
            st.markdown("**Check sources**")
            st.write(
                "Inspect retrieved passages and their "
                "original PDF page references."
            )

    st.stop()


# --------------------------------------------------
# 7. SIDEBAR: DOCUMENT CONTROLS
# --------------------------------------------------

with st.sidebar:
    st.title("📚 Your Workspace")

    st.success("Document processed")

    st.write(f"**Current PDF:** {st.session_state.pdf_name}")

    st.metric(
        "Pages with extracted text",
        st.session_state.page_count
    )

    st.metric(
        "Text chunks",
        st.session_state.chunk_count
    )

    st.divider()

    with st.expander("📂 Upload a different PDF"):
        replacement_file = st.file_uploader(
            "Choose another PDF",
            type=["pdf"],
            key="replacement_pdf"
        )

        if st.button(
            "Process new PDF",
            use_container_width=True
        ):
            if replacement_file is None:
                st.warning("Please choose a PDF first.")
            elif process_pdf(replacement_file):
                st.rerun()

    if st.button(
        "🗑️ Clear conversation",
        use_container_width=True
    ):
        st.session_state.chat_history = []
        st.session_state.pending_question = None
        st.rerun()


# --------------------------------------------------
# 8. DOCUMENT DASHBOARD
# --------------------------------------------------

st.subheader("📄 Your document is ready")

st.info(
    f"Currently exploring: **{st.session_state.pdf_name}**"
)

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric(
        "📑 PDF pages",
        st.session_state.page_count
    )

with metric2:
    st.metric(
        "🧩 Text chunks",
        st.session_state.chunk_count
    )

with metric3:
    st.metric(
        "💬 Questions asked",
        len(st.session_state.chat_history)
    )


# --------------------------------------------------
# 9. INTERACTIVE SUGGESTED QUESTIONS
# --------------------------------------------------

st.divider()

st.subheader("💡 Need an idea? Try asking")

suggestions = [
    "What is this document about?",
    "Summarise the main concepts.",
    "Which important points should I remember?"
]

suggestion_columns = st.columns(3)

for index, suggestion in enumerate(suggestions):
    with suggestion_columns[index]:
        if st.button(
            suggestion,
            key=f"suggestion_{index}",
            use_container_width=True
        ):
            st.session_state.pending_question = suggestion


# --------------------------------------------------
# 10. CHAT HISTORY AND SOURCE PASSAGES
# --------------------------------------------------

st.divider()

st.subheader("💬 Your conversation")

if not st.session_state.chat_history:
    st.info(
        "👋 Your conversation starts here. "
        "Ask a question below or choose a suggestion above."
    )

for chat in st.session_state.chat_history:

    with st.chat_message("user"):
        st.write(chat["question"])

    with st.chat_message("assistant", avatar="📚"):
        st.markdown(chat["answer"])

        with st.expander("📌 Explore retrieved source passages"):
            if not chat["sources"]:
                st.write("No source passages were retrieved.")

            for index, source in enumerate(
                chat["sources"],
                start=1
            ):
                pages = source["pages"]

                if isinstance(pages, int):
                    pages = [pages]

                page_labels = ", ".join(
                    str(page) for page in pages
                ) or "Unknown"

                st.markdown(
                    f"**Source {index} — PDF page(s): "
                    f"{page_labels}**"
                )

                st.write(source["text"])
                st.divider()


# --------------------------------------------------
# 11. CHAT INPUT
# --------------------------------------------------

question_from_chat = st.chat_input(
    "Ask anything about your PDF..."
)

question_to_answer = None

if st.session_state.pending_question:
    question_to_answer = st.session_state.pending_question
    st.session_state.pending_question = None

elif question_from_chat:
    question_to_answer = question_from_chat.strip()

if question_to_answer:
    if answer_question(question_to_answer):
        st.rerun()
