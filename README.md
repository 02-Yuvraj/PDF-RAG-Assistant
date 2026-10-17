
# 📚 PDF-RAG Assistant

### Ask questions about your PDFs and get AI-generated answers grounded in your documents.

PDF-RAG Assistant is an interactive document question-answering application built with Python, LangChain, and Retrieval-Augmented Generation (RAG). It lets users upload a PDF, ask questions in natural language, and receive answers based on relevant passages retrieved from the document.

The application combines **BM25 keyword retrieval** with **FAISS-based semantic search**, Hugging Face embeddings, and a Groq-hosted language model to make document exploration more convenient and traceable.

---

## ✨ Features

- 📄 **PDF Upload:** Upload text-based PDF documents through a Streamlit interface.
- 🔍 **Hybrid Retrieval:** Combines BM25 keyword search and semantic similarity search.
- 🧠 **Hugging Face Embeddings:** Converts text into vector representations using `all-MiniLM-L6-v2`.
- ⚡ **Groq LLM Integration:** Generates answers using the `openai/gpt-oss-20b` model through Groq.
- 📌 **Source References:** Displays retrieved passages and their original PDF page numbers.
- 💬 **Interactive Q&A:** Ask follow-up questions and review conversation history during the session.
- 🧩 **Text Chunking:** Splits extracted PDF text into overlapping chunks for retrieval.
- 🛡️ **Evidence-Grounded Responses:** Instructs the model to answer using retrieved passages and acknowledge insufficient evidence.

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Frontend | Streamlit |
| RAG orchestration | LangChain |
| PDF text extraction | PyPDF |
| Text splitting | RecursiveCharacterTextSplitter |
| Embeddings | Hugging Face Sentence Transformers |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | FAISS |
| Keyword retrieval | BM25 |
| Hybrid retrieval | LangChain EnsembleRetriever |
| Language model | Groq API — `openai/gpt-oss-20b` |
| Environment variables | python-dotenv |

## 🏗️ Architecture

```text
             ┌─────────────────────┐
             │   Upload PDF         │
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │  Extract PDF Text   │
             │       (PyPDF)       │
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │   Text Chunking     │
             │  Overlapping Chunks │
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │  Build Retrievers   │
             │                     │
             │ BM25 + FAISS Search │
             └──────────┬──────────┘
                        │
             User Question
                        │
                        ▼
             ┌─────────────────────┐
             │   Hybrid Retrieval  │
             │ Relevant Passages   │
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │     Groq LLM        │
             │ Answer Generation   │
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │ Answer + Sources    │
             │   in Streamlit      │
             └─────────────────────┘
```

## 📂 Project Structure

```text
PDF-RAG-Assistant/
│
├── app.py
├── src/
│   ├── pdf_processor.py
│   ├── text_chunker.py
│   ├── embedding_generator.py
│   ├── vector_store.py
│   └── answer_generator.py
│
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
└── sample.pdf
```

Additional testing and debugging scripts may also be present in the repository.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/02-Yuvraj/PDF-RAG-Assistant.git
cd PDF-RAG-Assistant
```

### 2. Create a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use the environment's Python executable directly instead.

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure your Groq API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get an API key from the Groq Console:

https://console.groq.com/

**Security:** Never commit your `.env` file or publish your API key.

### 5. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal, usually:

http://localhost:8501

## 🚀 How to Use

1. Launch the Streamlit application.
2. Upload a text-based PDF.
3. Click **Upload & Process PDF**.
4. Wait for the text extraction, embedding generation, and retrieval setup to finish.
5. Enter a question about the uploaded document or choose a suggested question.
6. Read the generated answer.
7. Expand the source section to inspect the retrieved passages and PDF page references.

## 🔎 Why Hybrid Retrieval?

Traditional keyword search works well when a question contains terminology that also appears in the document. Semantic search can find passages that express related ideas using different wording.

This project combines both approaches:

- **BM25:** Retrieves documents based on keyword relevance.
- **FAISS:** Supports vector similarity search over document embeddings.
- **EnsembleRetriever:** Combines the two retrieval methods with configured weights.

The current implementation assigns equal weights to keyword and semantic retrieval.

## ⚠️ Limitations

- Scanned or image-only PDFs may require OCR before text can be extracted.
- Answer quality depends on the PDF's text quality and the relevance of retrieved passages.
- The model may still produce incorrect or unsupported statements; source references should be checked.
- API usage is subject to Groq's availability, rate limits, and account policies.
- The embedding model may need to download on the first run.
- The application processes a document in the current session; persistent document storage is not implemented.

## 🔮 Potential Improvements

- Add OCR support for scanned PDFs.
- Improve retrieval using reranking and retrieval evaluation.
- Add multi-PDF search.
- Add persistent chat history and document storage.
- Add configurable retrieval settings and source relevance scores.
- Deploy the application to a suitable hosting platform.

## 👨‍💻 Author

**Yuvraj Singh**

- GitHub: [02-Yuvraj](https://github.com/02-Yuvraj)
- LinkedIn: [Yuvraj Singh](https://linkedin.com/in/yuvraj-singh-37833929b)

---

If you find this project useful, consider giving the repository a ⭐.
