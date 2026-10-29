# AI Document Q&A Assistant

A local AI-powered application that allows users to upload PDF documents, ask questions about their content, and receive answers supported by document citations.

The project uses a local Large Language Model (LLM) through Ollama, semantic search with FAISS, and a Streamlit interface to make document question-answering simple and accessible.

## Features

- **PDF Upload:** Upload PDF documents for analysis.
- **Text Extraction:** Extract text and page information using PyMuPDF.
- **Text Chunking:** Split documents into smaller overlapping chunks for retrieval.
- **Semantic Search:** Use Sentence Transformers embeddings and FAISS to retrieve relevant passages.
- **AI-Powered Answers:** Generate answers using a locally running Ollama model.
- **Source Citations:** Display document names, page numbers, and cited passages.
- **Citation Verification:** Use a second LLM call to check whether the answer is supported by the cited evidence.
- **Document Summarization:** Select passages from across a document to help summarize its overall content.
- **Chat Interface:** Ask follow-up questions through a Streamlit application.
- **Local Processing:** Run the language model locally without requiring a paid LLM API.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Application logic |
| Streamlit | User interface |
| Ollama | Local LLM inference |
| Qwen 2.5 3B | Answer generation and citation checking |
| Sentence Transformers | Text embeddings |
| `all-MiniLM-L6-v2` | Embedding model |
| FAISS | Vector similarity search |
| PyMuPDF | PDF text extraction |
| NumPy | Numerical operations |

## How It Works

1. The user uploads a PDF document.
2. PyMuPDF extracts text and page information.
3. The text is divided into overlapping chunks.
4. Sentence Transformers converts the chunks into embeddings.
5. FAISS retrieves relevant passages for a user's question.
6. Ollama generates an answer using the retrieved evidence.
7. The application displays citations and source passages.
8. A second LLM call checks whether the cited evidence supports the answer.

For summary requests, the application selects passages distributed across the document rather than relying only on the most similar passages.

## Installation and Setup

### Prerequisites

- Python 3.11
- Git
- Ollama installed on your computer

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/ai-document-qa.git
cd ai-document-qa
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

If a `requirements.txt` file is included:

```bash
pip install -r requirements.txt
```

### 4. Download the Ollama model

```bash
ollama pull qwen2.5:3b
```

Make sure Ollama is running before launching the application.

### 5. Run the application

```bash
streamlit run app.py
```

Replace `app.py` with your actual Python entry-point filename if it is different.

## Example Use Cases

- Ask questions about research papers.
- Summarize lengthy PDF documents.
- Find relevant information without manually searching every page.
- Review answers alongside their supporting passages.
- Explore technical documents using natural-language questions.

## Limitations

- Answer quality depends on the document content, retrieved passages, and local LLM.
- Citation verification is automated screening, not a guarantee of correctness.
- Scanned PDFs may require OCR if they do not contain extractable text.
- Local model performance depends on available system resources.
- Document summaries may omit details that are not present in the selected passages.

## Future Improvements

- Improve retrieval and summary quality.
- Add support for multiple document formats.
- Improve citation verification and reduce false positives.
- Add document-level metadata filtering.
- Add evaluation metrics for retrieval accuracy and answer faithfulness.

## Project Status

Developed as an AI/ML project to explore local language models, semantic retrieval, document question-answering, and evidence-based responses.

## License

Choose an appropriate open-source license before distributing the project. For example, you may use the MIT License by adding a `LICENSE` file.
