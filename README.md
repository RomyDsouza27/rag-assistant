# RAG Assistant
### Detection and Mitigation of Hallucinations in Retrieval-Augmented Generation Systems

RAG Assistant is a Python-based Retrieval-Augmented Generation (RAG) application designed to answer questions using information retrieved from research documents. It combines document processing, semantic search, vector databases, and Large Language Models (LLMs) to generate context-based responses.

The project also incorporates a preliminary hallucination detection mechanism to identify responses that may not be sufficiently supported by the retrieved context.

---

## Project Overview

Large Language Models can sometimes generate incorrect, misleading, or unsupported information, commonly known as hallucinations.

RAG Assistant aims to address this challenge by retrieving relevant information from source documents before generating an answer. It also evaluates the relationship between the generated response and retrieved context to flag potentially unsupported answers.

**Main objective:** To develop a document-based question-answering system that improves response grounding, transparency, and reliability through retrieval and preliminary hallucination checking.

## Key Features

- **PDF Document Ingestion:** Extracts text from research documents.
- **Text Processing and Chunking:** Divides extracted content into smaller sections for efficient retrieval.
- **Semantic Embeddings:** Converts text chunks into numerical vector representations.
- **FAISS Vector Search:** Stores and retrieves relevant document chunks using similarity search.
- **Retrieval-Augmented Generation:** Uses retrieved passages as context for generating answers.
- **Hallucination Detection:** Applies a preliminary heuristic mechanism to flag potentially unsupported responses.
- **Multi-Document Support:** Designed to retrieve information from multiple indexed research documents.
- **Evaluation and Testing:** Includes separate test scripts for embeddings, retrieval, ingestion, and hallucination checking.

## System Architecture

```text
        Research PDF Documents
                 |
                 v
          PDF Ingestion
                 |
                 v
        Text Extraction
                 |
                 v
          Text Chunking
                 |
                 v
       Sentence Transformers
          (Embeddings)
                 |
                 v
          FAISS Index
                 |
                 v
           User Query
                 |
                 v
        Semantic Retrieval
                 |
                 v
       Relevant Text Chunks
                 |
                 v
       Large Language Model
                 |
                 v
        Generated Answer
                 |
                 v
       Hallucination Checker
                 |
                 v
        Final Response
```

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Application interface |
| Sentence Transformers | Text embedding generation |
| all-MiniLM-L6-v2 | Sentence embedding model |
| FAISS | Vector storage and similarity search |
| Ollama | Local LLM integration, when configured |
| PyPDF / PDF extraction libraries | Document processing |
| NumPy | Numerical operations |
| Pytest / Python testing scripts | Component testing |

## Project Structure

```text
RAG_Project/
│
├── app.py
├── rag_pipeline.py
├── build_index.py
├── check_index.py
├── check_pdf.py
├── paper_list.txt
├── requirements.txt
│
├── src/
│   ├── __init__.py
│   ├── embeddings.py
│   ├── evaluation.py
│   ├── generator.py
│   ├── hallucination.py
│   ├── ingestion.py
│   ├── pdf_visual_extractor.py
│   ├── retriever.py
│   └── vector_store.py
│
├── test_embedding.py
├── test_hallucination.py
├── test_ingestion.py
├── test_multi_paper.py
├── test_pipeline.py
├── test_references.py
└── test_retrieval.py
```

## How It Works

### 1. Document Ingestion
The system accepts research documents in PDF format and extracts their textual content for further processing.

### 2. Text Chunking
Extracted text is divided into smaller chunks. This allows the retrieval system to identify relevant sections instead of processing entire documents at once.

### 3. Embedding Generation
The Sentence Transformers model `all-MiniLM-L6-v2` converts text chunks into numerical vector embeddings that capture semantic relationships.

### 4. Vector Storage and Retrieval
FAISS is used to index document embeddings and perform similarity searches. When a user submits a question, the retriever searches for the most relevant document chunks.

### 5. Answer Generation
The retrieved passages are supplied as context to the configured language model. The model uses this information to generate a response to the user's question.

### 6. Hallucination Detection
A preliminary heuristic checker examines the generated answer against the retrieved context and flags potentially unsupported responses.

**Note:** The heuristic checker is an initial detection mechanism. It does not guarantee factual correctness or completely eliminate hallucinations.

## Installation and Setup

### Prerequisites

- Python 3.10 or a compatible version
- Git
- Visual Studio Code (recommended)
- Ollama and a compatible local model, if using local LLM generation

### Step 1: Clone the Repository

```bash
git clone https://github.com/RomyDsouza27/rag-assistant.git
cd rag-assistant
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Add Research Documents

Place the PDF documents you are authorized to process in the project's configured document directory.

### Step 5: Build the Vector Index

```bash
python build_index.py
```

This prepares the document embeddings and vector index according to the project's indexing implementation.

### Step 6: Run the Application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal to access the application.

*Note: The exact setup may depend on the configured LLM, environment variables, and document paths in your local project.*

## Testing

The repository includes test scripts for different components:

```bash
python test_embedding.py
python test_ingestion.py
python test_retrieval.py
python test_hallucination.py
python test_multi_paper.py
python test_pipeline.py
python test_references.py
```

These scripts can be used to examine individual components of the RAG pipeline. Run them in your configured Python environment.

## AI Use Case

**Use case: Research Paper Question Answering**

A researcher uploads or indexes academic papers and asks:

"What limitations of Retrieval-Augmented Generation systems are discussed in the research papers?"

The system retrieves relevant passages from the indexed documents, supplies them to the language model, and generates a context-based answer. The hallucination checker then examines whether the response has textual support in the retrieved passages.

This helps users review research information with greater transparency and encourages verification against original sources.

## Limitations

- Retrieval quality depends on document quality, chunking strategy, and embedding performance.
- The generated answer may still contain unsupported or inaccurate information.
- The heuristic hallucination checker is not a definitive factual verification system.
- Answer quality depends on the configured language model.
- Complex tables, figures, and scanned PDF content may require additional extraction techniques.

## Future Scope

- Integrate advanced hallucination detection models.
- Implement semantic entailment-based answer verification.
- Introduce source-level citations for generated responses.
- Evaluate retrieval performance using Precision@K, Recall@K, and Mean Reciprocal Rank.
- Evaluate answer quality using faithfulness and answer relevance metrics.
- Add confidence calibration and improved uncertainty estimation.
- Support additional document formats.
- Improve the user interface and document management capabilities.

## Research Focus

This project explores the intersection of:

- Retrieval-Augmented Generation (RAG)
- Large Language Models (LLMs)
- Natural Language Processing (NLP)
- Vector Databases
- Semantic Search
- Explainable and Trustworthy AI
- Hallucination Detection and Mitigation

## Author

**Romy D'Souza**

M.Sc. Data Science

GitHub: [@RomyDsouza27](https://github.com/RomyDsouza27)

## Repository

[https://github.com/RomyDsouza27/rag-assistant](https://github.com/RomyDsouza27/rag-assistant)

---

*Academic project focused on improving the reliability and transparency of Retrieval-Augmented Generation systems.*