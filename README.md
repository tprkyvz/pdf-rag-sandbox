# PDF RAG Sandbox 

This project is a local RAG (Retrieval-Augmented Generation) chatbot developed as help to "Agentic SOC" project. It allows you to chat with your local PDF documents (cybersecurity articles, reports, etc.) using Llama 3 and ChromaDB.

## 🛠 Tech Stack
- **LLM:** Ollama (Llama 3)
- **Embedding Model:** nomic-embed-text (768 dimensions)
- **Vector Database:** ChromaDB
- **Orchestration:** LangChain (LCEL)
- **Operating System:** CachyOS (Arch Linux based)
- **Hardware:** NVIDIA RTX 3050 (4GB VRAM)

## 📁 Project Structure
- `scripts/ingest.py`: Reads PDFs, splits them into chunks, and saves embeddings to ChromaDB.
- `scripts/chat.py`: The chatbot interface that performs similarity search and generation.
- `data/papers/`: Directory to store your PDF files.
- `chromadb_storage/`: Local persistence directory for the vector database.

## 🚀 Getting Started

### Prerequisites
- Install [Ollama](https://ollama.ai/)
- Pull required models:
  ```bash
  ollama pull llama3
  ollama pull nomic-embed-text

### Installation
- 1.Clone this repository
  ```bash
  git clone [https://github.com/tprkyvz/pdf-rag-sandbox.git] 
  cd pdf-rag-sandbox(https://github.com/tprkyvz/pdf-rag-sandbox.git)

- 2.Create and activate virtual environment:
  ```bash
  python -m venv .venv
  source .venv/bin/activate

- 3.Install Dependencies:
  ```bash
  pip install -r requirements.txt

### Usage

- 1.Place your PDFs in `data/papers`.

- 2.Run the ingestion script:
  ```bash
  python scripts/ingest.py

- 3.Start chatting:
  ```bash
  python scripts/chat.py
