import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma

def run_ingestion():
    # 1. PDF Yükleme
    pdf_path = "../data/papers/siber_guvenlik_nedir.pdf" # PDF dosyanın adı
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(docs)

    
    print(f"{len(chunks)} parça veritabanına ekleniyor...")
    
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model="nomic-embed-text"), # Ollama'da llama3'ün kurulu olduğundan emin ol
        persist_directory="../chromadb_storage"
    )
    
    print("İşlem tamamlandı. Veritabanı oluşturuldu.")

if __name__ == "__main__":
    run_ingestion()