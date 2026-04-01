import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# ingest.py ile tutarlı olması için kütüphaneleri aynı şekilde import ediyoruz:
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama

def format_docs(docs):
    # Dönen Document objelerinin içindeki metinleri (page_content) birleştirip tek bir string (metin) yapar
    return "\n\n".join(doc.page_content for doc in docs)

def inspect(state):
    print("----- RETRIEVED CONTEXT -----")
    print(state["context"][:500] + "... (devamı var)") # sadece ilk 500 karakteri görelim
    print("-----------------------------")
    return state

def start_chat():
    print("--- Modeller ve Veritabanı Hazırlanıyor ---")
    
    # 1. Embedding ve DB Yükle
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_db = Chroma(
        persist_directory="../chromadb_storage",
        embedding_function=embeddings
    )
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})

    # 2. Model ve Prompt Hazırla
    model = ChatOllama(model="llama3", temperature=0)
    
    template = """Sana verilen bağlam (context) bilgilerine dayanarak soruyu cevapla. 
    Eğer cevap dökümanda yoksa 'Bilmiyorum' de.
    
    Bağlam: {context}
    Soru: {question}
    
    Cevap:"""
    prompt = ChatPromptTemplate.from_template(template)

    # 3. Modern RAG Zinciri (LCEL)
    print(retriever)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RunnableLambda(inspect)
        | prompt
        | model
        | StrOutputParser()
    )

    print("--- Chatbot Hazır! (Çıkmak için 'exit' yazın) ---")
    
    while True:
        user_input = input("\nSiber Güvenlik Sorun: ")
        if user_input.lower() == "exit":
            break
            
        print("\nDüşünülüyor...")
        for chunk in rag_chain.stream(user_input):
            print(chunk, end="", flush=True)
        print("\n")

if __name__ == "__main__":
    start_chat()