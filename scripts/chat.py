import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def start_chat():
    print("--- Modeller ve Veritabanı Hazırlanıyor ---")
    
    # 1. Embedding ve DB Yükle
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_db = Chroma(
        persist_directory="../chromadb_storage",
        embedding_function=embeddings
    )
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # 2. Model ve Prompt Hazırla
    model = ChatOllama(model="llama3", temperature=0)
    
    template = """Sana verilen bağlam (context) bilgilerine dayanarak soruyu cevapla. 
    Eğer cevap dökümanda yoksa 'Bilmiyorum' de.
    
    Bağlam: {context}
    Soru: {question}
    
    Cevap:"""
    prompt = ChatPromptTemplate.from_template(template)

    # 3. Modern RAG Zinciri (LCEL)
    # Bu yapı "langchain.chains" bağımlılığını ortadan kaldırır
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
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