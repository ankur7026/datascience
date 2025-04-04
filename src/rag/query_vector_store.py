import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

# Path to FAISS vector store
VECTOR_STORE_PATH = "models/faiss_index"

def load_vector_store():
    """Loads stored FAISS vector embeddings."""
    if os.path.exists(VECTOR_STORE_PATH):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization = True)
    else:
        raise FileNotFoundError("❌ Vector store not found. Run store_vectors first.")

def ask_question(query):
    """Handles question answering using FAISS and Ollama."""
    vector_store = load_vector_store()

    # Create a retriever (fetches top-k most relevant documents)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})  

    # Load Ollama (Local Model Running on Port 11434)
    llm = Ollama(model="llama3.2:latest", base_url="http://localhost:11434")

    # Setup LangChain Retrieval-based QA chain
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

    # Run query through the QA chain
    response = qa_chain.run(query)
    return response

# Example usage
if __name__ == "__main__":
    while True:
        user_query = input("\nAsk a question (or type 'exit' to quit): ")
        if user_query.lower() == "exit":
            break
        answer = ask_question(user_query)
        print("\n🤖 AI Answer:", answer)
