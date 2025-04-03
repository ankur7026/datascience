from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import os

VECTOR_STORE_PATH = "models/faiss_index"

def inspect_faiss():
    """Loads FAISS index and prints stored data."""
    
    # Check if FAISS index exists
    if not os.path.exists(VECTOR_STORE_PATH):
        raise FileNotFoundError("❌ FAISS index not found! Run store_vectors first.")
    
    # Load FAISS index
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization = True)
    
    # Get FAISS index object
    faiss_index = vector_store.index

    # Get number of stored vectors
    num_vectors = faiss_index.ntotal
    print(f"✅ FAISS Index Loaded. Total Vectors: {num_vectors}")

    # Retrieve stored text data
    if vector_store.docstore._dict:
        stored_texts = list(vector_store.docstore._dict.values())
        print("\n🔹 Sample Stored Texts:")
        for i, text in enumerate(stored_texts[:5]):  # Show only first 5 entries
            print(f"{i+1}. {text}")

    # Inspect vectors (optional)
    print("\n🔹 Sample Vectors (First 2):")
    vectors = faiss_index.reconstruct_n(0, min(2, num_vectors))  # Get first 2 vectors
    print(vectors)

# Run the function
inspect_faiss()
