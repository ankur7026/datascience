import os
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import pandas as pd

VECTOR_STORE_PATH = "models/faiss_index"

def store_vectors(csv_path):
    """Stores vector embeddings for given texts."""
    load_dotenv()

    # Convert relative path to absolute
    csv_path = os.path.abspath(csv_path)

    print(csv_path)

    # Check if the file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ CSV file not found: {csv_path}")

    # Load dataset
    df = pd.read_csv(csv_path)

    # Ensure required columns exist
    required_columns = {"hotel", "adr", "country"}
    missing_cols = required_columns - set(df.columns)
    if missing_cols:
        raise ValueError(f"❌ Missing columns in CSV: {missing_cols}")

    # Convert dataset to text format (concatenating important columns)
    texts = df.apply(lambda row: f"Hotel: {row['hotel']}, Revenue: {row['adr']}, Country: {row['country']}", axis=1).tolist()

    # Ensure "models" directory exists
    os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)

    # Use HuggingFace Embeddings (local)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Store vectors
    vector_store = FAISS.from_texts(texts, embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)
    print(f"✅ Vectors stored successfully at {VECTOR_STORE_PATH}")

def load_vector_store():
    """Loads stored vector embeddings from FAISS."""
    if os.path.exists(VECTOR_STORE_PATH):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.load_local(VECTOR_STORE_PATH, embeddings)
    else:
        raise FileNotFoundError("❌ Vector store not found. Run store_vectors first.")

# Run vector storage
store_vectors("/Users/ankur/Documents/python_rapos_solvei8/datascience/data/hoteldata.csv")  # Adjust if needed
