import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Load environment variables (if any)
load_dotenv()

# Define paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

def load_documents_from_txt(data_dir):
    """
    Reads all .txt files from the data directory and returns a list of LangChain Document objects.
    """
    documents = []
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' does not exist.")
        return documents
        
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_dir, filename)
            print(f"Loading: {file_path}")
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    
                # Create a LangChain Document object
                doc = Document(
                    page_content=content,
                    metadata={"source": filename}
                )
                documents.append(doc)
            except Exception as e:
                print(f"Failed to read {filename}: {e}")
                
    return documents

def main():
    print("--- Starting HandyRAG Ingestion ---")
    
    # 1. Load documents
    raw_docs = load_documents_from_txt(DATA_DIR)
    if not raw_docs:
        print("No documents found. Ingestion aborted.")
        return
        
    # 2. Split documents into chunks
    # We use a chunk size of 800 characters with 100 character overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
    )
    
    print("Splitting documents into chunks...")
    chunks = text_splitter.split_documents(raw_docs)
    print(f"Created {len(chunks)} chunks from {len(raw_docs)} documents.")
    
    # 3. Create Hugging Face embeddings model
    # This uses sentence-transformers/all-MiniLM-L6-v2 by default, which runs locally.
    print("Initializing HuggingFace Embeddings model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # 4. Save chunks to local Chroma DB
    print(f"Saving vectors to local database at '{DB_DIR}'...")
    
    # Clean up existing database if it exists to prevent duplicate accumulation
    # in local runs
    if os.path.exists(DB_DIR):
        print("Existing database found. Replacing with fresh ingestion.")
        # We can let Chroma handle it, but deleting the folder ensures a clean state.
        import shutil
        try:
            shutil.rmtree(DB_DIR)
        except Exception as e:
            print(f"Warning: Could not clear directory '{DB_DIR}': {e}")
            
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    
    # Persist the database
    vector_db.persist()
    print("--- Ingestion Completed Successfully! ---")

if __name__ == "__main__":
    main()
