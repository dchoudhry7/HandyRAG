import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import sys

# Streamlit Cloud fix for ChromaDB requiring sqlite3 >= 3.35.0
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import pypdf
import streamlit as st
from dotenv import load_dotenv

# Import LangChain and Chroma components
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables from .env
load_dotenv()

# Define Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "chroma_db")
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")

# Make sure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Page configuration
st.set_page_config(
    page_title="HandyRAG - DIY Safety Advisor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom minimalistic styling
st.markdown("""
<style>
    .reportview-container {
        background: #111;
    }
    .sidebar .sidebar-content {
        background: #1e1e1e;
    }
    h1, h2, h3 {
        font-weight: 700;
        color: #f8f9fa;
    }
    .stButton>button {
        border-radius: 6px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_db_and_llm():
    """
    Initializes and caches the local vector database, embeddings, and Groq LLM.
    """
    # 1. Initialize embeddings (local sentence-transformer model)
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
    except Exception as e:
        st.error(f"Error loading embedding model: {e}")
        embeddings = None

    # 2. Initialize Chroma Vector Database
    vector_db = None
    if os.path.exists(DB_DIR):
        try:
            vector_db = Chroma(
                persist_directory=DB_DIR,
                embedding_function=embeddings
            )
        except Exception as e:
            st.error(f"Error loading Chroma DB: {e}")
    else:
        try:
            os.makedirs(DB_DIR, exist_ok=True)
            vector_db = Chroma(
                persist_directory=DB_DIR,
                embedding_function=embeddings
            )
        except Exception as e:
            st.error(f"Error creating empty Chroma DB: {e}")

    # 3. Initialize Groq Chat Model
    llm = None
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.warning("Warning: GROQ_API_KEY is not set in environment variables.")
    else:
        try:
            llm = ChatGroq(
                temperature=0.2,
                model_name="llama-3.1-8b-instant",
                api_key=api_key
            )
        except Exception as e:
            st.error(f"Error initializing ChatGroq: {e}")
            
    return vector_db, llm, embeddings

def get_documents():
    """
    Returns a list of all active documents in data/ and uploads/ folders.
    """
    docs = []
    
    # 1. Read files from data/
    if os.path.exists(DATA_DIR):
        for filename in os.listdir(DATA_DIR):
            if filename.endswith(".txt"):
                category = "General"
                if "drywall" in filename.lower():
                    category = "Drywall"
                elif "electrical" in filename.lower():
                    category = "Electrical"
                elif "plumbing" in filename.lower():
                    category = "Plumbing"
                
                docs.append({
                    "name": filename,
                    "type": "txt",
                    "category": category,
                    "size": os.path.getsize(os.path.join(DATA_DIR, filename))
                })
                
    # 2. Read files from uploads/
    if os.path.exists(UPLOADS_DIR):
        for filename in os.listdir(UPLOADS_DIR):
            if filename.endswith(".pdf"):
                display_name = filename
                category = "General"
                if "__" in filename:
                    parts = filename.split("__", 1)
                    category = parts[0]
                    display_name = parts[1]
                
                docs.append({
                    "name": display_name,
                    "raw_name": filename,
                    "type": "pdf",
                    "category": category,
                    "size": os.path.getsize(os.path.join(UPLOADS_DIR, filename))
                })
                
    return docs

def delete_document_from_db_and_disk(raw_name, category, db):
    """
    Removes a document from the Chroma vector database index and deletes the file from uploads/.
    """
    # 1. Delete associated chunks from Chroma
    if db is not None:
        try:
            results = db.get(where={"source": raw_name})
            ids = results.get("ids", [])
            if ids:
                db.delete(ids)
                if hasattr(db, "persist"):
                    db.persist()
        except Exception as e:
            st.error(f"Error removing chunks from vector database: {e}")
            return False

    # 2. Delete file from uploads directory
    file_path = os.path.join(UPLOADS_DIR, f"{category}__{raw_name}")
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            st.error(f"Error deleting file from disk: {e}")
            return False
    else:
        # Check if the filename itself is directly in uploads/
        file_path_direct = os.path.join(UPLOADS_DIR, raw_name)
        if os.path.exists(file_path_direct):
            try:
                os.remove(file_path_direct)
            except Exception as e:
                st.error(f"Error deleting file from disk: {e}")
                return False
                
    return True

def classify_document_category(text, filename, model):
    """
    Uses Groq LLM to classify a document's category based on its text content and filename.
    """
    categories = ["Kitchen", "Plumbing", "Electrical", "Drywall", "Bathroom", "HVAC", "General"]
    if model is None:
        return "General"
        
    snippet = text[:1000]
    prompt = (
        f"Analyze the following document title and content snippet. "
        f"Classify it into exactly one of these categories: {', '.join(categories)}.\n\n"
        f"Document Title: {filename}\n"
        f"Content Snippet:\n{snippet}\n\n"
        f"Respond with ONLY the matched category name from the list and nothing else (no punctuation, no preamble, no explanations)."
    )
    try:
        response = model.invoke([("human", prompt)])
        result = response.content.strip()
        for cat in categories:
            if cat.lower() in result.lower():
                return cat
    except Exception as e:
        print(f"Error during category classification: {e}")
        
    return "General"

# Load resource cache
db, llm, embeddings = get_db_and_llm()

# SIDEBAR: PDF uploader & active documents listing with inline deletion
with st.sidebar:
    st.markdown("## 🛡️ HandyRAG")
    st.caption("DIY Safety Diagnostics Panel")
    st.markdown("---")
    
    # 1. PDF Ingestion form
    st.markdown("### 📥 Ingest Builder Guide")
    category_selection = st.selectbox(
        "Target Area Category", 
        ["Auto-Detect (AI)", "Kitchen", "Plumbing", "Electrical", "Drywall", "Bathroom", "HVAC", "General"],
        key="ingest_category_sidebar"
    )
    uploaded_file = st.file_uploader("Upload PDF Guide", type=["pdf"], key="pdf_uploader_sidebar")
    if st.button("Parse & Chunk PDF", use_container_width=True, key="btn_ingest_sidebar"):
        if uploaded_file is not None:
            with st.spinner("Ingesting guide..."):
                safe_filename = uploaded_file.name.replace(" ", "_")
                temp_filename = f"temp__{safe_filename}"
                file_path = os.path.join(UPLOADS_DIR, temp_filename)
                
                try:
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    reader = pypdf.PdfReader(file_path)
                    extracted_text = ""
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            extracted_text += page_text + "\n"
                    
                    if not extracted_text.strip():
                        st.error("The uploaded PDF has no extractable text.")
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    else:
                        # Auto-detect category using AI if requested
                        category = category_selection
                        if category == "Auto-Detect (AI)":
                            category = classify_document_category(extracted_text, uploaded_file.name, llm)
                            st.toast(f"AI classified category as: {category}")
                        
                        # Rename local file to match the assigned category prefix
                        new_filename = f"{category}__{safe_filename}"
                        new_file_path = os.path.join(UPLOADS_DIR, new_filename)
                        
                        try:
                            # If file already exists, remove it first to allow rename
                            if os.path.exists(new_file_path):
                                os.remove(new_file_path)
                            os.rename(file_path, new_file_path)
                            file_path = new_file_path
                        except Exception as rename_err:
                            st.warning(f"Could not rename file, keeping temp format: {rename_err}")
                            
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
                        chunks = text_splitter.split_text(extracted_text)
                        
                        documents = [
                            Document(page_content=chunk, metadata={"source": uploaded_file.name, "category": category})
                            for chunk in chunks
                        ]
                        
                        if db is not None:
                            db.add_documents(documents)
                            if hasattr(db, "persist"):
                                db.persist()
                            st.success(f"Successfully ingested {len(chunks)} chunks into **{category}**!")
                            st.rerun()
                        else:
                            st.error("Vector database is not initialized.")
                except Exception as e:
                    st.error(f"Error during ingestion: {e}")
                    if os.path.exists(file_path):
                        os.remove(file_path)
        else:
            st.warning("Please select a PDF file first.")
            
    st.markdown("---")
    
    # 2. Knowledge Library List with inline delete buttons
    st.markdown("### 📚 Knowledge Library")
    docs = get_documents()
    if not docs:
        st.info("No documents currently indexed.")
    else:
        categories = {}
        for doc in docs:
            cat = doc["category"]
            categories.setdefault(cat, []).append(doc)
            
        for cat, doc_list in categories.items():
            with st.expander(f"{cat} ({len(doc_list)})", expanded=False):
                for doc in doc_list:
                    doc_type = doc["type"]
                    size_kb = doc["size"] / 1024
                    
                    if doc_type == "pdf":
                        # For PDF files, add metadata and a small inline delete button
                        col_doc, col_del = st.columns([7, 3])
                        col_doc.markdown(f"📄 **{doc['name']}**  \n`PDF • {size_kb:.0f} KB`", help=doc["name"])
                        # Simple trashcan delete button next to the filename
                        if col_del.button("🗑️", key=f"del_sidebar_{doc['raw_name']}", use_container_width=True, help=f"Delete {doc['name']}"):
                            with st.spinner("Deleting..."):
                                success = delete_document_from_db_and_disk(doc['raw_name'], doc['category'], db)
                                if success:
                                    st.rerun()
                    else:
                        st.markdown(f"📝 **{doc['name']}**  \n`TXT • {size_kb:.1f} KB`")

# MAIN PAGE CONTENT: Single page chat
st.title("🛡️ HandyRAG Consultant")
st.caption("Llama 3.1 & MiniLM Embedding Agent • Safety-First Home Repair")
st.markdown("---")

# 1. Diagnostic Starter Suggestions (Always Visible at the top of the chat area)
st.markdown("##### 💡 Popular DIY Diagnostics Topics")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🔧 Kitchen Sink Clog\n\nClear pipes and check traps.", use_container_width=True, key="starter_clog_main"):
        st.session_state.starter_prompt = "My kitchen sink won't drain and is backed up. How do I fix it safely?"
        st.rerun()
with col2:
    if st.button("⚡ Change Outlet/Switch\n\nSafe shut-off and validation.", use_container_width=True, key="starter_outlet_main"):
        st.session_state.starter_prompt = "How do I safely replace a broken light switch or wall outlet?"
        st.rerun()
with col3:
    if st.button("🧱 Drywall Patching\n\nSanding, plastering & safety.", use_container_width=True, key="starter_drywall_main"):
        st.session_state.starter_prompt = "There is a small crack and hole in my drywall. What is the process to patch and paint it?"
        st.rerun()
        
st.markdown("---")

# Session state for message storage
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption(f"Sources: {', '.join(msg['sources'])}")

# Check for input from starter prompt or chat input box
user_query = None
if "starter_prompt" in st.session_state and st.session_state.starter_prompt:
    user_query = st.session_state.starter_prompt
    del st.session_state.starter_prompt  # clear trigger

chat_input_val = st.chat_input("Ask HandyRAG a question...")
if chat_input_val:
    user_query = chat_input_val

# Process query if entered
if user_query:
    # 1. Add and display user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
        
    # 2. Query RAG backend & display response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing guides and preparing instructions..."):
            if not llm:
                st.error("Groq LLM is not configured. Please set the GROQ_API_KEY environment variable.")
            else:
                context = ""
                retrieved_docs = []
                
                # Perform similarity search if the DB is available
                if db is not None:
                    try:
                        retrieved_docs = db.similarity_search(user_query, k=3)
                        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                    except Exception as e:
                        st.error(f"Error during similarity search: {e}")
                
                # System instructions enforcing safety structure
                system_prompt = (
                    "You are HandyRAG, an expert DIY home repair and safety consultant. "
                    "Your task is to answer the user's home repair question using the provided context from guidebooks. "
                    "If you do not know the answer or it is not in the context, use your general knowledge but prioritize safety.\n\n"
                    "Formatting Requirements:\n"
                    "You MUST structure your response into these exact markdown sections:\n\n"
                    "### ⚠️ SAFETY WARNING\n"
                    "[List the mandatory safety precautions, shut-offs, and gear required BEFORE starting. Highlight hazards like electrical shock or dust inhalation.]\n\n"
                    "### 🔧 REQUIRED TOOLS\n"
                    "[List the tools needed for the job as clean bullet points.]\n\n"
                    "### 📋 STEP-BY-STEP PROCEDURE\n"
                    "[Provide clear, numbered steps explaining how to fix the problem.]\n\n"
                    "Make sure to keep instructions concise, clean, and highly practical."
                )
                
                messages = [
                    ("system", system_prompt),
                    ("human", f"Context from guides:\n{context}\n\nQuestion: {user_query}")
                ]
                
                try:
                    response = llm.invoke(messages)
                    response_text = response.content
                    
                    sources = list(set([doc.metadata.get("source", "Unknown") for doc in retrieved_docs]))
                    
                    st.markdown(response_text)
                    if sources:
                        st.caption(f"Sources: {', '.join(sources)}")
                        
                    # Save assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text,
                        "sources": sources
                    })
                except Exception as e:
                    st.error(f"Error generating response: {e}")

# End of application
