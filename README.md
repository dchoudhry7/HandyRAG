# HandyRAG - DIY Home Repair & Safety Advisor

HandyRAG is a lightweight, safety-first RAG (Retrieval-Augmented Generation) application designed to assist with DIY home repair tasks. It retrieves content from active builder manuals, contractor guides, or custom uploaded documents, using it to compile safety precautions, required tools, and step-by-step procedures.

The application is built entirely in Python using **Streamlit** for the frontend and **LangChain** with **Chroma DB** and **Groq LLM (Llama 3.1)** on the backend.

---

## Workspace Structure

- `main.py`: The entrypoint Streamlit web application. Contains the UI logic and RAG query processing.
- `ingest.py`: Core ingestion pipeline script to pre-populate vectors from guides in `data/`.
- `requirements.txt`: Project dependencies list.
- `data/`: Local storage for standard reference guide `.txt` files.
- `uploads/`: Dynamic storage directory for uploaded PDF manuals.
- `chroma_db/`: Persistent local directory for Chroma vector store databases.

---

## Local Setup

### 1. Requirements Installation
Ensure Python (3.9+) is installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Environment Configuration
Create a `.env` file in the root directory (based on `.env.example` template) and insert your Groq API Key:
```env
GROQ_API_KEY="your-actual-groq-api-key"
PORT=8505
```

### 3. Run Ingestion (Optional)
To index any standard text files placed in the `data/` directory into Chroma DB:
```bash
python ingest.py
```

### 4. Start the Application
You can start the app directly using Python or Streamlit:
```bash
python main.py
```
Or:
```bash
streamlit run main.py --server.port 8505
```
Open your browser and navigate to `http://localhost:8505` to access the HandyRAG console.

---

## Deploying to Streamlit Community Cloud

Streamlit Community Cloud is a free hosting platform specifically tailored for Streamlit applications. Here is the step-by-step process to deploy HandyRAG:

### Step 1: Push Project to GitHub
1. Create a new repository on GitHub (e.g., `handyrag`).
2. Initialize Git in your local folder:
   ```bash
   git init
   git add .
   git commit -m "Initial commit of HandyRAG Streamlit app"
   ```
3. Link to your GitHub remote and push:
   ```bash
   git remote add origin https://github.com/your-username/handyrag.git
   git branch -M main
   git push -u origin main
   ```
   *Note: Ensure your `.env` and `chroma_db/` folders are ignored in your `.gitignore` to prevent committing secrets or cache directories.*

### Step 2: Sign In to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io/).
2. Sign in using your GitHub account.

### Step 3: Deploy App
1. Click **New app** in your Streamlit Cloud workspace.
2. Select your repository (`handyrag`), branch (`main`), and main file path (`main.py`).
3. Click on the **Advanced settings** button BEFORE deploying.

### Step 4: Configure Environment Secrets
Streamlit Community Cloud uses a secrets management panel instead of `.env` files. 
In the **Secrets** text area, paste your Groq API Key:
```toml
GROQ_API_KEY = "your-actual-groq-api-key"
```
Click **Save**.

### Step 5: Complete Deployment
Click **Deploy!**. Streamlit Cloud will automatically provision a container, install dependencies from `requirements.txt`, configure the environment secrets, and launch your live application.
