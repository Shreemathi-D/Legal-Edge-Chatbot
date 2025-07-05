# Legal Edge Chatbot - Optimized Version with Chat Sessions and Enhanced UI

import os
import streamlit as st
from groq import Groq
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
import hashlib, json, re, time

# Load API keys
load_dotenv()

# --- Configurations ---
DEFAULT_DOCS_PATH = "./Documents"
EMBEDDINGS_PATH = "./embeddings"
SESSIONS_DIR = "./chat_sessions"
SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".md", ".docx", ".doc"]
CHUNK_SIZE, CHUNK_OVERLAP = 500, 10
EMBEDDINGS_MODEL = "sentence-transformers/all-mpnet-base-v2"
MAX_WORKERS = 4

# Ensure required folders exist
for path in [DEFAULT_DOCS_PATH, EMBEDDINGS_PATH, SESSIONS_DIR]:
    Path(path).mkdir(parents=True, exist_ok=True)

# Streamlit UI setup
st.set_page_config(page_title="Legal Edge Chatbot", page_icon="⚖️", layout="wide")
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #1a3a8f 0%, #0d6e6e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .status-success { background-color: #d4edda; color: #155724; padding: 1rem; margin-bottom: 1rem; border-radius: 0.375rem; }
    .status-warning { background-color: #fff3cd; color: #856404; padding: 1rem; margin-bottom: 1rem; border-radius: 0.375rem; }
    .chat-container { max-height: 500px; overflow-y: auto; padding: 1rem; background-color: #fafafa; border-radius: 0.5rem; border: 1px solid #e0e0e0; }
    .user-message, .assistant-message { padding: 1rem; margin: 0.5rem 0; border-radius: 0.5rem; }
    .user-message { background-color: #e9f7ef; border-left: 4px solid #0d6e6e; }
    .assistant-message { background-color: #f8f9fa; border-left: 4px solid #1a3a8f; }
    .citation { font-size: 0.85rem; color: #666; font-style: italic; margin-top: 0.5rem; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'><span class='legal-icon'>⚖️</span> Legal Edge Chatbot</h1>", unsafe_allow_html=True)

# --- Utilities ---
def get_file_hash(path):
    with open(path, 'rb') as f: return hashlib.md5(f.read()).hexdigest()

def get_embeddings_file(path):
    return os.path.join(EMBEDDINGS_PATH, f"{hashlib.md5(path.encode()).hexdigest()}.faiss")

def documents_changed(doc_dir, emb_file):
    if not os.path.exists(emb_file): return True
    hash_path = os.path.join(EMBEDDINGS_PATH, "file_hashes.json")
    current_files, file_hashes = set(), {}
    for ext in SUPPORTED_EXTENSIONS:
        for file in Path(doc_dir).rglob(f"*{ext}"):
            file = str(file)
            current_files.add(file)
            file_hashes[file] = get_file_hash(file)
    try:
        with open(hash_path, 'r') as f:
            old_hashes = json.load(f)
        if set(old_hashes.keys()) != current_files:
            return True
        return any(old_hashes.get(k) != file_hashes[k] for k in file_hashes)
    except:
        return True

def save_file_hashes(hashes):
    with open(os.path.join(EMBEDDINGS_PATH, "file_hashes.json"), 'w') as f:
        json.dump(hashes, f)

# --- Session Management ---
def get_session_file(session_id):
    return os.path.join(SESSIONS_DIR, f"{session_id}.json")

def save_session(session_id, messages):
    session_data = {
        "id": session_id,
        "messages": messages,
        "created_at": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat()
    }
    with open(get_session_file(session_id), 'w') as f:
        json.dump(session_data, f, indent=2)

def load_session(session_id):
    try:
        with open(get_session_file(session_id), 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def get_all_sessions():
    sessions = []
    for file in os.listdir(SESSIONS_DIR):
        if file.endswith('.json'):
            session_id = file[:-5]
            session_data = load_session(session_id)
            if session_data:
                sessions.append(session_data)
    return sorted(sessions, key=lambda x: x['last_updated'], reverse=True)

def delete_session(session_id):
    path = get_session_file(session_id)
    if os.path.exists(path): os.remove(path)

def create_new_session():
    return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# --- Document Processing ---
def load_single_document(file_path):
    try:
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith((".txt", ".md")):
            loader = TextLoader(file_path, encoding='utf-8')
        elif file_path.endswith((".docx", ".doc")):
            loader = Docx2txtLoader(file_path)
        else:
            return []
        return loader.load()
    except Exception as e:
        st.warning(f"Error loading {file_path}: {e}")
        return []

@st.cache_resource(show_spinner=False)
def load_documents():
    files = [p for ext in SUPPORTED_EXTENSIONS for p in Path(DEFAULT_DOCS_PATH).rglob(f"*{ext}")]
    if not files: return None
    emb_file = get_embeddings_file(DEFAULT_DOCS_PATH)
    if os.path.exists(emb_file) and not documents_changed(DEFAULT_DOCS_PATH, emb_file):
        return FAISS.load_local(emb_file, HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL), allow_dangerous_deserialization=True)
    documents, file_hashes = [], {}
    with ThreadPoolExecutor(MAX_WORKERS) as executor:
        futures = {executor.submit(load_single_document, str(fp)): str(fp) for fp in files}
        for future in as_completed(futures):
            result = future.result()
            documents.extend(result)
            file_hashes[futures[future]] = get_file_hash(futures[future])
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(emb_file)
    save_file_hashes(file_hashes)
    return vectorstore

# --- Prompt Template ---
TEMPLATE = """
As Legal Edge Chatbot, provide clear, factual legal answers based on the given context:

Documents:
{context}

Conversation History:
{history}

User's Question:
{question}

INSTRUCTIONS:
- Do NOT mention 'Document 1', 'Document 2', etc.
- Provide authoritative answers without source references in the main response
- Write like a professional legal analyst
- Use clear, direct, and respectful legal language
"""

# --- State Initialization ---
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = create_new_session()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = load_documents()
if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = os.getenv("GROQ_API_KEY")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Chat Settings")
    st.subheader("Session Info")
    st.text(f"ID: {st.session_state.current_session_id}")
    if not st.session_state.groq_api_key:
        api_key = st.text_input("Enter Groq API Key", type="password")
        if api_key:
            st.session_state.groq_api_key = api_key
            st.rerun()
    if st.button("🆕 New Chat"):
        save_session(st.session_state.current_session_id, st.session_state.messages)
        st.session_state.current_session_id = create_new_session()
        st.session_state.messages = []
        st.rerun()
    for session in get_all_sessions():
        preview = session['messages'][0]['content'][:50] + "..." if session['messages'] else "Empty"
        col1, col2 = st.columns([6, 1])
        with col1:
            if st.button(f"📝 {preview}", key=session['id']):
                st.session_state.current_session_id = session['id']
                st.session_state.messages = session['messages']
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"del_{session['id']}"):
                delete_session(session['id'])
                st.rerun()

# --- Chat Interface ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(f"<div class='{msg['role']}-message'>{msg['content']}</div>", unsafe_allow_html=True)

if prompt := st.chat_input("Ask about your legal documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(f"<div class='user-message'>{prompt}</div>", unsafe_allow_html=True)
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                docs = st.session_state.vectorstore.similarity_search(prompt, k=20)
                context = "\n\n".join([d.page_content for d in docs])
                history = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.messages[-6:]])
                final_prompt = PromptTemplate.from_template(TEMPLATE).format(context=context, history=history, question=prompt)
                client = Groq(api_key=st.session_state.groq_api_key)
                completion = client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[{"role": "system", "content": "You are a legal assistant."}, {"role": "user", "content": final_prompt}],
                    temperature=0.2, max_tokens=2048, top_p=0.8
                )
                response = completion.choices[0].message.content.strip()
                st.markdown(f"<div class='assistant-message'>{response}</div>", unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": response})
                save_session(st.session_state.current_session_id, st.session_state.messages)
            except Exception as e:
                st.error(f"❌ {str(e)}")
