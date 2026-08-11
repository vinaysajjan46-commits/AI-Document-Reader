import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader
from docx import Document
import io

# ── Load environment ──────────────────────────────────────────────────────────
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY not found in .env file. Please add it and restart.")
    st.stop()

client = genai.Client(api_key=GOOGLE_API_KEY)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Document Reader",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}

[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255,255,255,0.1);
}

[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.06) !important;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px !important;
    margin-bottom: 12px;
    padding: 4px 8px;
    color: #e2e8f0 !important;
}

@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

[data-testid="stChatInput"] {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(139,92,246,0.5) !important;
    border-radius: 16px !important;
}

[data-testid="stChatInput"] textarea {
    color: #e2e8f0 !important;
    background: transparent !important;
}

[data-testid="stFileUploader"] {
    background: rgba(139,92,246,0.1);
    border: 2px dashed rgba(139,92,246,0.5);
    border-radius: 12px;
    padding: 16px;
}

.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #6366f1) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 8px 20px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(124,58,237,0.4) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(124,58,237,0.6) !important;
}

h1, h2, h3 { color: #e2e8f0 !important; }

[data-testid="stMetric"] {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 16px;
}

[data-testid="stMetricLabel"], [data-testid="stMetricValue"] { color: #e2e8f0 !important; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(139,92,246,0.5); border-radius: 3px; }

p, li, span, div { color: #cbd5e1; }

.hero {
    text-align: center;
    padding: 2rem 0 1rem;
}
.hero h1 {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a78bfa, #818cf8, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
}
.hero p { color: #94a3b8; font-size: 1.1rem; }

.gradient-divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, #7c3aed, #6366f1, transparent);
    border: none;
    margin: 1rem 0;
}

.doc-preview {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 16px;
    max-height: 300px;
    overflow-y: auto;
    font-size: 0.85rem;
    color: #94a3b8;
    white-space: pre-wrap;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# ── Helper Functions ──────────────────────────────────────────────────────────

def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text.strip()


def extract_text_from_docx(file_bytes: bytes) -> str:
    doc = Document(io.BytesIO(file_bytes))
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])


def extract_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")


def get_document_text(uploaded_file):
    file_bytes = uploaded_file.read()
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    elif name.endswith(".docx"):
        return extract_text_from_docx(file_bytes)
    elif name.endswith(".txt"):
        return extract_text_from_txt(file_bytes)
    return None


def get_ai_response(user_question: str, document_text: str, history: list, model_name: str) -> str:
    system_prompt = (
        "You are an expert AI Document Analyst. "
        "You are given the full text of a document and your job is to "
        "answer questions about it accurately and helpfully. "
        "Always base your answers on the provided document. "
        "If the answer is not in the document, say so clearly. "
        "Format responses using markdown when useful."
    )

    # Build conversation contents
    contents = []

    # Add previous turns from history (skip the latest user message, added below)
    for msg in history[:-1]:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))

    # Add current question with document context
    full_question = (
        f"**Document Content:**\n```\n{document_text[:12000]}\n```\n\n"
        f"**Question:** {user_question}"
    )
    contents.append(types.Content(role="user", parts=[types.Part(text=full_question)]))

    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.4,
        ),
    )
    return response.text


# ── Session State Init ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_text" not in st.session_state:
    st.session_state.document_text = None
if "document_name" not in st.session_state:
    st.session_state.document_name = None
if "model_name" not in st.session_state:
    st.session_state.model_name = "gemini-3.5-flash"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 AI Document Reader")
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    st.markdown("### 🤖 AI Model")
    model_choice = st.selectbox(
        "Choose Gemini model",
        options=[
            "gemini-3.5-flash",
            "gemini-3.6-flash",
            "gemini-3.1-flash-lite",
            "gemini-3-flash-preview",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
        ],
        index=0,
        label_visibility="collapsed",
    )
    st.session_state.model_name = model_choice

    st.markdown("### 📁 Upload Document")
    uploaded_file = st.file_uploader(
        "Supported: PDF, DOCX, TXT",
        type=["pdf", "docx", "txt"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        with st.spinner("📖 Reading document…"):
            text = get_document_text(uploaded_file)

        if text:
            st.session_state.document_text = text
            st.session_state.document_name = uploaded_file.name
            st.session_state.messages = []
            st.success(f"✅ Loaded: **{uploaded_file.name}**")
            col1, col2 = st.columns(2)
            col1.metric("Words", f"{len(text.split()):,}")
            col2.metric("Chars", f"{len(text):,}")
        else:
            st.error("❌ Could not read this file type.")

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    if st.session_state.document_text:
        with st.expander("📃 Document Preview", expanded=False):
            preview = st.session_state.document_text[:1500]
            st.markdown(f'<div class="doc-preview">{preview}…</div>', unsafe_allow_html=True)

    if st.session_state.messages:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.75rem; color:#64748b; text-align:center;'>"
        "Powered by Google Gemini 🔮</p>",
        unsafe_allow_html=True,
    )

# ── Main Area ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🔮 AI Document Reader</h1>
    <p>Upload a document and chat with it using Google Gemini</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

if not st.session_state.document_text:
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem;">
        <div style="font-size: 5rem; margin-bottom: 1rem;">📂</div>
        <h3 style="color:#a78bfa; margin-bottom:0.5rem;">No Document Loaded</h3>
        <p style="color:#64748b; max-width:400px; margin:auto;">
            Upload a <b>PDF</b>, <b>DOCX</b>, or <b>TXT</b> file from the sidebar
            to start asking questions about it.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if st.session_state.document_name:
    st.markdown(
        f"💬 **Chatting about:** `{st.session_state.document_name}` &nbsp;|&nbsp; "
        f"Model: `{st.session_state.model_name}`"
    )

# Chat history
for message in st.session_state.messages:
    role = message["role"]
    avatar = "🧑" if role == "user" else "🔮"
    with st.chat_message(role, avatar=avatar):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask anything about the document…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🔮"):
        with st.spinner("🔍 Analysing document…"):
            try:
                answer = get_ai_response(
                    user_question=prompt,
                    document_text=st.session_state.document_text,
                    history=st.session_state.messages,
                    model_name=st.session_state.model_name,
                )
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                err_msg = f"⚠️ **Error:** {str(e)}"
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})