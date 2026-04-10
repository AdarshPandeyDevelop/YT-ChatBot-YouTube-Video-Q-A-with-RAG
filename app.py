import os
import re
import streamlit as st
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# ─────────────────────────────────────────────
# Load environment variables
# ─────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="YT ChatBot",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS — dark cinematic theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
 
/* ── Root tokens ── */
:root {
    --bg:        #0a0a0f;
    --surface:   #12121a;
    --surface2:  #1c1c28;
    --border:    #2a2a3d;
    --accent:    #ff4757;
    --accent2:   #ffa502;
    --accent3:   #2ed573;
    --text:      #e8e8f0;
    --muted:     #6b6b85;
    --user-bg:   #1a1a2e;
    --bot-bg:    #12121a;
    --radius:    14px;
}
 
/* ── Global reset ── */
html, body, .stApp {
    background-color: var(--bg) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}
 
/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
 
/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 2rem;
}
 
/* ── Main container ── */
.main .block-container {
    padding: 1.5rem 2rem 6rem 2rem !important;
    max-width: 860px;
    margin: 0 auto;
}
 
/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a0a1a 50%, #0a1020 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60%;
    right: -10%;
    width: 320px;
    height: 320px;
    background: radial-gradient(circle, rgba(255,71,87,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-banner h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    margin: 0 0 0.3rem 0;
    background: linear-gradient(90deg, #ff4757, #ffa502);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-banner p {
    color: var(--muted);
    margin: 0;
    font-size: 0.92rem;
    font-weight: 300;
}
 
/* ── Status badge ── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(46,213,115,0.1);
    border: 1px solid rgba(46,213,115,0.3);
    color: var(--accent3);
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 500;
    margin-top: 0.8rem;
}
.status-badge.loading {
    background: rgba(255,165,2,0.1);
    border-color: rgba(255,165,2,0.3);
    color: var(--accent2);
}
.status-badge.error {
    background: rgba(255,71,87,0.1);
    border-color: rgba(255,71,87,0.3);
    color: var(--accent);
}
 
/* ── Video info card ── */
.video-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: var(--radius);
    padding: 1rem 1.2rem;
    margin-bottom: 1.5rem;
    font-size: 0.88rem;
}
.video-card .label {
    color: var(--muted);
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
}
.video-card .value {
    color: var(--text);
    font-weight: 500;
}
 
/* ── Chat messages ── */
.chat-wrapper {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    padding-bottom: 1rem;
}
.chat-msg {
    display: flex;
    gap: 12px;
    align-items: flex-start;
    animation: fadeUp 0.3s ease forwards;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
.chat-msg.user { flex-direction: row-reverse; }
 
.avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
}
.avatar.user { background: linear-gradient(135deg, #ff4757, #c0392b); }
.avatar.bot  { background: linear-gradient(135deg, #2ed573, #009432); }
 
.bubble {
    max-width: 78%;
    padding: 0.85rem 1.1rem;
    border-radius: var(--radius);
    line-height: 1.6;
    font-size: 0.92rem;
}
.bubble.user {
    background: var(--user-bg);
    border: 1px solid #2a2a45;
    border-bottom-right-radius: 4px;
}
.bubble.bot {
    background: var(--bot-bg);
    border: 1px solid var(--border);
    border-bottom-left-radius: 4px;
}
.bubble p { margin: 0; }
 
/* ── Divider ── */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 1.5rem 0;
}
 
/* ── Streamlit input overrides ── */
.stTextInput > div > div > input {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: 0.75rem 1rem !important;
    font-size: 0.9rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(255,71,87,0.15) !important;
}
.stTextInput > label {
    color: var(--muted) !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
 
/* ── Streamlit button overrides ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), #c0392b) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.65rem 1.6rem !important;
    letter-spacing: 0.04em !important;
    transition: all 0.2s ease !important;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(255,71,87,0.3) !important;
}
.stButton > button:active {
    transform: translateY(0px) !important;
}
 
/* ── Chat input ── */
.stChatInput > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
}
.stChatInput textarea {
    background: transparent !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stChatInput button {
    background: var(--accent) !important;
}
 
/* ── Streamlit chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
 
/* ── Sidebar content ── */
.sidebar-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 0.2rem;
}
.sidebar-sub {
    color: var(--muted);
    font-size: 0.8rem;
    margin-bottom: 1.5rem;
}
.sidebar-section {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 0.9rem 1rem;
    margin-bottom: 0.8rem;
}
.sidebar-section .label {
    color: var(--muted);
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
}
.sidebar-section .value {
    color: var(--text);
    font-size: 0.88rem;
    font-weight: 500;
}
 
/* ── Spinner ── */
.stSpinner > div {
    border-top-color: var(--accent) !important;
}
 
/* ── Info / warning boxes ── */
.stAlert {
    border-radius: var(--radius) !important;
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
}
 
/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }
 
/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: var(--muted);
}
.empty-state .icon { font-size: 3rem; margin-bottom: 1rem; }
.empty-state h3 {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    color: var(--text);
    margin-bottom: 0.5rem;
}
.empty-state p { font-size: 0.88rem; line-height: 1.6; }
 
/* ── Stat chip ── */
.stat-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.75rem;
    color: var(--muted);
    margin-right: 6px;
    margin-top: 4px;
}
.stat-chip span { color: var(--text); font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Utility: extract video ID from URL
# ─────────────────────────────────────────────
def extract_video_id(url: str) -> str | None:
    """
    Supports all common YouTube URL formats:
      - https://www.youtube.com/watch?v=VIDEO_ID
      - https://youtu.be/VIDEO_ID
      - https://www.youtube.com/embed/VIDEO_ID
      - https://www.youtube.com/shorts/VIDEO_ID
      - Plain video IDs (11 chars)
    """
    url = url.strip()
    patterns = [
        r"(?:youtube\.com/watch\?(?:.*&)?v=)([A-Za-z0-9_-]{11})",
        r"(?:youtu\.be/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/embed/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/shorts/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/v/)([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    # Plain ID fallback
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return url
    return None


# ─────────────────────────────────────────────
# Core: fetch transcript (any language)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def fetch_transcript(video_id: str) -> tuple[str, int, str]:
    """
    Returns (full_transcript_text, snippet_count, detected_language_code).
    Works with any language by:
    1. Listing all available transcripts
    2. Preferring manually-created over auto-generated
    3. Falling back gracefully to whatever is available
    """
    ytt_api = YouTubeTranscriptApi()

    # Step 1: List all available transcripts for the video
    transcript_list = ytt_api.list(video_id)

    selected = None
    detected_lang = "unknown"

    # Step 2: Prefer manually-created transcripts (more accurate)
    try:
        for t in transcript_list:
            if not t.is_generated:
                selected = t
                detected_lang = t.language_code
                break
    except Exception:
        pass

    # Step 3: Fall back to auto-generated if no manual one found
    if selected is None:
        try:
            for t in transcript_list:
                selected = t
                detected_lang = t.language_code
                break
        except Exception:
            pass

    if selected is None:
        raise ValueError("No transcript available for this video.")

    # Step 4: Fetch the transcript
    data = selected.fetch()
    text = " ".join(chunk.text for chunk in data)
    return text, len(data), detected_lang


# ─────────────────────────────────────────────
# Core: build RAG chain (cached by video_id)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_rag_chain(video_id: str):
    """Build and cache the full RAG pipeline for a given video."""
    transcript, _, detected_lang = fetch_transcript(
        video_id)  # ← unpack 3 values now

    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    # Embed + index
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Retriever
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 4})

    # LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        api_key=GROQ_API_KEY,
    )

    # Prompt  — includes chat history
    prompt = PromptTemplate(
        template="""You are a knowledgeable and friendly AI assistant that answers questions about a YouTube video transcript.

The transcript language is: {transcript_lang}

IMPORTANT LANGUAGE RULE:
- If the user asks in English, respond in English — even if the transcript is in another language.
- If the user asks in the transcript's language, respond in that same language.
- Always match the language the user is writing in.

TRANSCRIPT CONTEXT:
{context}
 
CONVERSATION HISTORY:
{chat_history}
 
CURRENT QUESTION: {question}
 
Instructions:
- Answer ONLY using information from the transcript context above.
- If the answer isn't in the transcript, say "I couldn't find that in the video."
- Keep answers concise but complete.
- When referencing specific parts, quote briefly.
- Be conversational and friendly.
 
Answer:""",
        input_variables=["context", "question",
                         "chat_history", "transcript_lang"],
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def format_history(history: list[dict]) -> str:
        if not history:
            return "No previous conversation."
        lines = []
        for msg in history[-6:]:  # last 3 exchanges = 6 messages
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    # Chain
    chain = (
        RunnableParallel(
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
                "chat_history": RunnableLambda(lambda _: format_history(
                    st.session_state.get("messages", [])
                )),
                # ← add this
                "transcript_lang": RunnableLambda(lambda _: detected_lang),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, len(chunks)


# ─────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────
def init_state():
    defaults = {
        "messages": [],
        "video_id": None,
        "video_url": "",
        "chain": None,
        "chunk_count": 0,
        "transcript_loaded": False,
        "loading_error": None,
        "detected_lang": "unknown",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">🎬 YT ChatBot</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Powered by Groq · LangChain · FAISS</div>',
                unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("**Load a YouTube Video**")
    url_input = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
        key="url_input",
        label_visibility="collapsed",
    )

    load_btn = st.button("⚡ Load & Index Video", use_container_width=True)

    if load_btn and url_input:
        vid_id = extract_video_id(url_input)
        if not vid_id:
            st.error("❌ Invalid YouTube URL. Please check and try again.")
        elif vid_id == st.session_state.video_id:
            st.info("✅ This video is already loaded!")
        else:
            with st.spinner("Fetching transcript & building index…"):
                try:
                    chain, n_chunks = build_rag_chain(vid_id)
                    st.session_state.video_id = vid_id
                    st.session_state.video_url = url_input
                    st.session_state.chain = chain
                    _, _, lang = fetch_transcript(vid_id)
                    st.session_state.detected_lang = lang
                    st.session_state.chunk_count = n_chunks
                    st.session_state.transcript_loaded = True
                    st.session_state.messages = []
                    st.session_state.loading_error = None
                    st.success("✅ Video indexed successfully!")
                except TranscriptsDisabled:
                    st.session_state.loading_error = "Transcripts are disabled for this video."
                    st.error("❌ Transcripts are disabled for this video.")
                except Exception as e:
                    st.session_state.loading_error = str(e)
                    st.error(f"❌ Error: {e}")

    # Video info panel
    if st.session_state.transcript_loaded and st.session_state.video_id:
        st.markdown("---")
        st.markdown("**📊 Current Video**")

        vid_id = st.session_state.video_id
        thumb_url = f"https://img.youtube.com/vi/{vid_id}/mqdefault.jpg"
        st.image(thumb_url, use_container_width=True)

        st.markdown(f"""
<div class="sidebar-section">
    <div class="label">Video ID</div>
    <div class="value">{vid_id}</div>
</div>
<div class="sidebar-section">
    <div class="label">Transcript Language</div>
    <div class="value">{st.session_state.detected_lang}</div>
</div>
<div class="sidebar-section">
    <div class="label">Index Size</div>
    <div class="value">{st.session_state.chunk_count} chunks · all-MiniLM-L6-v2</div>
</div>
<div class="sidebar-section">
    <div class="label">LLM</div>
    <div class="value">Llama 3.3 · 70B · Groq</div>
</div>
<div class="sidebar-section">
    <div class="label">Messages</div>
    <div class="value">{len(st.session_state.messages)} in session</div>
</div>
""", unsafe_allow_html=True)

        yt_url = f"https://www.youtube.com/watch?v={vid_id}"
        st.markdown(
            f'<a href="{yt_url}" target="_blank" style="color:#ff4757;font-size:0.82rem;">▶ Open on YouTube ↗</a>', unsafe_allow_html=True)

        st.markdown("---")
        if st.button("🗑 Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style="color:#6b6b85;font-size:0.75rem;line-height:1.6;">
    <b style="color:#e8e8f0;">How it works</b><br>
    1. Paste any YouTube URL<br>
    2. Click Load &amp; Index<br>
    3. Ask anything about the video<br><br>
    Answers are grounded in the video transcript only.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Main area — Hero banner
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <h1>YouTube Video ChatBot</h1>
    <p>Load any YouTube video and have an intelligent conversation about its content.<br>
    Powered by RAG · Groq Llama 3.3 70B · FAISS vector search</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Chat area
# ─────────────────────────────────────────────
if not st.session_state.transcript_loaded:
    st.markdown("""
    <div class="empty-state">
        <div class="icon">🎬</div>
        <h3>No video loaded yet</h3>
        <p>Paste a YouTube URL below and click Load to get started.<br><br>
        Supports youtube.com/watch, youtu.be,<br>
        Shorts, and Embeds.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Mobile-friendly inline loader ──
    col1, col2 = st.columns([4, 1])
    with col1:
        mobile_url = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            key="mobile_url_input",
            label_visibility="collapsed",
        )
    with col2:
        mobile_btn = st.button("⚡ Load", use_container_width=True)

    if mobile_btn and mobile_url:
        vid_id = extract_video_id(mobile_url)
        if not vid_id:
            st.error("❌ Invalid YouTube URL.")
        elif vid_id == st.session_state.video_id:
            st.info("✅ Already loaded!")
        else:
            with st.spinner("Fetching transcript & building index…"):
                try:
                    chain, n_chunks = build_rag_chain(vid_id)
                    st.session_state.video_id = vid_id
                    st.session_state.video_url = mobile_url
                    st.session_state.chain = chain
                    _, _, lang = fetch_transcript(vid_id)
                    st.session_state.detected_lang = lang
                    st.session_state.chunk_count = n_chunks
                    st.session_state.transcript_loaded = True
                    st.session_state.messages = []
                    st.session_state.loading_error = None
                    st.rerun()
                except TranscriptsDisabled:
                    st.error("❌ Transcripts are disabled for this video.")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

else:
    # Show existing messages
    if not st.session_state.messages:
        # Welcome message from bot
        vid_id = st.session_state.video_id
        welcome = (
            f"👋 Hey! I've indexed the video (`{vid_id}`). "
            f"I split the transcript into **{st.session_state.chunk_count} chunks** and built a semantic search index. "
            "Ask me anything about the video content — I'll answer based only on what's in the transcript!"
        )
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(welcome)

    for msg in st.session_state.messages:
        avatar = "🧑" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # Chat input
    if question := st.chat_input("Ask something about the video…"):
        # Display user message
        with st.chat_message("user", avatar="🧑"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        # Generate response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking…"):
                try:
                    answer = st.session_state.chain.invoke(question)
                    st.markdown(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer})
                except Exception as e:
                    err_msg = f"⚠️ An error occurred: `{e}`\n\nPlease check your API key in the `.env` file."
                    st.error(err_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": err_msg})
