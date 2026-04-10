🎬 YT ChatBot — YouTube Video Q&A with RAG

Paste any YouTube URL. Ask anything about it. Get answers grounded purely in the video's transcript — powered by a full RAG pipeline with semantic search, chat history, and multilingual support.

📌 What It Does
YouTube videos can be long. Watching an hour-long lecture just to find one specific concept is frustrating. This app lets you chat with any YouTube video — load it, and immediately start asking questions about its content.
Under the hood, it:

Fetches the video's transcript (auto-generated or manual, any language)
Splits it into overlapping chunks and embeds them into a FAISS vector index
On each question, retrieves the most semantically relevant chunks
Passes those chunks + conversation history into a Groq-hosted LLaMA 3.3 70B model
Returns an answer grounded only in the transcript

No hallucinations about the video. No answers from general knowledge. If it's not in the transcript, the bot says so.

✨ Features
FeatureDetails

🔗 Universal URL Supportyoutube.com/watch, youtu.be, /shorts/, /embed/, plain video IDs

🧠 RAG PipelineRetrieval-Augmented Generation using FAISS + all-MiniLM-L6-v2 embeddings

💬 Conversational MemoryLast 3 exchanges (6 messages) injected into every prompt for coherent follow-ups

🌍 Multilingual TranscriptsDetects transcript language; prefers manually-created over auto-generated

🗣️ Language-Aware ResponsesResponds in the language the user writes in, regardless of transcript language

⚡ Fast InferenceGroq's free-tier API runs LLaMA 3.3 70B with extremely low latency

🖼️ Thumbnail PreviewSidebar shows the video thumbnail after indexing

📊 Index MetadataDisplays chunk count, embedding model, LLM name, and message count

📱 Mobile-FriendlyInline URL loader on the main page — no sidebar required on mobile

♻️ Smart Caching@st.cache_resource prevents re-indexing the same video on page reruns

🎨 Custom Dark UICinematic dark theme with custom fonts (Syne + DM Sans), no default Streamlit look

🏗️ Architecture
User Input (YouTube URL)
        │
        ▼
┌───────────────────┐
│  extract_video_id │  ← Regex-based URL parser (handles all YT formats)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  fetch_transcript │  ← YouTubeTranscriptApi
│                   │    • Lists all available transcripts
│                   │    • Prefers manual > auto-generated
│                   │    • Detects language code
└────────┬──────────┘
         │
         ▼
┌──────────────────────────┐
│  RecursiveCharacterText  │  ← chunk_size=1000, overlap=200
│       Splitter           │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  HuggingFaceEmbeddings   │  ← sentence-transformers/all-MiniLM-L6-v2
│  + FAISS Vector Store    │     (runs locally, no API key needed)
└────────┬─────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│              LangChain RAG Chain            │
│                                             │
│  RunnableParallel:                          │
│    context      → retriever → format_docs  │
│    question     → RunnablePassthrough       │
│    chat_history → last 6 messages           │
│    transcript_lang → detected language      │
│         │                                   │
│         ▼                                   │
│    PromptTemplate → ChatGroq (LLaMA 3.3)   │
│         │                                   │
│         ▼                                   │
│    StrOutputParser → answer string          │
└─────────────────────────────────────────────┘
         │
         ▼
  Streamlit Chat UI  (session_state message history)

🛠️ Tech Stack
LayerTechnologyUIStreamlitTranscript Fetchingyoutube-transcript-apiText Splittinglangchain-text-splitters — RecursiveCharacterTextSplitterEmbeddingssentence-transformers/all-MiniLM-L6-v2 via langchain-huggingfaceVector StoreFAISS (local, in-memory)LLMLLaMA 3.3 70B Versatile via Groq APIOrchestrationLangChain LCEL (RunnableParallel, RunnablePassthrough, RunnableLambda)Environmentpython-dotenv

🚀 Getting Started
Prerequisites

Python 3.10 or higher
A free Groq API key (takes ~30 seconds to get)

1. Clone the repository
bashgit clone https://github.com/your-username/yt-chatbot.git
cd yt-chatbot
2. Create and activate a virtual environment
bashpython -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
3. Install dependencies
bashpip install -r requirements.txt

⚠️ First run will download the all-MiniLM-L6-v2 model (~90MB). This is a one-time download and gets cached locally.

4. Set up your API key
Create a .env file in the project root:
bashGROQ_API_KEY=your_groq_api_key_here


5. Run the app
bashstreamlit run app.py
The app will open at http://localhost:8501.

📖 How to Use

Paste a YouTube URL in the sidebar (or the inline loader on mobile)
Click ⚡ Load & Index Video — the transcript is fetched and indexed in seconds
Once indexed, the sidebar shows the thumbnail, chunk count, and detected language
Start asking questions in the chat input at the bottom
The bot answers using only what's in the video transcript

Works with: lectures, tutorials, podcasts, interviews, documentaries, and any video with subtitles/transcripts enabled.

🧪 Example Questions to Try
Load any tutorial video and ask:

"What are the main topics covered in this video?"
"Explain the concept discussed around the middle of the video"
"What tools or libraries does the speaker recommend?"
"Summarize the key takeaways"
"What did the speaker say about [specific topic]?"

⚙️ Configuration
You can tweak these values inside app.py to tune performance:
ParameterLocationDefaultEffectchunk_sizeRecursiveCharacterTextSplitter1000Larger = more context per chunk, fewer chunkschunk_overlapRecursiveCharacterTextSplitter200Higher = better continuity across chunksk (retrieval)as_retriever(search_kwargs={"k": 4})4More chunks retrieved = broader contexttemperatureChatGroq(temperature=0.2)0.2Lower = more factual, higher = more creativeHistory windowhistory[-6:] in format_history6 msgsHow many past messages are sent to the LLM

🔒 Environment Variables
VariableRequiredDescriptionGROQ_API_KEY✅ YesYour API key from console.groq.com
No other API keys are needed. Embeddings run locally via HuggingFace.
And I'm really sorry that I haven't uploaded the API, please get your own API from GROQ. Thank you.

📦 requirements.txt
streamlit>=1.35.0
python-dotenv>=1.0.0
youtube-transcript-api>=0.6.2
langchain>=0.2.0
langchain-text-splitters>=0.2.0
langchain-huggingface>=0.0.3
langchain-community>=0.2.0
langchain-groq>=0.1.5
langchain-core>=0.2.0
faiss-cpu>=1.8.0
sentence-transformers>=2.7.0

⚠️ Known Limitations

Transcripts required: Videos with disabled subtitles cannot be loaded. The app shows a clear error in this case.
Transcript accuracy: Auto-generated transcripts (especially for non-English videos) may have errors that affect answer quality.
No timestamps: The current version doesn't link answers back to specific timestamps in the video.
In-memory index: The FAISS index is not persisted to disk. Closing the app clears the index (but Streamlit's @st.cache_resource keeps it alive across reruns in the same session).
Context window limit: Very long videos produce many chunks; the retriever still only passes k=4 to the LLM, so extremely detailed questions about specific moments may miss context.


🗺️ Possible Future Improvements

 Timestamp-linked answers ("This was discussed at 12:34")
 Persistent FAISS index saved to disk per video
 Multi-video support (load and switch between several videos)
 YouTube Data API integration for video title and channel name
 Export chat history as PDF or markdown
 Streaming LLM responses (token-by-token output)


🙋 FAQ

Q: Do I need a paid API key?
A: No. Groq offers a generous free tier that is sufficient for personal use and demos.

Q: Does it work with non-English videos?
A: Yes. The app detects the transcript language and instructs the LLM to respond in whatever language the user writes in.

Q: Why FAISS instead of a cloud vector DB?
A: FAISS runs entirely in-memory with zero setup, no additional API keys, and no cost. It's the right choice for a single-video, single-session use case.

Q: Why Groq instead of OpenAI?
A: Groq's free tier provides extremely fast LLaMA 3.3 70B inference — comparable quality to GPT-4 class models at no cost, which makes this project fully free to run.

👤 Author
Adarsh Pandey
BCA Student | ML/AI Developer

📄 License
This project is licensed under the MIT License — see the LICENSE file for details.

![WhatsApp Image 2026-04-10 at 6 52 29 PM](https://github.com/user-attachments/assets/93ffe3c1-b075-4242-af27-96473ef3f2ed)
![WhatsApp Image 2026-04-10 at 6 56 02 PM](https://github.com/user-attachments/assets/5ff0d555-5b29-4b75-9567-39aee46d991b)
