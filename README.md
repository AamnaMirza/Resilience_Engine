The Problem
When disaster strikes — earthquake, flood, armed conflict — people die from survivable injuries because they have no access to medical knowledge. Existing tools assume internet, English, and app installation. In active crisis zones, none of those assumptions hold.
Resilience Engine was built for that gap.

What It Does
Send a WhatsApp message — text, voice note, or photo — and get verified first aid and humanitarian guidance back within seconds, in your language.
InputWhat happensTextHybrid RAG search → Gemma generates response in user's languageVoice noteGemini Flash transcribes → same RAG pipeline → voice note responsePhoto of injury/situationGemma vision analyzes → RAG retrieves relevant procedures → grounded responseSOSInstant emergency protocol + country-specific emergency numbers

Knowledge Base
Grounded in 5 verified humanitarian sources — 6,089 chunks stored in ChromaDB:
SourcePagesTypeIFRC First Aid & Resuscitation Guidelines 2025594MedicalSphere Humanitarian Handbook 2018458Disaster ResponseTCCC Guidelines17Combat Casualty CareWHO/WEDC Emergency WASH Technical Notes (Ch. 5,7,8,9,13)20Water & SanitationUS Army Survival Manual FM 21-76233Survival

Architecture
WhatsApp message
      ↓
Twilio webhook → FastAPI (background task)
      ↓
┌─────────────────────────────────────┐
│         Input Router                │
│  Text / Audio / Image / SOS         │
└─────────────────────────────────────┘
      ↓              ↓            ↓
   [Audio]        [Image]      [Text]
   Gemini Flash   Gemma Vision  Translate
   transcribes    describes     to English
      ↓              ↓            ↓
      └──────────────┴────────────┘
                     ↓
         Hybrid Search (BM25 + Dense Vector)
         Reciprocal Rank Fusion → Top 6 chunks
                     ↓
         Gemma assembles response
         in user's original language
                     ↓
         + Country emergency numbers
         (detected from phone prefix)
                     ↓
         Twilio sends text + optional audio

Tech Stack
ComponentTechnologyLLM (text + vision)Gemma 4 31B via OllamaAudio transcriptionGemini Flash APIText-to-speechgTTSVector databaseChromaDBEmbeddingsall-MiniLM-L6-v2Hybrid searchBM25 + Dense Vectors + RRFBackendFastAPI + uvicornWhatsApp interfaceTwilioPDF extractionpymupdf4llm + pdftotextChunkingLangChain text splitters

Project Structure
Resilience Engine/
├── data/                          # Raw PDF sources
│   ├── IRFC/
│   ├── Sphere/
│   ├── TCCC/
│   ├── Survival Manual/
│   └── WASH/
├── extracted_texts/               # Processed text from PDFs
├── chroma_db/                     # Persistent vector store
├── audio_cache/                   # Temp voice responses (auto-deleted)
├── extract_pdfs_v1.py             # PDF extraction pipeline
├── chunk_and_embed.py             # Chunking + embedding pipeline
├── query_pipeline_working.py      # Main FastAPI bot
├── emergency_contact.json         # 195-country emergency numbers
└── requirements.txt

Setup & Run
Prerequisites

Python 3.10+
Ollama installed and running
Poppler (Windows) added to PATH
Twilio account with WhatsApp sandbox
Google AI Studio API key
ngrok for local tunneling

1. Install dependencies
bashpip install -r requirements.txt
2. Set up environment variables
Create a .env file:
envGOOGLE_API_KEY=your_google_ai_studio_key
TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886
3. Pull Gemma via Ollama
bashollama pull gemma4:31b
4. Build the knowledge base
bash# Extract PDFs
python extract_pdfs_v1.py

# Chunk and embed into ChromaDB
python chunk_and_embed.py
5. Run the bot
bash# Terminal 1 — start the server
uvicorn query_pipeline_working:app --reload --port 8000

# Terminal 2 — expose to internet
ngrok http 8000
6. Configure Twilio
Paste your ngrok URL + /bot into the Twilio WhatsApp sandbox webhook field.
7. Test without WhatsApp
bash# Test retrieval quality
python chunk_and_embed.py test

# Test full pipeline
curl "http://localhost:8000/test?q=how+to+stop+bleeding"

# Test emergency number lookup
curl "http://localhost:8000/test-emergency?number=whatsapp:+919876543210"

Key Design Decisions
Why WhatsApp: Operates on 2G, works in 180+ countries, no installation required. The infrastructure already exists in the hands of the people who need this most.
Why verified handbooks: A wrong answer in a crisis kills. TCCC, Sphere, and IFRC are the documents trained doctors and aid workers carry into conflict zones. Every response is grounded in those same sources.
Why hybrid search: BM25 catches exact acronyms (CAT tourniquet, TCCC, UXO) that vector search misses. Dense vectors catch semantic meaning across languages. Together they handle both "CAT tourniquet" and "كيف أوقف النزيف" (how do I stop bleeding).
Why background tasks: Twilio webhooks timeout after 15 seconds. Gemma inference can take longer. All RAG queries run as FastAPI background tasks, returning an empty 200 immediately and pushing the reply via Twilio REST when ready.

Safety Design

Gemma is instructed to refuse to answer if the procedure is not in the retrieved context
Outdated advice (urine on wounds, butter on burns, sucking snake venom) is explicitly flagged and corrected
Heart attack vs cardiac arrest are treated as distinct emergencies with different protocols
All responses are character-limited for WhatsApp readability


Submission
Built for the Gemma 4 Good Hackathon — Global Resilience + Health & Sciences tracks.
License: CC-BY 4.0