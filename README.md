Markdown
# 🛡️ Resilience Engine

> **Built for the Gemma 4 Good Hackathon — Global Resilience + Health & Sciences tracks.**

## 🚨 The Problem
When disaster strikes—earthquake, flood, armed conflict—people die from survivable injuries because they have no access to medical knowledge. Existing tools assume reliable internet, English proficiency, and app installation. In active crisis zones, none of those assumptions hold. 

The **Resilience Engine** was built for that exact gap.

## 💡 What It Does
Send a WhatsApp message—text, voice note, or photo—and get verified first aid and humanitarian guidance back within seconds, in your native language.

| Input | Pipeline Action | Result |
| :--- | :--- | :--- |
| 💬 **Text** | Hybrid RAG search → Gemma 4 reasoning | Response in user's language |
| 🎙️ **Voice Note** | Gemini Flash transcribes → RAG pipeline | Text + Voice note response |
| 📸 **Image / Photo** | Gemma Vision analyzes → RAG retrieves protocol | Grounded medical response |
| 🆘 **SOS Alert** | Detects country code from phone prefix | Instant local emergency dispatch numbers |

## 📚 Knowledge Base
Grounded in 5 verified humanitarian sources—**6,089 chunks** stored in ChromaDB:

| Source | Pages | Category |
| :--- | :--- | :--- |
| **IFRC First Aid & Resuscitation Guidelines 2025** | 594 | Medical |
| **Sphere Humanitarian Handbook 2018** | 458 | Disaster Response |
| **TCCC Guidelines** | 17 | Combat Casualty Care |
| **WHO/WEDC Emergency WASH Notes** | 20 | Water & Sanitation |
| **US Army Survival Manual FM 21-76** | 233 | Survival |

## 🏗️ Architecture

```text
WhatsApp message ↓ Twilio webhook → FastAPI (background task)

       ┌─────────────────────────────────────┐
       │             Input Router            │
       │      Text / Audio / Image / SOS     │
       └─────────────────────────────────────┘
             ↓             ↓            ↓
          [Audio]       [Image]       [Text]
       Gemini Flash   Gemma Vision   Translate
       transcribes     describes     to English
             ↓             ↓            ↓
             └─────────────┴────────────┘
                           ↓
          Hybrid Search (BM25 + Dense Vector)
          Reciprocal Rank Fusion → Top 6 chunks
                           ↓
     Gemma assembles response in user's original language
                           ↓
   + Country emergency numbers (detected from phone prefix)
                           ↓
            Twilio sends text + optional audio
⚙️ Tech Stack
LLM (Text + Vision): Gemma 4 31B via Ollama

Audio Transcription: Gemini 2.5 Flash API

Vector Database: ChromaDB

Embeddings: all-MiniLM-L6-v2

Hybrid Search: BM25 + Dense Vectors + RRF

Backend Pipeline: FastAPI + uvicorn

Communication Interface: Twilio Sandbox for WhatsApp

📂 Project Structure
Plaintext
Resilience Engine/
├── data/                         # Raw PDF sources
├── extracted_texts/              # Processed text from PDFs
├── chroma_db/                    # Persistent vector store
├── audio_cache/                  # Temp voice responses (auto-deleted)
├── extract_pdfs_v1.py            # PDF extraction pipeline
├── chunk_and_embed.py            # Chunking + embedding pipeline
├── main.py                       # Main FastAPI bot
├── emergency_contact.json        # Global emergency numbers DB
└── requirements.txt
🚀 Setup & Run
Prerequisites
Python 3.10+

Ollama installed and running

Poppler (Windows) added to PATH

Twilio account with WhatsApp sandbox

Google AI Studio API key

ngrok for local tunneling

1. Install Dependencies
Bash
pip install -r requirements.txt
2. Environment Variables
Create a .env file in the root directory:

Code snippet
GOOGLE_API_KEY=your_google_ai_studio_key
TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886
3. Initialize Models & Database
Bash
# Pull Gemma via Ollama
ollama pull gemma4:31b

# Extract PDFs
python extract_pdfs_v1.py

# Chunk and embed into ChromaDB
python chunk_and_embed.py
4. Run the Bot
Terminal 1 — Start the server:

Bash
uvicorn main:app --reload --port 8000
Terminal 2 — Expose to internet:

Bash
ngrok http 8000
Paste your ngrok URL + /bot into the Twilio WhatsApp sandbox webhook field.

🧠 Key Design Decisions
Why WhatsApp: Operates on 2G, works in 180+ countries, and requires no installation. The infrastructure already exists in the hands of the people who need it most.

Why verified handbooks: A wrong answer in a crisis kills. TCCC, Sphere, and IFRC are the exact documents trained doctors carry into conflict zones.

Why hybrid search: BM25 catches exact acronyms (CAT tourniquet, UXO) that vector search misses. Dense vectors catch semantic meaning across languages.

Why background tasks: Twilio webhooks timeout after 15 seconds. RAG queries run as FastAPI background tasks, returning an empty 200 immediately and pushing the reply via Twilio REST when ready.

🛡️ Safety Design
The system is instructed to refuse to answer if the procedure is not in the retrieved context.

Outdated advice (e.g., urine on wounds, sucking snake venom) is explicitly flagged and corrected.

Heart attack vs. cardiac arrest are treated as distinct emergencies with different protocols.

All responses are character-limited for WhatsApp readability on small screens.

License: CC-BY 4.0
