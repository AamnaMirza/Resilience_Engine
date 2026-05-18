"""
Crisis Response RAG — Query Pipeline (Fixed)
=============================================
FastAPI WhatsApp bot that:
1. Receives WhatsApp messages via Twilio webhook
2. Runs hybrid search (BM25 + ChromaDB) on crisis manuals
3. Assembles a prompt with retrieved context
4. Calls Gemma 3 27B via Ollama Cloud for text + vision
5. Uses Gemini Flash for audio transcription (STT)
6. Responds with text + optional voice note via Twilio

Run:
    uvicorn query_pipeline_fixed:app --reload --port 8000

Then in another terminal:
    ngrok http 8000

Paste the ngrok URL + /bot into Twilio sandbox webhook.
"""

# ── Imports ───────────────────────────────────
import os
import base64
import uuid
import threading
import requests
from pathlib import Path

from fastapi import FastAPI, Request, Form, BackgroundTasks
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
from dotenv import load_dotenv

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from gtts import gTTS
from langdetect import detect

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
load_dotenv()

GOOGLE_API_KEY         = os.getenv("GOOGLE_API_KEY")
TWILIO_ACCOUNT_SID     = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN      = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")

CHROMA_DIR      = r"C:\Users\Aamna\Documents\Resilience Engine\chroma_db"
COLLECTION_NAME = "crisis_knowledge"
EMBED_MODEL     = "all-MiniLM-L6-v2"

# FIX 8: Single source of truth for model name — used everywhere
OLLAMA_MODEL    = "gemma3:27b-cloud"
OLLAMA_URL      = "http://localhost:11434/api/generate"

# FIX 6: Reduced from 9 to 6 — faster, same quality, lower token cost
TOP_K = 6

# Audio cleanup delay in seconds — delete files after sending
AUDIO_CLEANUP_DELAY = 120

# ─────────────────────────────────────────────

app = FastAPI()

# Absolute path for audio cache
BASE_DIR  = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "audio_cache"
AUDIO_DIR.mkdir(exist_ok=True)

app.mount("/audio", StaticFiles(directory=str(AUDIO_DIR)), name="audio")


# ── Load resources at startup ─────────────────

print("Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)

print("Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection     = chroma_client.get_collection(COLLECTION_NAME)

print("Loading BM25 index...")
all_results = collection.get(include=["documents", "metadatas"])
bm25_docs   = [
    Document(page_content=doc, metadata=meta)
    for doc, meta in zip(all_results["documents"], all_results["metadatas"])
]
bm25_retriever   = BM25Retriever.from_documents(bm25_docs)
bm25_retriever.k = TOP_K

print(f"✅ BM25 index built with {len(bm25_docs)} documents")
print(f"✅ Model: {OLLAMA_MODEL}")
print("✅ All systems ready — bot is live!\n")


# ── Utilities ─────────────────────────────────

def schedule_file_cleanup(filepath: Path, delay: int = AUDIO_CLEANUP_DELAY):
    """FIX 3: Delete audio files after sending to prevent disk filling up."""
    def _delete():
        filepath.unlink(missing_ok=True)
        print(f"🗑️ Cleaned up audio: {filepath.name}")
    threading.Timer(delay, _delete).start()


# ── Translation ───────────────────────────────

def translate_to_english(text: str) -> str:
    """
    Translates non-English queries to English for better RAG search.
    FIX 5: Only skip translation for pure ASCII — keeps the logic honest.
    Note: French/Spanish/Portuguese are ASCII but embed well in English DB anyway.
    """
    if text.isascii():
        return text

    print("🌍 Non-English detected. Translating with Gemma...")

    payload = {
        "model":  OLLAMA_MODEL,
        "system": "You are a professional translator. Translate the user's text into English. Output ONLY the raw English translation. Do not include quotation marks, conversational filler, or explanations.",
        "prompt": text,
        "stream": False,
        "options": {"temperature": 0.1}
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=20)
        resp.raise_for_status()
        english_text = resp.json()["response"].strip().strip('"').strip("'")
        print(f"   Translated: {english_text}")
        return english_text
    except Exception as e:
        print(f"⚠️ Translation error: {e} — using original text")
        return text


# ── Hybrid search ─────────────────────────────

def hybrid_search(query: str) -> list[Document]:
    """
    BM25 (keyword) + dense vector (semantic) search with reciprocal rank fusion.
    BM25 catches acronyms: CAT, TCCC, WASH, UXO.
    Dense vectors catch cross-language semantic meaning.
    """
    # Dense vector search
    query_embedding = embedder.encode([query])[0].tolist()
    vector_results  = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
    )

    vector_docs = [
        Document(
            page_content=doc,
            metadata={**meta, "distance": dist}
        )
        for doc, meta, dist in zip(
            vector_results["documents"][0],
            vector_results["metadatas"][0],
            vector_results["distances"][0]
        )
    ]

    # BM25 keyword search
    bm25_results = bm25_retriever.invoke(query)

    # Reciprocal rank fusion
    scores = {}
    rrf_k  = 60

    for rank, doc in enumerate(vector_docs):
        key = doc.page_content[:100]
        scores[key] = scores.get(key, 0) + 1 / (rank + rrf_k)

    for rank, doc in enumerate(bm25_results):
        key = doc.page_content[:100]
        scores[key] = scores.get(key, 0) + 1 / (rank + rrf_k)

    all_docs = {doc.page_content[:100]: doc for doc in vector_docs + bm25_results}
    ranked   = sorted(all_docs.items(), key=lambda x: scores.get(x[0], 0), reverse=True)

    return [doc for _, doc in ranked[:TOP_K]]


# ── Prompt assembly ───────────────────────────

SYSTEM_PROMPT = """You are a crisis response assistant helping people during wars, natural disasters, and emergencies.
You have access to verified emergency handbooks including TCCC (combat casualty care), IFRC First Aid Guidelines, Sphere Humanitarian Standards, WHO WASH guidelines, and the US Army Survival Manual.

CRITICAL RULES:
- Give clear, numbered, actionable steps
- Be concise — the person may be in danger right now
- If the question is medical, prioritize life-saving steps first
- Never say "consult a doctor" as the ONLY answer — give immediate first aid steps
- Do not confuse one emergency with another unless explicitly stated in the context
- If you don't know something, say so clearly rather than guessing or hallucinating
- Respond in the SAME LANGUAGE the user wrote in
- MEDICAL TRIAGE: Never confuse a Heart Attack (conscious, chest pain → sit down, chew aspirin) with Cardiac Arrest (unconscious, not breathing → CPR). If the user says "heart attack", assume conscious first aid. Do NOT immediately recommend CPR.
- Keep responses under 1200 characters for WhatsApp readability
- If the answer is not in the context, reply: I cannot verify this procedure. Please seek human medical or humanitarian assistance.
- If you encounter outdated or unsafe advice (using urine to clean wounds, sucking snake venom, applying butter to burns), warn the user NOT to do it and follow modern IFRC standards instead
- Do not mention the outdated advice itself — just give the correct alternative
- EVERY word you generate MUST be in the exact language the user used in their question
- STRICT RULE: NEVER output bilingual text. DO NOT add English translations in parentheses. Output strictly and ONLY in the user's language."""


def assemble_prompt(user_query: str, retrieved_docs: list[Document]) -> str:
    """Build the full prompt with retrieved context."""
    context_parts = []

    for i, doc in enumerate(retrieved_docs):
        source = doc.metadata.get("source", "Unknown source")
        text   = doc.page_content.strip()
        context_parts.append(f"[Source {i+1}: {source}]\n{text}")

    context = "\n\n---\n\n".join(context_parts)

    return f"""REFERENCE MATERIAL:
{context}

QUESTION: {user_query}

CRITICAL INSTRUCTION: Reply in the EXACT SAME LANGUAGE the user used in the QUESTION.
- English question → English reply only
- Arabic question → Arabic reply only
- Hindi question → Hindi reply only
Provide a clear, actionable response. Cite which source your answer comes from."""


# ── Gemma via Ollama Cloud ────────────────────

def call_gemma(prompt: str, system: str = SYSTEM_PROMPT) -> str:
    """Call Gemma via Ollama (local or cloud routing)."""
    payload = {
        "model":  OLLAMA_MODEL,
        "system": system,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "top_p":       0.9
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["response"]

    except requests.exceptions.ConnectionError:
        print("❌ Ollama not running! Start Ollama in the background.")
        return "⚠️ I cannot connect to the AI right now. Please try again in a moment."
    except Exception as e:
        print(f"❌ Ollama error: {e}")
        return "⚠️ I'm having trouble generating a response. Please try again."


# ── Audio transcription (Gemini Flash STT) ────

def transcribe_audio(audio_url: str) -> str:
    """Downloads WhatsApp voice note and transcribes via Gemini Flash."""
    try:
        print("🎙️ Transcribing voice note via Gemini Flash...")

        audio_response = requests.get(
            audio_url,
            auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
            timeout=10
        )
        audio_response.raise_for_status()

        base64_audio = base64.b64encode(audio_response.content).decode("utf-8")

        url     = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GOOGLE_API_KEY}"
        payload = {
            "contents": [{
                "parts": [
                    {"text": "Transcribe this emergency voice note accurately. Return ONLY the transcribed text. No formatting, no commentary."},
                    {"inline_data": {"mime_type": "audio/ogg", "data": base64_audio}}
                ]
            }]
        }

        resp = requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()

        transcription = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        print(f"🎙️ Transcribed: {transcription}")
        return transcription.strip()

    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return ""


# ── Image analysis (Gemma Vision) ────────────

def analyze_image_background(image_url: str, from_number: str, bot_number: str):
    """
    FIX 2: Image analysis runs in background task — no more Twilio timeouts.
    Downloads image, analyzes via Gemma vision, sends reply via Twilio REST.
    """
    try:
        print("📸 Analyzing image via Gemma vision...")

        img_response = requests.get(
            image_url,
            auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
            timeout=10
        )
        img_response.raise_for_status()

        base64_img = base64.b64encode(img_response.content).decode("utf-8")

        # FIX 7: Use OLLAMA_MODEL constant, not hardcoded string
        payload = {
            "model":  OLLAMA_MODEL,
            "prompt": "You are a crisis response medical assistant. Look at this image carefully. Describe what you see and provide concise, life-saving first aid steps based strictly on IFRC or TCCC standards. Be direct and actionable. Keep under 150 words.",
            "images": [base64_img],
            "stream": False,
            "options": {"temperature": 0.2}
        }

        resp = requests.post(OLLAMA_URL, json=payload, timeout=90)
        resp.raise_for_status()

        reply = resp.json()["response"].strip()

        # Safeguard truncation
        if len(reply) > 1590:
            reply = reply[:1585] + "..."

    except Exception as e:
        print(f"❌ Vision error: {e}")
        reply = "⚠️ I could not analyze the image. Please describe the injury or situation in text."

    # Send reply via Twilio REST
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        client.messages.create(from_=bot_number, body=reply, to=from_number)
        print("✅ Image analysis reply sent!")
    except Exception as e:
        print(f"❌ Twilio error sending image reply: {e}")


# ── SOS handler ───────────────────────────────

def handle_sos(from_number: str) -> str:
    """FIX 1: Accepts from_number parameter — was missing, caused crash."""
    return (
        "🆘 *SOS RECEIVED*\n\n"
        "1. Stay calm and stay where you are if safe\n"
        "2. Your message has been logged\n"
        "3. Share your GPS coordinates or describe your surroundings\n"
        "4. If injured, type INJURED — I will guide you through first aid\n"
        "5. If you need water or shelter, type SHELTER\n\n"
        "What is your current situation?"
    )


# ── Intent detection ──────────────────────────

def detect_intent(message: str) -> str:
    """Route messages by intent."""
    msg = message.lower().strip()
    sos_keywords = ["sos", "help me", "emergency", "dying", "trapped", "mayday"]
    if any(kw in msg for kw in sos_keywords):
        return "sos"
    return "query"


# ── Background RAG worker ─────────────────────

def async_rag_and_send(
    user_message: str,
    from_number:  str,
    host_url:     str,
    bot_number:   str,
    is_audio_input: bool
):
    """Runs full RAG pipeline in background and pushes reply to WhatsApp."""

    # 1. Translate non-English to English for better search
    search_query = translate_to_english(user_message)
    print(f"🔍 [BG] Searching for: {search_query}")

    # 2. Hybrid search using English translation
    retrieved = hybrid_search(search_query)

    # 3. Assemble prompt with ORIGINAL message (so Gemma replies in user's language)
    prompt = assemble_prompt(user_message, retrieved)

    print(f"🤖 [BG] Calling {OLLAMA_MODEL}...")
    reply = call_gemma(prompt)

    # Hard truncation safeguard
    if len(reply) > 1590:
        reply = reply[:1585] + "..."

    # 4. Optionally generate audio response
    media_url = None
    if is_audio_input:
        print("🔊 [BG] Generating voice response...")
        try:
            try:
                detected_lang = detect(reply)
            except Exception:
                detected_lang = "en"

            print(f"   Language detected: {detected_lang}")
            tts      = gTTS(text=reply, lang=detected_lang, slow=False)
            filename = f"{uuid.uuid4().hex}.mp3"
            filepath = AUDIO_DIR / filename
            tts.save(str(filepath))

            media_url = f"{host_url}/audio/{filename}"
            print(f"   Audio saved: {filepath.name}")

            # FIX 3: Schedule cleanup after delivery
            schedule_file_cleanup(filepath)

        except Exception as e:
            print(f"⚠️ Audio generation failed: {e} — text-only fallback")

    # 5. Send via Twilio REST
    print(f"📤 [BG] Sending to {from_number}")
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

        # Always send text first — works even on lowest connectivity
        client.messages.create(from_=bot_number, body=reply, to=from_number)
        print("✅ Text delivered!")

        # Follow up with audio if available
        if media_url:
            client.messages.create(
                from_=bot_number,
                media_url=[media_url],
                to=from_number
            )
            print("✅ Audio delivered!")

    except Exception as e:
        print(f"❌ Twilio delivery error: {e}")


# ── Main WhatsApp webhook ─────────────────────

@app.post("/bot")
async def whatsapp_bot(
    request:          Request,
    background_tasks: BackgroundTasks,
    Body:             str  = Form(""),
    From:             str  = Form(...),
    To:               str  = Form(...),
    MediaUrl0:        str  = Form(None),
    MediaContentType0: str = Form(None)
):
    """Main webhook — routes Text, Audio, and Images."""

    # FIX 4: Keep original case for language detection — only lower for intent check
    user_message = Body.strip()
    from_number  = From
    bot_number   = To
    reply        = ""
    is_audio_input = False

    print(f"\n📱 Incoming from {from_number} | media: {MediaContentType0 or 'none'}")

    # 1. Image — route to background (FIX 2: was synchronous, now async)
    if MediaContentType0 and MediaContentType0.startswith("image/"):
        background_tasks.add_task(
            analyze_image_background, MediaUrl0, from_number, bot_number
        )
        return PlainTextResponse("<Response></Response>", media_type="application/xml")

    # 2. Audio voice note
    elif MediaContentType0 and MediaContentType0.startswith("audio/"):
        is_audio_input = True
        user_message   = transcribe_audio(MediaUrl0)
        if not user_message:
            reply = "⚠️ I could not hear that clearly. Please type your emergency."

    # 3. Text or transcribed audio
    if not reply and user_message:
        intent = detect_intent(user_message)

        if intent == "sos":
            # FIX 1: Pass from_number — was missing
            reply = handle_sos(from_number)
            print("🆘 SOS — sending emergency response")

        else:
            host_url = str(request.base_url).rstrip("/")
            background_tasks.add_task(
                async_rag_and_send,
                user_message, from_number, host_url, bot_number, is_audio_input
            )
            return PlainTextResponse("<Response></Response>", media_type="application/xml")

    # Synchronous reply (SOS or audio error)
    twiml = MessagingResponse()
    twiml.message(reply)
    return PlainTextResponse(str(twiml), media_type="application/xml")


# ── Health check ──────────────────────────────

@app.get("/")
async def health():
    # FIX 8/11: Returns actual model being used, not stale variable
    return {
        "status":      "online",
        "chunks":      collection.count(),
        "model":       OLLAMA_MODEL,
        "embed_model": EMBED_MODEL,
        "top_k":       TOP_K
    }


# ── Test endpoint ─────────────────────────────

@app.get("/test")
async def test_query(q: str = "how to stop bleeding"):
    """Test full pipeline without WhatsApp."""
    docs   = hybrid_search(q)
    prompt = assemble_prompt(q, docs)
    reply  = call_gemma(prompt)
    return {
        "query":   q,
        "sources": [d.metadata.get("source") for d in docs],
        "reply":   reply
    }