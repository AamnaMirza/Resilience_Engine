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
7. Appends country-specific emergency numbers (from emergency_contact.json)
8. Vision pipeline now uses ChromaDB context alongside Gemma vision

Run:
    uvicorn query_pipeline_fixed:app --reload --port 8000

Then in another terminal:
    ngrok http 8000

Paste the ngrok URL + /bot into Twilio sandbox webhook.
"""

# ── Imports ───────────────────────────────────
import os
import re
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

# Path to the emergency contacts JSON file
EMERGENCY_JSON_PATH = Path(r"C:\Users\Aamna\Documents\Resilience Engine\emergency_contact.json")

OLLAMA_MODEL = "gemma4:31b-cloud"
OLLAMA_URL   = "http://localhost:11434/api/generate"

TOP_K = 6

AUDIO_CLEANUP_DELAY = 120

# ─────────────────────────────────────────────

app = FastAPI()

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

# ── Load emergency numbers from JSON once at startup ──────────────────────────
# JSON format (emergency_contact.json):
#   { "IN": { "calling_code": "91", "police": ["100","112"],
#             "ambulance": ["108","102"], "fire": ["101","112"] }, ... }
# Keys are ISO 3166-1 alpha-2 country codes.
# We build TWO lookup dicts for fast retrieval:
#   EMERGENCY_BY_CALLING_CODE  — "91"    → formatted string
#   EMERGENCY_BY_ISO           — "IN"    → formatted string  (fallback)

EMERGENCY_BY_CALLING_CODE: dict[str, str] = {}
EMERGENCY_BY_ISO:          dict[str, str] = {}


def _load_emergency_json() -> None:
    """
    Load emergency_contact.json and populate both lookup dicts.
    Each entry is formatted as a compact WhatsApp-friendly string, e.g.:
        🚨 IN emergency: Police: 100, 112 | Ambulance: 108, 102 | Fire: 101, 112
    """
    if not EMERGENCY_JSON_PATH.exists():
        print(f"⚠️  Emergency JSON not found: {EMERGENCY_JSON_PATH}")
        return

    import json
    with open(EMERGENCY_JSON_PATH, encoding="utf-8") as f:
        data = json.load(f)

    for iso_code, entry in data.items():
        calling_code = str(entry.get("calling_code", "")).strip()

        # Build labelled parts — skip service if list is empty
        parts = []
        for service, label in [("police", "Police"), ("ambulance", "Ambulance"), ("fire", "Fire")]:
            nums = entry.get(service, [])
            if nums:
                # Deduplicate while preserving order
                seen_nums: list[str] = []
                for n in nums:
                    if n not in seen_nums:
                        seen_nums.append(n)
                parts.append(f"{label}: {', '.join(seen_nums)}")

        if not parts:
            continue

        formatted = f"🚨 {iso_code} emergency: " + " | ".join(parts)

        if calling_code:
            EMERGENCY_BY_CALLING_CODE[calling_code] = formatted
        EMERGENCY_BY_ISO[iso_code] = formatted

    print(f"✅ Emergency contacts loaded: {len(EMERGENCY_BY_CALLING_CODE)} by calling code, "
          f"{len(EMERGENCY_BY_ISO)} by ISO code")


_load_emergency_json()


def get_emergency_numbers(from_number: str) -> str:
    """
    Given a Twilio WhatsApp number like 'whatsapp:+919876543210',
    extracts the E.164 calling code and returns the formatted emergency string.

    Strategy:
      1. Strip non-digits from from_number  → e.g. "919876543210"
      2. Try calling codes longest-first (4→3→2→1 digits) against EMERGENCY_BY_CALLING_CODE
         This ensures "+1868" (Trinidad) matches before "+1" (USA/Canada)
      3. If still no match, return ""

    Debug tip: hit GET /test-emergency?number=whatsapp:+919876543210
    """
    digits = re.sub(r"\D", "", from_number)   # "whatsapp:+919..." → "919..."

    for length in (4, 3, 2, 1):
        prefix = digits[:length]
        if prefix in EMERGENCY_BY_CALLING_CODE:
            print(f"📞 Emergency lookup: +{prefix} → {EMERGENCY_BY_CALLING_CODE[prefix]}")
            return EMERGENCY_BY_CALLING_CODE[prefix]

    print(f"📞 Emergency lookup: no match for digits prefix of '{from_number}'")
    return ""


# ─────────────────────────────────────────────

print(f"✅ BM25 index built with {len(bm25_docs)} documents")
print(f"✅ Model: {OLLAMA_MODEL}")
print("✅ All systems ready — bot is live!\n")


# ── Utilities ─────────────────────────────────

def schedule_file_cleanup(filepath: Path, delay: int = AUDIO_CLEANUP_DELAY):
    """Delete audio files after sending to prevent disk filling up."""
    def _delete():
        filepath.unlink(missing_ok=True)
        print(f"🗑️ Cleaned up audio: {filepath.name}")
    threading.Timer(delay, _delete).start()


# ── Translation ───────────────────────────────

def translate_to_english(text: str) -> str:
    """
    Translates non-English queries to English for better RAG search.
    Only skip translation for pure ASCII.
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

    bm25_results = bm25_retriever.invoke(query)

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

SYSTEM_PROMPT = """You are an expert crisis response medic and survival instructor. You speak directly to someone who may be dying right now.

Your job is to give IMMEDIATE, PRACTICAL guidance — like a calm expert physically present with them.
You have studied TCCC, IFRC First Aid, Sphere Standards, WHO WASH, and the US Army Survival Manual deeply.
The reference material below is your knowledge base — use it to inform your answer, but DO NOT quote or list it.

HOW TO RESPOND:
- Speak in direct, human sentences — NOT as a list of source citations
- Say things like "Press firmly on the wound with your palm" NOT "[Source 3] Apply direct pressure"
- Number your steps clearly (1, 2, 3...) but write each step as a real instruction
- You are a SYNTHESIZER, not a document reader — combine your training + the references into real advice
- Start immediately with the most critical action — no preamble, no "based on the sources..."
- If the references cover the topic: use them silently to back your steps, never quote "[Source N]"
- If the references don't cover it: use your own medical/survival knowledge and say so briefly
- NEVER say "consult a doctor" as the only answer — give hands-on first aid steps first

CLINICAL RULES:
- Heart Attack (conscious, chest pain) → sit them down, loosen clothing, chew aspirin. NOT CPR.
- Cardiac Arrest (unconscious, not breathing) → CPR immediately
- Outdated advice (urine on wounds, butter on burns, sucking venom) → correct it silently, give modern steps only
- Keep response under 1200 characters for WhatsApp

LANGUAGE RULE — THIS IS ABSOLUTE:
- Detect the language of the user's question
- Reply in THAT language and ONLY that language — every single word
- NEVER mix languages. NEVER add English translations in parentheses
- Hindi question → full Hindi reply. Arabic question → full Arabic reply. English → English."""


def assemble_prompt(user_query: str, retrieved_docs: list[Document]) -> str:
    """
    Build the full prompt with retrieved context.
    Sources are labelled internally for Gemma's reference but the system prompt
    explicitly tells it NOT to echo '[Source N]' in its reply.
    """
    context_parts = []
    for i, doc in enumerate(retrieved_docs):
        source = doc.metadata.get("source", "Unknown source")
        text   = doc.page_content.strip()
        context_parts.append(f"[Reference {i+1} — {source}]:\n{text}")

    context = "\n\n---\n\n".join(context_parts)

    return f"""KNOWLEDGE BASE (use this to inform your answer — do NOT quote or list these as sources):
{context}

USER'S MESSAGE: {user_query}

RESPOND NOW in the EXACT language of the USER'S MESSAGE above.
Give direct, numbered, actionable steps. Do NOT cite [Reference N] in your reply.
If you draw on the knowledge base, do so silently — speak as an expert, not as a document reader."""


def assemble_vision_prompt(
    vision_description: str,
    retrieved_docs: list[Document],
    user_caption: str = "",
    reply_language: str = "English"
) -> str:
    """
    Build a prompt combining Gemma's visual description + ChromaDB context.
    If the user sent a caption with the image, that becomes the primary question.
    reply_language forces the output language explicitly.
    """
    context_parts = []
    for i, doc in enumerate(retrieved_docs):
        source = doc.metadata.get("source", "Unknown source")
        text   = doc.page_content.strip()
        context_parts.append(f"[Reference {i+1} — {source}]:\n{text}")

    context = "\n\n---\n\n".join(context_parts)

    # If user sent a caption, make it the primary question; image adds visual context
    if user_caption:
        question_block = f"""THE PERSON WROTE (in their message with this image):
\"{user_caption}\"

WHAT THE IMAGE SHOWS (your visual analysis):
{vision_description}"""
    else:
        question_block = f"""WHAT THE IMAGE SHOWS (your visual analysis):
{vision_description}"""

    return f"""A person in an emergency sent you an image{' with a message' if user_caption else ''}.

{question_block}

KNOWLEDGE BASE (use silently — do NOT quote or list these):
{context}

YOUR TASK:
Answer based on what you see in the image AND what the person wrote (if anything).
Use the knowledge base to back your steps — but speak as an expert medic, not a document reader.
Give clear, numbered, actionable steps. Do NOT cite [Reference N].
Keep under 1200 characters.

LANGUAGE RULE — ABSOLUTE: Respond ENTIRELY in {reply_language}.
Every single word must be in {reply_language}. Zero English if {reply_language} is not English."""


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


# ── Image analysis (Gemma Vision + ChromaDB RAG) ──────────────────────────────

# langdetect language code → human name for prompt injection
_LANG_NAMES = {
    "hi": "Hindi", "ar": "Arabic", "fr": "French", "es": "Spanish",
    "pt": "Portuguese", "de": "German", "ur": "Urdu", "bn": "Bengali",
    "ru": "Russian", "zh-cn": "Chinese", "zh-tw": "Chinese",
    "ja": "Japanese", "ko": "Korean", "tr": "Turkish", "fa": "Persian",
    "en": "English",
}

def _detect_language_name(text: str) -> str:
    """Return human-readable language name for prompt injection, default English."""
    if not text or text.isascii():
        return "English"
    try:
        code = detect(text)
        return _LANG_NAMES.get(code, "English")
    except Exception:
        return "English"


def analyze_image_background(
    image_url:    str,
    from_number:  str,
    bot_number:   str,
    host_url:     str = "",
    user_caption: str = "",          # ← NEW: the text sent alongside the image
):
    """
    Three-stage image pipeline:
      Stage 1 — Gemma Vision:  describe what the image shows in English (internal only)
      Stage 2 — ChromaDB RAG:  use caption (if any) + visual description to retrieve chunks
      Stage 3 — Final answer:  Gemma synthesizes image + handbook knowledge,
                               replies in the user's language (detected from caption)
    """
    # Detect reply language from caption — fall back to English if no caption
    reply_language = _detect_language_name(user_caption) if user_caption else "English"
    print(f"🌐 [Vision] Reply language detected: {reply_language}")

    try:
        print("📸 [Vision Stage 1] Downloading image...")
        img_response = requests.get(
            image_url,
            auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
            timeout=10
        )
        img_response.raise_for_status()
        base64_img = base64.b64encode(img_response.content).decode("utf-8")

        # ── Stage 1: Pure vision — describe what Gemma sees (always in English for search) ──
        print("📸 [Vision Stage 1] Asking Gemma to describe the image...")

        # Enrich the vision prompt if the user sent a caption
        caption_hint = (
            f" The person described their situation as: \"{user_caption}\". "
            "Use this to focus your description."
        ) if user_caption else ""

        vision_payload = {
            "model":  OLLAMA_MODEL,
            "prompt": (
                "You are a crisis medical triage assistant. Look at this image carefully."
                f"{caption_hint} "
                "Describe in plain English: (1) what type of injury or emergency is visible, "
                "(2) which body part or situation is affected, (3) visible severity. "
                "Be specific and factual. Output ONLY the description — no advice yet. "
                "Keep under 100 words."
            ),
            "images": [base64_img],
            "stream": False,
            "options": {"temperature": 0.1}
        }

        vision_resp = requests.post(OLLAMA_URL, json=vision_payload, timeout=90)
        vision_resp.raise_for_status()
        vision_description = vision_resp.json()["response"].strip()
        print(f"   👁️ Vision description: {vision_description}")

        # ── Stage 2: Search ChromaDB using caption (if available) + visual description ──
        print("📸 [Vision Stage 2] Searching ChromaDB...")
        # Prefer caption for search if available (more specific than visual description)
        search_text  = user_caption if user_caption else vision_description
        search_query = translate_to_english(search_text)
        retrieved    = hybrid_search(search_query)
        print(f"   📚 Retrieved {len(retrieved)} handbook chunks")

        # ── Stage 3: RAG-grounded, language-correct final response ────────────
        print(f"📸 [Vision Stage 3] Generating response in {reply_language}...")
        rag_prompt = assemble_vision_prompt(
            vision_description, retrieved,
            user_caption=user_caption,
            reply_language=reply_language
        )
        reply = call_gemma(rag_prompt)

        # Append emergency numbers
        emergency_info = get_emergency_numbers(from_number)
        if emergency_info:
            reply = reply.rstrip() + f"\n\n{emergency_info}"

        if len(reply) > 1590:
            reply = reply[:1585] + "..."

    except Exception as e:
        print(f"❌ Vision pipeline error: {e}")
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
    emergency_info = get_emergency_numbers(from_number)
    base = (
        "🆘 *SOS RECEIVED*\n\n"
        "1. Stay calm and stay where you are if safe\n"
        "2. Your message has been logged\n"
        "3. Share your GPS coordinates or describe your surroundings\n"
        "4. If injured, type INJURED — I will guide you through first aid\n"
        "5. If you need water or shelter, type SHELTER\n\n"
        "What is your current situation?"
    )
    if emergency_info:
        base += f"\n\n{emergency_info}"
    return base


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
    user_message:   str,
    from_number:    str,
    host_url:       str,
    bot_number:     str,
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

    # 4. Append country-specific emergency numbers
    emergency_info = get_emergency_numbers(from_number)
    if emergency_info:
        print(f"📞 [BG] Appending emergency numbers: {emergency_info}")
        reply = reply.rstrip() + f"\n\n{emergency_info}"

    # Hard truncation safeguard
    if len(reply) > 1590:
        reply = reply[:1585] + "..."

    # 5. Optionally generate audio response
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

            schedule_file_cleanup(filepath)

        except Exception as e:
            print(f"⚠️ Audio generation failed: {e} — text-only fallback")

    # 6. Send via Twilio REST
    print(f"📤 [BG] Sending to {from_number}")
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

        client.messages.create(from_=bot_number, body=reply, to=from_number)
        print("✅ Text delivered!")

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
    request:           Request,
    background_tasks:  BackgroundTasks,
    Body:              str = Form(""),
    From:              str = Form(...),
    To:                str = Form(...),
    MediaUrl0:         str = Form(None),
    MediaContentType0: str = Form(None)
):
    """Main webhook — routes Text, Audio, and Images."""

    user_message   = Body.strip()
    from_number    = From
    bot_number     = To
    reply          = ""
    is_audio_input = False

    print(f"\n📱 Incoming from {from_number} | media: {MediaContentType0 or 'none'}")

    host_url = str(request.base_url).rstrip("/")

    # 1. Image — three-stage Vision+RAG pipeline in background
    if MediaContentType0 and MediaContentType0.startswith("image/"):
        # user_message here is the caption typed alongside the image (may be empty "")
        print(f"   📎 Image caption: '{user_message or 'none'}'")
        background_tasks.add_task(
            analyze_image_background,
            MediaUrl0, from_number, bot_number, host_url,
            user_message  # ← caption forwarded; empty string if none sent
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
            reply = handle_sos(from_number)
            print("🆘 SOS — sending emergency response")

        else:
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
    return {
        "status":           "online",
        "chunks":           collection.count(),
        "model":            OLLAMA_MODEL,
        "embed_model":      EMBED_MODEL,
        "top_k":            TOP_K,
        "emergency_countries": len(EMERGENCY_BY_CALLING_CODE)
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


@app.get("/test-emergency")
async def test_emergency(number: str = "whatsapp:+919876543210"):
    """Test emergency number lookup. Pass ?number=whatsapp:+XXXXXXXXXXX"""
    result = get_emergency_numbers(number)
    return {
        "input":  number,
        "result": result or "No emergency numbers found for this country code"
    }