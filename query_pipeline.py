"""
Crisis Response RAG — Query Pipeline
=====================================
FastAPI WhatsApp bot that:
1. Receives WhatsApp messages via Twilio webhook
2. Runs hybrid search (BM25 + ChromaDB) on crisis manuals
3. Assembles a prompt with retrieved context
4. Calls Gemma 4 via Google AI Studio
5. Sends response back via Twilio WhatsApp

Run:
    uvicorn query_pipeline:app --reload --port 8000

Then in another terminal:
    ngrok http 8000

Paste the ngrok URL + /bot into Twilio sandbox webhook.
"""

import os
import json
import requests
from pathlib import Path
import base64

from fastapi import FastAPI, Request, Form, BackgroundTasks
from fastapi.responses import PlainTextResponse
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
from dotenv import load_dotenv

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
load_dotenv()

GOOGLE_API_KEY        = os.getenv("GOOGLE_API_KEY")
TWILIO_ACCOUNT_SID    = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN     = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")

CHROMA_DIR      = r"C:\Users\Aamna\Documents\Resilience Engine\chroma_db"
COLLECTION_NAME = "crisis_knowledge"
EMBED_MODEL     = "all-MiniLM-L6-v2"
GEMMA_MODEL     = "gemini-2.5-flash"   # Gemma 4 on AI Studio

# How many chunks to retrieve
TOP_K = 9

# ─────────────────────────────────────────────
from pathlib import Path
import uuid
from gtts import gTTS
from langdetect import detect
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# 🚀 BULLETPROOF ABSOLUTE PATHS FOR AUDIO
BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "audio_cache"
AUDIO_DIR.mkdir(exist_ok=True)

app.mount("/audio", StaticFiles(directory=str(AUDIO_DIR)), name="audio")

# ── Load resources at startup ─────────────────

print("Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)

print("Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection     = chroma_client.get_collection(COLLECTION_NAME)

# Load all docs for BM25 (keyword search)
print("Loading BM25 index...")
all_results = collection.get(include=["documents", "metadatas"])
bm25_docs   = [
    Document(page_content=doc, metadata=meta)
    for doc, meta in zip(all_results["documents"], all_results["metadatas"])
]
bm25_retriever = BM25Retriever.from_documents(bm25_docs)
bm25_retriever.k = TOP_K

print(f"✅ BM25 index built with {len(bm25_docs)} documents")
print("✅ All systems ready — bot is live!\n")

def translate_to_english(text: str) -> str:
    """Translates non-English queries to English using Gemma via Ollama."""
    # If the text is mostly standard English characters, skip to save time
    if text.isascii():
        return text 
        
    print(f"🌍 Non-English detected. Translating with Gemma 27B...")
    
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "gemma3:27b-cloud", 
        "system": "You are a professional translator. Translate the user's text into English. Output ONLY the raw English translation. Do not include quotation marks, conversational filler, or explanations.",
        "prompt": text,
        "stream": False,
        "options": {
            "temperature": 0.1  # Low temperature for strict, robotic translation
        }
    }
    
    try:
        resp = requests.post(url, json=payload, timeout=20)
        resp.raise_for_status()
        english_text = resp.json()["response"].strip()
        
        # Clean up any rogue quotes Gemma might try to add
        english_text = english_text.strip('"').strip("'")
        
        print(f"   Translated Query: {english_text}")
        return english_text
    except Exception as e:
        print(f"⚠️ Gemma translation error: {e}")
        return text # Fallback to original text if the cloud bridge fails

# ── Hybrid search ─────────────────────────────

def hybrid_search(query: str) -> list[Document]:
    """
    Combines BM25 (keyword) and dense vector (semantic) search.
    BM25 catches acronyms like CAT, TCCC, WASH.
    Dense vectors catch cross-language semantic meaning.
    Reciprocal rank fusion merges both result sets.
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

    # Reciprocal rank fusion — merge and re-rank both lists
    scores = {}
    k = 60  # RRF constant

    for rank, doc in enumerate(vector_docs):
        key = doc.page_content[:100]
        scores[key] = scores.get(key, 0) + 1 / (rank + k)

    for rank, doc in enumerate(bm25_results):
        key = doc.page_content[:100]
        scores[key] = scores.get(key, 0) + 1 / (rank + k)

    # Combine all unique docs
    all_docs = {doc.page_content[:100]: doc for doc in vector_docs + bm25_results}

    # Sort by RRF score
    ranked = sorted(all_docs.items(), key=lambda x: scores.get(x[0], 0), reverse=True)

    return [doc for _, doc in ranked[:TOP_K]]


# ── Prompt assembly ───────────────────────────

SYSTEM_PROMPT = """You are a crisis response assistant helping people during wars, natural disasters, and emergencies.
You have access to verified emergency handbooks including TCCC (combat casualty care), IFRC First Aid Guidelines, Sphere Humanitarian Standards, WHO WASH guidelines, and the US Army Survival Manual.

CRITICAL RULES:
- Give clear, numbered, actionable steps
- Be concise — the person may be in danger right now
- If the question is medical, prioritize life-saving steps first
- Never say "consult a doctor" as the ONLY answer — give immediate first aid steps
- Donot confuse one emergency with another unless explicitly stated in the context
- If you don't know something, say so clearly rather than guessing or giving any hallucinated answer
- Respond in the SAME LANGUAGE the user wrote in
- MEDICAL TRIAGE: Never confuse a Heart Attack (conscious, chest pain -> sit down, chew aspirin) with Cardiac Arrest (unconscious, not breathing -> CPR). If the user just says "heart attack", assume conscious first aid, do NOT immediately recommend CPR.
- Keep responses under 1200 characters for strict WhatsApp readability
- If the answer is not explicitly detailed in the context, reply: 'I cannot verify this procedure. Please seek human medical or humanitarian assistance.
- Critically evaluate the context. If you encounter outdated or unsafe wilderness medical advice (e.g., using urine to clean wounds, sucking out snake venom, applying butter to burns), explicitly warn the user NOT to do it and default strictly to modern IFRC standards.
- Donot mention outdated unsafe advice
- EVERY word you generate, including any warnings or disclaimers, MUST be translated into the user's language.
- EVERY word you generate MUST be translated into the exact language the user used in their QUESTION. 
- STRICT RULE: NEVER output bilingual text. DO NOT add English translations in parentheses. Output strictly and ONLY in the user's language."""

def assemble_prompt(user_query: str, retrieved_docs: list[Document]) -> str:
    """Build the full prompt with retrieved context."""
    context_parts = []
    seen_sources  = set()

    for i, doc in enumerate(retrieved_docs):
        source = doc.metadata.get("source", "Unknown source")
        text   = doc.page_content.strip()
        if source not in seen_sources:
            seen_sources.add(source)
        context_parts.append(f"[Source {i+1}: {source}]\n{text}")

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""REFERENCE MATERIAL:
{context}

QUESTION: {user_query}

CRITICAL INSTRUCTION: You must reply in the EXACT SAME LANGUAGE that the user used in the QUESTION. 
- If the QUESTION is in English, reply ONLY in English.
- If the QUESTION is in Hindi, reply ONLY in Hindi.
- If the QUESTION is in Portuguese, reply ONLY in Portuguese.
Do NOT translate English into another language. Provide a clear, actionable medical response."""
    return prompt


# ── Gemma 4 via AI Studio ─────────────────────

def call_gemma(prompt: str, system: str = SYSTEM_PROMPT) -> str:
    """Call Gemma 4 via Ollama (which routes to Ollama Cloud)."""
    
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": "gemma3:27b-cloud", 
        "system": system,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "top_p": 0.9
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data["response"]
        
    except requests.exceptions.ConnectionError:
        print("Ollama Connection Error: Make sure Ollama is running in the background!")
        return "⚠️ I cannot connect to the local AI. Please ensure Ollama is running."
    except Exception as e:
        print(f"Ollama API error: {e}")
        return "⚠️ I'm having trouble generating a response right now. Please try again."

# ── SOS handler ───────────────────────────────

def handle_sos(from_number: str) -> str:
    """
    Handle SOS messages — highest priority.
    In a real deployment this would also alert NGO contacts.
    """
    return (
        "🆘 *SOS RECEIVED*\n\n"
        "1. Stay calm and stay where you are if safe\n"
        "2. Your message has been logged\n"
        "3. Send your location: share GPS coordinates or describe your surroundings\n"
        "4. If injured, type INJURED and I'll guide you through first aid\n"
        "5. If you need water/shelter, type SHELTER\n\n"
        "What is your current situation?"
    )

# ── Multimodal Handlers (Audio & Image) ───────────────

def transcribe_audio(audio_url: str) -> str:
    """Downloads WhatsApp voice note and transcribes it via Gemini API."""
    try:
        print("🎙️ Processing voice note...")
        
        # THE FIX: Add Twilio Auth to bypass the security wall
        audio_response = requests.get(
            audio_url, 
            auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
            timeout=10
        )
        audio_response.raise_for_status()
        audio_data = audio_response.content
        
        base64_audio = base64.b64encode(audio_data).decode("utf-8")
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GOOGLE_API_KEY}"
        payload = {
            "contents": [{
                "parts": [
                    {"text": "Transcribe this emergency voice note accurately. Only return the transcribed text. Do not add formatting."},
                    {"inline_data": {"mime_type": "audio/ogg", "data": base64_audio}}
                ]
            }]
        }
        resp = requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        
        transcription = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        print(f"🎙️ Transcription success: {transcription}")
        return transcription.strip()
        
    except Exception as e:
        print(f"❌ Audio transcription error: {e}")
        return ""

def extract_vision_intent(image_url: str) -> str:
    """Pass 1: Performs visual triage to extract a search query."""
    try:
        print("📸 Pass 1: Gemma 4 performing visual triage...")
        auth = (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        img_response = requests.get(image_url, auth=auth, timeout=10)
        img_response.raise_for_status()
        
        base64_img = base64.b64encode(img_response.content).decode("utf-8")
        
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": "gemma4:31b-cloud", 
            "prompt": "Analyze this medical image. Provide a 1-sentence English medical summary of the injury or emergency to be used for a database search. Output ONLY the summary.",
            "images": [base64_img], # Native multimodal input
            "stream": False
        }
        
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        intent = resp.json()["response"].strip()
        print(f"   Extracted Vision Intent: {intent}")
        return intent
    except Exception as e:
        print(f"❌ Vision Pass 1 error: {e}")
        return ""

# ── Intent detection ──────────────────────────

def detect_intent(message: str) -> str:
    """Simple intent detection to route messages."""
    msg = message.lower().strip()

    sos_keywords = ["sos", "help me", "emergency", "dying", "trapped", "mayday"]
    if any(kw in msg for kw in sos_keywords):
        return "sos"

    return "query"

# ── Asynchronous AI Worker ────────────────────────────

# Pass the is_audio_input flag into the function
def async_rag_and_send(intent_query: str, from_number: str, host_url: str, bot_number: str, is_audio_input: bool, original_media_url: str = None, media_type: str = "text"):
    """Pass 2 (Search) and Pass 3 (Grounded Multimodal Reasoning)."""
    
    # --- Pass 2: The Search ---
    augmented_query = f"{intent_query} global emergency contact numbers"
    retrieved = hybrid_search(augmented_query) 
    context_text = "\n\n".join([doc.page_content for doc in retrieved])
    
    # --- Pass 3: Final Reasoning ---
    media_binary = None
    if original_media_url:
        media_binary = base64.b64encode(requests.get(
            original_media_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        ).content).decode("utf-8")

    # Enable Thinking Mode with <|think|> for frontier reasoning
    final_prompt = f"""<|think|>
VERIFIED MEDICAL CONTEXT:
{context_text}

USER LOCATION CODE: {from_number}
IDENTIFIED EMERGENCY: {intent_query}

Final Instruction: Using the medical context and the visual/audio evidence, provide life-saving first aid steps in the user's language. Be concise and actionable."""

    payload = {
        "model": "gemma4:31b-cloud",
        "prompt": final_prompt,
        "images": [media_binary] if media_binary else [],
        "stream": False,
        "options": {"temperature": 1.0, "top_p": 0.95}
    }
    
    print(f"🤖 Pass 3: Gemma 4 performing grounded reasoning...")
    reply = requests.post("http://localhost:11434/api/generate", json=payload).json()["response"]
    
    # Truncate for WhatsApp limits
    if len(reply) > 1590: reply = reply[:1585] + "..."
    
    media_url = None
    if is_audio_input: # Fixed flag logic
        print(f"🔊 Generating voice note...")
        try:
            detected_lang = detect(reply)
            tts = gTTS(text=reply, lang=detected_lang, slow=False) 
            filename = f"{uuid.uuid4().hex}.mp3"
            filepath = AUDIO_DIR / filename
            tts.save(str(filepath))
            media_url = f"{host_url}/audio/{filename}"
        except Exception as e: print(f"Audio generation failed: {e}")

    # --- Send via Twilio ---
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        client.messages.create(from_=bot_number, body=reply, to=from_number)
        if media_url:
            client.messages.create(from_=bot_number, media_url=[media_url], to=from_number)
    except Exception as e: print(f"❌ Twilio Error: {e}")

# ── Main WhatsApp webhook ─────────────────────

@app.post("/bot")
async def whatsapp_bot(
    request: Request, background_tasks: BackgroundTasks,
    Body: str = Form(""), From: str = Form(...), To: str = Form(...),
    MediaUrl0: str = Form(None), MediaContentType0: str = Form(None)
):
    user_message = Body.strip()
    from_number, bot_number = From, To
    host_url = str(request.base_url).rstrip("/")
    is_audio_input = False

    # 1. Handle Images (Vision RAG)
    if MediaContentType0 and MediaContentType0.startswith("image/"):
        vision_intent = extract_vision_intent(MediaUrl0)
        if vision_intent:
            background_tasks.add_task(async_rag_and_send, vision_intent, from_number, host_url, bot_number, False, MediaUrl0, "image")
            return PlainTextResponse("<Response></Response>", media_type="application/xml")
        return PlainTextResponse(str(MessagingResponse().message("⚠️ Image analysis failed.")), media_type="application/xml")

    # 2. Handle Audio (Voice RAG)
    elif MediaContentType0 and MediaContentType0.startswith("audio/"):
        is_audio_input = True
        user_message = transcribe_audio(MediaUrl0)

    # 3. Handle Text or Audio
    if not reply and user_message:
        intent = detect_intent(user_message)

        if intent == "sos":
            reply = handle_sos()
            print(f"🆘 SOS detected — sending emergency response")
        else:
            host_url = str(request.base_url).rstrip("/")
            
            # 🚀 PASS THE FLAG TO THE BACKGROUND WORKER
            background_tasks.add_task(async_rag_and_send, user_message, from_number, host_url, bot_number, is_audio_input)
            
            return PlainTextResponse("<Response></Response>", media_type="application/xml")

    # Send standard synchronous reply
    twiml = MessagingResponse()
    twiml.message(reply)
    return PlainTextResponse(str(twiml), media_type="application/xml")

# ── Health check ──────────────────────────────

@app.get("/")
async def health():
    return {
        "status":     "online",
        "chunks":     collection.count(),
        "model":      GEMMA_MODEL,
        "embed_model": EMBED_MODEL
    }


# ── Test endpoint (no WhatsApp needed) ────────

@app.get("/test")
async def test_query(q: str = "how to stop bleeding"):
    """Test the full pipeline without WhatsApp."""
    docs   = hybrid_search(q)
    prompt = assemble_prompt(q, docs)
    reply  = call_gemma(prompt)
    return {
        "query":   q,
        "sources": [d.metadata.get("source") for d in docs],
        "reply":   reply
    }
