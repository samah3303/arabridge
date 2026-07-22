"""
AraBridge API — FastAPI Application
Arabic-English Bilingual NLP Toolkit
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import DATABASE_URL
from models import (
    SentimentRequest,
    SentimentResponse,
    DialectRequest,
    DialectResponse,
    CodeSwitchRequest,
    CodeSwitchResponse,
    CodeSwitchSegment,
    ChatRequest,
    ChatResponse,
    BenchmarkEntry,
    BenchmarkResponse,
)
from arabic_sentiment import analyze_arabic_sentiment
from english_sentiment import analyze_english_sentiment
from dialect_id import identify_dialect
from code_switching import process_code_switched
from chatbot import chat
from database import init_db, save_benchmark, get_benchmarks

# ── Logging ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("arabridge")


# ── Lifespan (startup/shutdown) ─────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB. Shutdown: cleanup."""
    logger.info("Starting AraBridge API...")
    if DATABASE_URL:
        init_db()
    yield
    logger.info("Shutting down AraBridge API.")


# ── FastAPI app ─────────────────────────────────────────

app = FastAPI(
    title="AraBridge API",
    description="Arabic-English Bilingual NLP Toolkit — Sentiment Analysis, "
                "Dialect Identification, Code-Switching Detection, "
                "and Bilingual Chatbot.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS — allow Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health Check ────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "arabridge"}


# ── Sentiment Analysis ──────────────────────────────────

@app.post("/sentiment", response_model=SentimentResponse)
async def analyze_sentiment(req: SentimentRequest):
    """Analyze sentiment for Arabic or English text."""
    start = time.perf_counter()

    if req.language == "ar":
        label, confidence, probs, model = analyze_arabic_sentiment(req.text)
    else:
        label, confidence, probs, model = analyze_english_sentiment(req.text)

    latency_ms = (time.perf_counter() - start) * 1000

    return SentimentResponse(
        text=req.text,
        language=req.language,
        sentiment=label,
        confidence=confidence,
        probabilities=probs,
        model_used=model,
    )


# ── Dialect Identification ──────────────────────────────

@app.post("/dialect", response_model=DialectResponse)
async def detect_dialect(req: DialectRequest):
    """Identify Arabic dialect: MSA, Gulf, Egyptian, or Levantine."""
    try:
        label, confidence, probs, model = identify_dialect(req.text)

        return DialectResponse(
            text=req.text,
            predicted_dialect=label,
            confidence=confidence,
            dialect_probabilities=probs,
            model_used=model,
        )
    except Exception as e:
        logger.error(f"Dialect detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Code-Switching Detection ────────────────────────────

@app.post("/code-switching", response_model=CodeSwitchResponse)
async def detect_code_switching(req: CodeSwitchRequest):
    """Detect and analyze code-switched Arabic-English text."""
    try:
        result = process_code_switched(req.text)

        segments = [
            CodeSwitchSegment(**seg) for seg in result["segments"]
        ]

        return CodeSwitchResponse(
            original_text=req.text,
            segments=segments,
            primary_language=result["primary_language"],
            has_code_switching=result["has_code_switching"],
        )
    except Exception as e:
        logger.error(f"Code-switching error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Chatbot ─────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chatbot_chat(req: ChatRequest):
    """Bilingual chatbot powered by DeepSeek."""
    try:
        messages = [msg.model_dump() for msg in req.messages]
        reply = await chat(
            messages,
            system_prompt_override=req.system_prompt_override,
        )
        return ChatResponse(reply=reply, model="deepseek-chat")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Benchmarks ──────────────────────────────────────────

@app.get("/benchmarks", response_model=BenchmarkResponse)
async def list_benchmarks(task: str | None = None):
    """Retrieve stored benchmark results."""
    entries = get_benchmarks(task)
    return BenchmarkResponse(
        entries=entries,
        count=len(entries),
    )


@app.post("/benchmarks", response_model=dict)
async def run_benchmark():
    """
    Run a full cross-lingual benchmark comparing Arabic vs English sentiment models.
    Results are saved to the database.
    """
    test_samples = {
        "ar_positive": [
            "هذا المنتج رائع وأنا سعيد جدا به",
            "الخدمة كانت ممتازة وسريعة",
            "الفيلم جميل وممتع للغاية",
            "أحب هذا المطعم كثيرا، الأكل لذيذ",
            "التجربة كانت مذهلة بكل المقاييس",
        ],
        "ar_negative": [
            "الخدمة سيئة جدا ولا أنصح بها",
            "المنتج لا يعمل بشكل جيد",
            "التجربة كانت مخيبة للآمال",
            "الجودة منخفضة والسعر مرتفع",
            "لم يعجبني هذا الفيلم أبدا",
        ],
        "ar_neutral": [
            "وصلت الطلبية اليوم",
            "الطقس معتدل هذا الأسبوع",
            "الاجتماع سيكون في المساء",
            "المتجر يفتح الساعة التاسعة",
            "الكتاب متوفر في المكتبة",
        ],
        "en_positive": [
            "This product is amazing and I love it",
            "The service was excellent and fast",
            "Great movie, very entertaining",
            "I really enjoyed this restaurant",
            "Absolutely wonderful experience",
        ],
        "en_negative": [
            "This is terrible, do not recommend",
            "The product does not work at all",
            "Very disappointing experience",
            "Poor quality for the price",
            "I hated this movie so much",
        ],
        "en_neutral": [
            "The package arrived today",
            "The weather is mild this week",
            "The meeting is at 3pm",
            "The store opens at 9am",
            "The book is available at the library",
        ],
    }

    # Ground truth labels
    ground_truth = {
        "ar_positive": "positive",
        "ar_negative": "negative",
        "ar_neutral": "neutral",
        "en_positive": "positive",
        "en_negative": "negative",
        "en_neutral": "neutral",
    }

    results = []

    for key, samples in test_samples.items():
        lang = "ar" if key.startswith("ar") else "en"
        expected = ground_truth[key]

        correct = 0
        total = len(samples)
        total_latency = 0.0

        for sample in samples:
            start = time.perf_counter()

            if lang == "ar":
                predicted, _, _, model = analyze_arabic_sentiment(sample)
            else:
                predicted, _, _, model = analyze_english_sentiment(sample)

            latency = (time.perf_counter() - start) * 1000
            total_latency += latency

            if predicted == expected:
                correct += 1

        accuracy = correct / total if total > 0 else 0.0
        avg_latency = total_latency / total if total > 0 else 0.0

        # F1 simplifies to accuracy for single-class evaluation
        f1 = accuracy

        # Save to DB
        save_benchmark(
            model_name=model,
            task=f"sentiment_{key}",
            accuracy=round(accuracy, 4),
            f1_score=round(f1, 4),
            latency_ms=round(avg_latency, 1),
        )

        results.append({
            "task": f"sentiment_{key}",
            "language": lang,
            "model": model,
            "accuracy": round(accuracy, 4),
            "f1_score": round(f1, 4),
            "avg_latency_ms": round(avg_latency, 1),
            "samples": total,
        })

    return {
        "results": results,
        "total_tests": len(results),
    }


# ── Run with: uvicorn main:app ──────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
