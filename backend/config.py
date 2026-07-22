"""
AraBridge Configuration
Loads environment variables and provides project-wide settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

# ── DeepSeek API ─────────────────────────────────────────
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

# ── Neon PostgreSQL ─────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "")

# ── Hugging Face model names ────────────────────────────
ARABIC_SENTIMENT_MODEL = "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"
ARABIC_SENTIMENT_FALLBACK = "aubmindlab/bert-base-arabertv2"
ENGLISH_SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
DIALECT_MODEL = "CAMeL-Lab/bert-base-arabic-camelbert-da-dialect"

# ── Model cache ─────────────────────────────────────────
MODEL_CACHE_DIR = os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface"))

# ── Dialect label maps for CamelBERT dialect model ──────
DIALECT_LABELS = {
    0: "MSA",
    1: "Egyptian",
    2: "Gulf",
    3: "Levantine",
}

# ── Sentiment label maps ────────────────────────────────
ARABIC_SENTIMENT_LABELS = {
    0: "negative",
    1: "positive",
    2: "neutral",
    3: "mixed",
}

ENGLISH_SENTIMENT_LABELS = {
    0: "negative",
    1: "neutral",
    2: "positive",
}

# ── Arabic Unicode range for code-switching ─────────────
ARABIC_CHAR_RANGE = (0x0600, 0x06FF)
# Extended: Arabic Presentation Forms-A, Arabic Presentation Forms-B
ARABIC_CHAR_RANGE_EXT = [(0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)]
