#!/usr/bin/env python3
"""
Pre-download all Hugging Face models required by AraBridge.

Run this before first use to cache models locally:
    python download_models.py
"""

import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("download_models")


def download_arabic_sentiment():
    """Download CAMeL-Lab Arabic sentiment model."""
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from config import ARABIC_SENTIMENT_MODEL

        logger.info(f"⬇️  Downloading: {ARABIC_SENTIMENT_MODEL}")
        AutoTokenizer.from_pretrained(ARABIC_SENTIMENT_MODEL)
        AutoModelForSequenceClassification.from_pretrained(ARABIC_SENTIMENT_MODEL)
        logger.info(f"✅ Done: {ARABIC_SENTIMENT_MODEL}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        return False


def download_english_sentiment():
    """Download English sentiment model."""
    try:
        from transformers import pipeline
        from config import ENGLISH_SENTIMENT_MODEL

        logger.info(f"⬇️  Downloading: {ENGLISH_SENTIMENT_MODEL}")
        pipeline("text-classification", model=ENGLISH_SENTIMENT_MODEL)
        logger.info(f"✅ Done: {ENGLISH_SENTIMENT_MODEL}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        return False


def download_dialect():
    """Download CAMeL-Lab dialect identification model."""
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from config import DIALECT_MODEL

        logger.info(f"⬇️  Downloading: {DIALECT_MODEL}")
        AutoTokenizer.from_pretrained(DIALECT_MODEL)
        AutoModelForSequenceClassification.from_pretrained(DIALECT_MODEL)
        logger.info(f"✅ Done: {DIALECT_MODEL}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        return False


def main():
    print("=" * 60)
    print("  AraBridge — Model Downloader")
    print("  Downloads all Hugging Face models (~1GB)")
    print("=" * 60)
    print()

    # Add backend to path to import config
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

    results = {}

    logger.info("Starting downloads... This may take 10-20 minutes on first run.\n")

    results["Arabic Sentiment"] = download_arabic_sentiment()
    print()
    results["English Sentiment"] = download_english_sentiment()
    print()
    results["Dialect ID"] = download_dialect()
    print()

    print("=" * 60)
    print("  DOWNLOAD SUMMARY")
    print("=" * 60)
    for model, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {model}")
    print()

    if all(results.values()):
        print("🎉 All models downloaded successfully!")
        print("   Ready to start: cd backend && uvicorn main:app")
    else:
        print("⚠️  Some models failed to download.")
        print("   The system will use TF-IDF fallbacks where available.")
        print("   You can retry: python download_models.py")

    print()


if __name__ == "__main__":
    main()
