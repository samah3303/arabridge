"""
English Sentiment Analysis using cardiffnlp/twitter-roberta-base-sentiment-latest.
Serves as the English baseline for cross-lingual comparison.
"""

import logging
from typing import Tuple

from config import ENGLISH_SENTIMENT_MODEL, ENGLISH_SENTIMENT_LABELS

logger = logging.getLogger(__name__)

_pipeline = None


def _load_model():
    """Lazy-load the English sentiment model."""
    global _pipeline

    if _pipeline is not None:
        return

    try:
        from transformers import pipeline

        logger.info(f"Loading English sentiment model: {ENGLISH_SENTIMENT_MODEL}")
        _pipeline = pipeline(
            "text-classification",
            model=ENGLISH_SENTIMENT_MODEL,
            return_all_scores=True,
        )
        logger.info("English sentiment model loaded.")
    except Exception as e:
        logger.error(f"Failed to load English sentiment model: {e}")
        raise


def analyze_english_sentiment(text: str) -> Tuple[str, float, dict, str]:
    """
    Analyze sentiment of English text.

    Returns:
        (sentiment_label, confidence, probabilities_dict, model_name)
    """
    _load_model()

    result = _pipeline(text)
    # result[0] is list of {label, score} dicts
    scores = {}
    for item in result[0]:
        label = item["label"].lower()
        scores[label] = round(item["score"], 4)

    top_label = max(scores, key=scores.get)
    confidence = scores[top_label]

    return top_label, confidence, scores, ENGLISH_SENTIMENT_MODEL


def download_model():
    """Pre-download the English sentiment model."""
    try:
        from transformers import pipeline

        logger.info(f"Downloading {ENGLISH_SENTIMENT_MODEL}...")
        pipeline(
            "text-classification",
            model=ENGLISH_SENTIMENT_MODEL,
        )
        logger.info("Download complete.")
        return True
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        return False
