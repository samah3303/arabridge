"""
Arabic Sentiment Analysis using CAMeL-Lab's bert-base-arabic-camelbert-da-sentiment.
Falls back to TF-IDF + Logistic Regression if the model fails to load.
"""

import logging
import re
import numpy as np
from typing import Tuple

from config import (
    ARABIC_SENTIMENT_MODEL,
    ARABIC_SENTIMENT_FALLBACK,
    ARABIC_SENTIMENT_LABELS,
)

logger = logging.getLogger(__name__)

# ── Global cached model ─────────────────────────────────

_transformer_model = None
_tokenizer = None
_fallback_vectorizer = None
_fallback_classifier = None
_using_fallback = False


# ── Arabic text preprocessing ───────────────────────────

def _preprocess_arabic(text: str) -> str:
    """Normalize Arabic text: remove tashkeel, normalize alef variants."""
    # Remove diacritics (tashkeel)
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    # Normalize alef variants to bare alef
    text = re.sub(r'[\u0622\u0623\u0625]', '\u0627', text)
    # Normalize teh marbuta to heh
    text = re.sub(r'\u0629', '\u0647', text)
    # Remove tatweel (kashida)
    text = re.sub(r'\u0640', '', text)
    return text.strip()


# ── Transformer model (CAMeL-Lab) ───────────────────────

def _load_transformer_model():
    """Lazy-load the CAMeL-Lab sentiment model."""
    global _transformer_model, _tokenizer, _using_fallback

    if _transformer_model is not None or _using_fallback:
        return

    try:
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            pipeline,
        )

        logger.info(f"Loading Arabic sentiment model: {ARABIC_SENTIMENT_MODEL}")
        _tokenizer = AutoTokenizer.from_pretrained(ARABIC_SENTIMENT_MODEL)
        _transformer_model = AutoModelForSequenceClassification.from_pretrained(
            ARABIC_SENTIMENT_MODEL
        )

        # Build a pipeline for easy inference
        _transformer_model = pipeline(
            "text-classification",
            model=_transformer_model,
            tokenizer=_tokenizer,
            return_all_scores=True,
        )
        logger.info("Arabic sentiment model loaded successfully.")
    except Exception as e:
        logger.warning(
            f"Failed to load {ARABIC_SENTIMENT_MODEL}: {e}. "
            f"Falling back to TF-IDF classifier."
        )
        _load_fallback_classifier()


# ── TF-IDF Fallback ─────────────────────────────────────

def _load_fallback_classifier():
    """Build a lightweight TF-IDF + Logistic Regression classifier."""
    global _fallback_vectorizer, _fallback_classifier, _using_fallback
    _using_fallback = True

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    logger.info("Building TF-IDF fallback sentiment classifier.")

    # Small training corpus of labeled Arabic sentiment examples
    corpus = [
        # Positive
        ("الفيلم كان رائعا وممتعا جدا", "positive"),
        ("الخدمة ممتازة وسريعة", "positive"),
        ("أحب هذا المنتج كثيرا", "positive"),
        ("التجربة كانت جميلة ومفيدة", "positive"),
        ("الأكل لذيذ والمطعم نظيف", "positive"),
        ("الجو رائع والمنظر جميل", "positive"),
        ("شكرا لكم على المجهود الرائع", "positive"),
        ("عمل ممتاز وفريق محترف", "positive"),
        ("سعيد جدا بهذه النتيجة", "positive"),
        ("الكتاب مفيد وشيق", "positive"),
        # Negative
        ("الخدمة سيئة جدا", "negative"),
        ("لم يعجبني الفيلم أبدا", "negative"),
        ("المنتج لا يستحق السعر", "negative"),
        ("التأخير متكرر والخدمة بطيئة", "negative"),
        ("الأكل بارد وطعمه سيئ", "negative"),
        ("تجربة مخيبة للآمال", "negative"),
        ("الموظف غير محترم", "negative"),
        ("لا أنصح بهذا المطعم", "negative"),
        ("الجودة منخفضة والسعر مرتفع", "negative"),
        ("كان يوما سيئا", "negative"),
        # Neutral
        ("وصلت الطلب اليوم", "neutral"),
        ("الاجتماع في المساء", "neutral"),
        ("الطقس معتدل اليوم", "neutral"),
        ("المحل يفتح الساعة التاسعة", "neutral"),
        ("الكتاب يتكون من ثلاث فصول", "neutral"),
        ("ذهبت إلى السوق صباحا", "neutral"),
        ("الشارع مزدحم كالعادة", "neutral"),
        ("الفيلم مدته ساعتان", "neutral"),
        # Mixed
        ("الفيلم جيد لكن النهاية محبطة", "mixed"),
        ("الخدمة سريعة لكن السعر مرتفع", "mixed"),
        ("الأكل لذيذ لكن المكان مزدحم", "mixed"),
        ("المنتج مفيد لكن التصميم قديم", "mixed"),
    ]

    texts = [item[0] for item in corpus]
    labels = [item[1] for item in corpus]

    _fallback_vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=(2, 5),
        max_features=5000,
    )
    X = _fallback_vectorizer.fit_transform(texts)

    _fallback_classifier = LogisticRegression(
        multi_class='multinomial',
        max_iter=500,
        random_state=42,
    )
    _fallback_classifier.fit(X, labels)

    logger.info("TF-IDF fallback classifier trained on %d examples.", len(texts))


# ── Public API ──────────────────────────────────────────

def analyze_arabic_sentiment(text: str) -> Tuple[str, float, dict, str]:
    """
    Analyze sentiment of Arabic text.

    Returns:
        (sentiment_label, confidence, probabilities_dict, model_name)
    """
    text = _preprocess_arabic(text)

    if _using_fallback or _transformer_model is None:
        try:
            _load_transformer_model()
        except Exception:
            pass

    if not _using_fallback and _transformer_model is not None:
        try:
            result = _transformer_model(text)
            # result is a list of [{label: ..., score: ...}, ...] for each input
            scores = {item["label"].lower(): item["score"] for item in result[0]}

            # Map model output to our labels
            mapped = {}
            for label_key, label_name in ARABIC_SENTIMENT_LABELS.items():
                # CamelBERT outputs LABEL_0, LABEL_1, etc.
                model_key = f"LABEL_{label_key}"
                mapped[label_name] = scores.get(model_key, scores.get(label_name, 0.0))

            top_label = max(mapped, key=mapped.get)
            confidence = mapped[top_label]

            return top_label, round(confidence, 4), mapped, ARABIC_SENTIMENT_MODEL
        except Exception as e:
            logger.warning(f"Transformer inference failed: {e}. Using fallback.")
            _using_fallback = True

    # Fallback path
    if _fallback_classifier is None:
        _load_fallback_classifier()

    X = _fallback_vectorizer.transform([text])
    probs = _fallback_classifier.predict_proba(X)[0]
    classes = _fallback_classifier.classes_

    prob_dict = {cls: round(float(prob), 4) for cls, prob in zip(classes, probs)}
    top_idx = np.argmax(probs)
    top_label = classes[top_idx]
    confidence = round(float(probs[top_idx]), 4)

    return top_label, confidence, prob_dict, "TF-IDF + Logistic Regression (fallback)"


# ── Pre-download helper ─────────────────────────────────

def download_model():
    """Pre-download the Arabic sentiment model."""
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        logger.info(f"Downloading {ARABIC_SENTIMENT_MODEL}...")
        AutoTokenizer.from_pretrained(ARABIC_SENTIMENT_MODEL)
        AutoModelForSequenceClassification.from_pretrained(ARABIC_SENTIMENT_MODEL)
        logger.info("Download complete.")
        return True
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        return False
