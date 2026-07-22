"""
Arabic Dialect Identification.

Identifies dialect as one of: MSA, Gulf, Egyptian, Levantine.
Uses CAMeL-Lab's bert-base-arabic-camelbert-da-dialect model.
Falls back to a TF-IDF + Logistic Regression classifier on MADAR-style samples.
"""

import logging
import re
from typing import Tuple

import numpy as np

from config import DIALECT_MODEL, DIALECT_LABELS

logger = logging.getLogger(__name__)

_transformer_model = None
_tokenizer = None
_fallback_vectorizer = None
_fallback_classifier = None
_using_fallback = False


# ── Preprocessing ───────────────────────────────────────

def _preprocess(text: str) -> str:
    """Normalize Arabic text for dialect identification."""
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)  # tashkeel
    text = re.sub(r'[\u0622\u0623\u0625]', '\u0627', text)  # alef variants
    text = re.sub(r'\u0629', '\u0647', text)  # teh marbuta
    text = re.sub(r'\u0640', '', text)  # tatweel
    return text.strip()


# ── Transformer model ───────────────────────────────────

def _load_transformer_model():
    global _transformer_model, _tokenizer, _using_fallback

    if _transformer_model is not None or _using_fallback:
        return

    try:
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            pipeline,
        )

        logger.info(f"Loading dialect model: {DIALECT_MODEL}")
        _tokenizer = AutoTokenizer.from_pretrained(DIALECT_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(DIALECT_MODEL)

        _transformer_model = pipeline(
            "text-classification",
            model=model,
            tokenizer=_tokenizer,
            return_all_scores=True,
        )
        logger.info("Dialect model loaded successfully.")
    except Exception as e:
        logger.warning(
            f"Failed to load {DIALECT_MODEL}: {e}. "
            f"Using TF-IDF fallback."
        )
        _load_fallback_classifier()


# ── TF-IDF Fallback ─────────────────────────────────────

def _load_fallback_classifier():
    global _fallback_vectorizer, _fallback_classifier, _using_fallback
    _using_fallback = True

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    logger.info("Building TF-IDF fallback dialect classifier.")

    # MADAR-inspired dialect samples
    corpus = [
        # MSA (Modern Standard Arabic)
        ("اللغة العربية الفصحى هي اللغة الرسمية", "MSA"),
        ("تسعى الحكومة إلى تطوير التعليم", "MSA"),
        ("الاقتصاد العالمي يشهد تغيرات كبيرة", "MSA"),
        ("أشار التقرير إلى وجود تحسن ملحوظ", "MSA"),
        ("ينبغي على الجميع الالتزام بالقوانين", "MSA"),
        ("الأمم المتحدة تدعو إلى السلام", "MSA"),
        ("الدراسة تؤكد أهمية البحث العلمي", "MSA"),

        # Gulf (UAE/KSA/Qatar — using Gulf vocabulary)
        ("شو أخبارك يالغالي", "Gulf"),
        ("ما عندي مانع أروح وياهم", "Gulf"),
        ("ترى الموضوع سهل", "Gulf"),
        ("شكثر سعر هذا الشي", "Gulf"),
        ("إي والله الأمور طيبة", "Gulf"),
        ("وين رايح الحين", "Gulf"),
        ("خلنا نشوف شو يصير", "Gulf"),
        ("هذا شي زين", "Gulf"),
        ("بس خلاص تعبت من هالوضع", "Gulf"),

        # Egyptian
        ("إزيك عامل إيه النهاردة", "Egyptian"),
        ("عايز أروح السينما معاك", "Egyptian"),
        ("الموضوع ده ممل أوي", "Egyptian"),
        ("أنا مش فاهم حاجة", "Egyptian"),
        ("إيه ده بجد", "Egyptian"),
        ("أنا جاي لك بكرة إن شاء الله", "Egyptian"),
        ("دي حاجة جميلة قوي", "Egyptian"),
        ("كده كده هنشوف", "Egyptian"),

        # Levantine (Lebanon/Syria/Jordan/Palestine)
        ("شو بدك تاكل اليوم", "Levantine"),
        ("كيفك يا حبيبي", "Levantine"),
        ("بدي روح عالسوق", "Levantine"),
        ("شو هالحكي", "Levantine"),
        ("له يا زلمة", "Levantine"),
        ("وين رايح هلا", "Levantine"),
        ("عم دور على شغل", "Levantine"),
        ("خلصت الشغل اليوم", "Levantine"),
        ("ما بدي أحكي", "Levantine"),
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

    logger.info(
        "TF-IDF dialect fallback trained on %d examples.", len(texts)
    )


# ── Public API ──────────────────────────────────────────

def identify_dialect(text: str) -> Tuple[str, float, dict, str]:
    """
    Identify the dialect of Arabic text.

    Returns:
        (dialect_label, confidence, probabilities_dict, model_name)
    """
    text = _preprocess(text)

    if _using_fallback or _transformer_model is None:
        try:
            _load_transformer_model()
        except Exception:
            pass

    if not _using_fallback and _transformer_model is not None:
        try:
            result = _transformer_model(text)
            scores = {}
            for item in result[0]:
                label = item["label"].lower()
                scores[label] = round(item["score"], 4)

            # Map model output to our standard labels
            mapped = {}
            for key, name in DIALECT_LABELS.items():
                mapped[name] = scores.get(
                    name.lower(), scores.get(f"LABEL_{key}", 0.0)
                )

            top_label = max(mapped, key=mapped.get)
            confidence = mapped[top_label]

            return top_label, confidence, mapped, DIALECT_MODEL
        except Exception as e:
            logger.warning(f"Transformer dialect inference failed: {e}. Using fallback.")
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
    """Pre-download the dialect identification model."""
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        logger.info(f"Downloading {DIALECT_MODEL}...")
        AutoTokenizer.from_pretrained(DIALECT_MODEL)
        AutoModelForSequenceClassification.from_pretrained(DIALECT_MODEL)
        logger.info("Download complete.")
        return True
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        return False
