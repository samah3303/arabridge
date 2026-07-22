"""
Code-Switching Detection and Handling.

Detects Arabic-English code-switching in text, segments by language,
and optionally processes each segment with the appropriate model.
"""

import logging
import re
from typing import List, Dict, Optional

from config import ARABIC_CHAR_RANGE, ARABIC_CHAR_RANGE_EXT

logger = logging.getLogger(__name__)

# ── Unicode helpers ─────────────────────────────────────

def _is_arabic_char(ch: str) -> bool:
    """Check if a character falls within Arabic Unicode ranges."""
    cp = ord(ch)
    for start, end in ARABIC_CHAR_RANGE_EXT:
        if start <= cp <= end:
            return True
    return False


def _is_arabic_word(word: str) -> bool:
    """Check if a word is predominantly Arabic (based on character count)."""
    if not word:
        return False
    arabic_chars = sum(1 for ch in word if _is_arabic_char(ch))
    total_chars = len(word)
    # If >50% of characters are Arabic, consider it Arabic
    return arabic_chars / total_chars > 0.5 if total_chars > 0 else False


def _is_english_word(word: str) -> bool:
    """Check if a word is predominantly Latin script."""
    if not word:
        return False
    latin_chars = sum(1 for ch in word if ch.isascii() and ch.isalpha())
    total_alpha = sum(1 for ch in word if ch.isalpha())
    if total_alpha == 0:
        return False
    return latin_chars / total_alpha > 0.5


# ── Tokenization ────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """Split text into words/tokens, preserving punctuation."""
    # Split on whitespace while keeping tokens
    return re.findall(r'\S+', text)


# ── Segmentation ────────────────────────────────────────

def segment_text(text: str) -> List[Dict[str, str]]:
    """
    Segment text into continuous language blocks.

    Returns list of dicts with keys: text, language
    Language is one of: 'ar', 'en', 'mixed', 'other'
    """
    tokens = _tokenize(text)
    if not tokens:
        return []

    segments = []
    current_text = []
    current_lang = None

    for token in tokens:
        lang = _classify_token(token)

        if current_lang is None:
            current_lang = lang
            current_text.append(token)
        elif lang == current_lang or lang == 'other' or current_lang == 'other':
            # Same language or other: merge
            current_text.append(token)
            if lang != 'other' and current_lang == 'other':
                current_lang = lang
        else:
            # Language switch
            segments.append({
                'text': ' '.join(current_text),
                'language': current_lang,
            })
            current_text = [token]
            current_lang = lang

    # Add the last segment
    if current_text:
        segments.append({
            'text': ' '.join(current_text),
            'language': current_lang,
        })

    return segments


def _classify_token(token: str) -> str:
    """Classify a single token as 'ar', 'en', or 'other'."""
    # Remove punctuation for language detection
    stripped = token.strip('.,!?;:()[]{}"\'-«»،؟؛')
    if not stripped:
        return 'other'

    # Check for numbers/punctuation-only
    if stripped.isdigit():
        return 'other'
    if not any(ch.isalpha() for ch in stripped):
        return 'other'

    if _is_arabic_word(stripped):
        return 'ar'
    if _is_english_word(stripped):
        return 'en'

    # Mixed word (e.g. arabic+english chars or symbols)
    return 'other'


# ── Code-switching detection ────────────────────────────

def detect_code_switching(text: str) -> Dict:
    """
    Detect whether text contains code-switching.

    Returns dict with:
        - has_code_switching: bool
        - primary_language: str
        - segments: list of {text, language}
        - language_ratio: {ar: float, en: float, other: float}
    """
    segments = segment_text(text)

    langs_found = set(seg['language'] for seg in segments)
    has_switching = len(langs_found - {'other'}) > 1

    # Determine primary language by word count
    ar_words = 0
    en_words = 0
    other_words = 0
    for seg in segments:
        word_count = len(seg['text'].split())
        if seg['language'] == 'ar':
            ar_words += word_count
        elif seg['language'] == 'en':
            en_words += word_count
        else:
            other_words += word_count

    total = ar_words + en_words + other_words

    if total == 0:
        return {
            'has_code_switching': False,
            'primary_language': 'unknown',
            'segments': [],
            'language_ratio': {'ar': 0.0, 'en': 0.0, 'other': 0.0},
        }

    # Pick primary
    if ar_words >= en_words and ar_words >= other_words:
        primary = 'ar'
    elif en_words >= ar_words and en_words >= other_words:
        primary = 'en'
    else:
        primary = 'other'

    return {
        'has_code_switching': has_switching,
        'primary_language': primary,
        'segments': segments,
        'language_ratio': {
            'ar': round(ar_words / total, 3) if total > 0 else 0.0,
            'en': round(en_words / total, 3) if total > 0 else 0.0,
            'other': round(other_words / total, 3) if total > 0 else 0.0,
        },
    }


# ── Process code-switched text ──────────────────────────

def process_code_switched(text: str) -> Dict:
    """
    Full code-switching analysis: detect, segment, and classify.

    Returns enriched result with optional sentiment per segment.
    """
    result = detect_code_switching(text)

    # Add segment-level sentiment (attempt, but don't fail)
    enriched_segments = []
    for seg in result['segments']:
        seg_info = {
            'text': seg['text'],
            'language': seg['language'],
            'sentiment': None,
        }

        if seg['language'] == 'ar' and seg['text'].strip():
            try:
                from arabic_sentiment import analyze_arabic_sentiment
                label, _, _, _ = analyze_arabic_sentiment(seg['text'])
                seg_info['sentiment'] = label
            except Exception as e:
                logger.debug(f"Arabic sentiment for segment failed: {e}")

        elif seg['language'] == 'en' and seg['text'].strip():
            try:
                from english_sentiment import analyze_english_sentiment
                label, _, _, _ = analyze_english_sentiment(seg['text'])
                seg_info['sentiment'] = label
            except Exception as e:
                logger.debug(f"English sentiment for segment failed: {e}")

        enriched_segments.append(seg_info)

    result['segments'] = enriched_segments
    return result
