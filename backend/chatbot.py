"""
Bilingual Chatbot using DeepSeek API.

Handles Arabic, English, and code-switched input naturally.
Responds in the language(s) the user uses.
"""

import logging
from typing import List, Dict, Optional

import httpx

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

logger = logging.getLogger(__name__)

# ── System prompt ───────────────────────────────────────

BILINGUAL_SYSTEM_PROMPT = """You are a bilingual (Arabic-English) assistant for AraBridge, an Arabic NLP toolkit.
Your role is to help users with natural conversation in any language they prefer.

Rules:
1. Respond in the SAME language(s) the user uses. If they write in Arabic, respond in Arabic. If in English, respond in English. If they mix Arabic and English (code-switching), respond in a similar mixed form.
2. When responding in Arabic, use proper Arabic script with diacritics (tashkeel) where appropriate for clarity.
3. Be helpful, concise, and culturally aware. Understand Arabic dialects (Gulf, Egyptian, Levantine, MSA) and respond naturally.
4. If asked about technical topics, you can switch between Arabic and English as needed — Arabic for general explanation, English for technical terms.
5. Always maintain proper RTL formatting for Arabic text in your responses.
6. You are powered by DeepSeek and part of the AraBridge toolkit for Arabic-English NLP.

أَنْتَ مُسَاعِدٌ ثُنَائِيُّ اللُّغَةِ (العربية-الإنجليزية) لِمِنَصَّةِ AraBridge.
دَوْرُكَ هُوَ مُسَاعَدَةُ الْمُسْتَخْدِمِينَ فِي الْمُحَادَثَاتِ الطَّبِيعِيَّةِ بِأَيِّ لُغَةٍ يُفَضِّلُونَهَا."""


# ── Chat implementation ─────────────────────────────────

async def chat(
    messages: List[Dict[str, str]],
    system_prompt_override: Optional[str] = None,
) -> str:
    """
    Send conversation to DeepSeek and return the assistant's reply.

    Args:
        messages: List of {role, content} dicts.
        system_prompt_override: Optional custom system prompt.

    Returns:
        Assistant's reply string.

    Raises:
        ValueError: If DEEPSEEK_API_KEY is not set.
        httpx.HTTPError: On API communication failure.
    """
    if not DEEPSEEK_API_KEY:
        raise ValueError(
            "DEEPSEEK_API_KEY is not set. "
            "Please set it in your .env file or environment variables."
        )

    system_prompt = system_prompt_override or BILINGUAL_SYSTEM_PROMPT

    # Build the message list with system prompt at the top
    payload_messages = [
        {"role": "system", "content": system_prompt},
    ]

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role in ("user", "assistant", "system"):
            payload_messages.append({"role": role, "content": content})

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": payload_messages,
        "temperature": 0.7,
        "max_tokens": 1024,
        "top_p": 0.95,
    }

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    logger.info(f"Sending chat request to DeepSeek ({len(payload_messages)} messages)")

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            DEEPSEEK_BASE_URL,
            json=payload,
            headers=headers,
        )

        if response.status_code != 200:
            logger.error(f"DeepSeek API error: {response.status_code} - {response.text}")
            raise httpx.HTTPStatusError(
                f"DeepSeek API returned {response.status_code}: {response.text}",
                request=response.request,
                response=response,
            )

        data = response.json()

        try:
            reply = data["choices"][0]["message"]["content"]
            return reply
        except (KeyError, IndexError) as e:
            logger.error(f"Unexpected DeepSeek response format: {data}")
            raise ValueError(f"Could not parse DeepSeek response: {e}")


# ── Synchronous wrapper (for Streamlit) ─────────────────

def chat_sync(
    messages: List[Dict[str, str]],
    system_prompt_override: Optional[str] = None,
) -> str:
    """
    Synchronous version of chat() for use in Streamlit.
    """
    import asyncio

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're in an async context (unlikely with Streamlit)
            import nest_asyncio
            nest_asyncio.apply()
    except RuntimeError:
        pass

    return asyncio.run(chat(messages, system_prompt_override))
