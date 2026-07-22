"""
Pydantic schemas for AraBridge API requests and responses.
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


# ── Sentiment Analysis ──────────────────────────────────

class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000,
                      description="Input text for sentiment analysis")
    language: str = Field(default="ar", pattern="^(ar|en)$",
                          description="Language: 'ar' or 'en'")


class SentimentResponse(BaseModel):
    text: str
    language: str
    sentiment: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: dict = Field(default_factory=dict)
    model_used: str


# ── Dialect Identification ──────────────────────────────

class DialectRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=2000,
                      description="Arabic text for dialect identification")


class DialectResponse(BaseModel):
    text: str
    predicted_dialect: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    dialect_probabilities: dict = Field(default_factory=dict)
    model_used: str


# ── Code-Switching ──────────────────────────────────────

class CodeSwitchSegment(BaseModel):
    text: str
    language: str  # "ar", "en", "other"
    sentiment: Optional[str] = None


class CodeSwitchRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=3000,
                      description="Text that may contain code-switching")


class CodeSwitchResponse(BaseModel):
    original_text: str
    segments: List[CodeSwitchSegment]
    primary_language: str
    has_code_switching: bool


# ── Chatbot ─────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(...,
                                        description="Conversation history")
    system_prompt_override: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    model: str


# ── Benchmark ───────────────────────────────────────────

class BenchmarkEntry(BaseModel):
    id: Optional[int] = None
    model_name: str
    task: str
    accuracy: float
    f1_score: float
    latency_ms: float
    timestamp: Optional[str] = None


class BenchmarkResponse(BaseModel):
    entries: List[BenchmarkEntry]
    count: int
