# 🌉 AraBridge — Arabic-English Bilingual NLP Toolkit

> **Bridging Arabic and English NLP** — Built for the UAE AI ecosystem  
> *422M Arabic speakers. One of the world's most underserved NLP markets. This is your edge.*

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B.svg)](https://streamlit.io)
[![Hugging Face](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Live Demo](https://img.shields.io/badge/Live-Demo-FF4B4B?logo=streamlit)](https://arabridge-nkswkmwdy8ggm8ahmjubrr.streamlit.app)

## Why AraBridge?

Arabic NLP is **the #1 differentiator** for AI engineering roles in the UAE and GCC. Companies like **G42, TII (Falcon team), and Mubadala** are investing billions in Arabic AI — and engineers who can work at the Arabic-English intersection command **20-40% salary premiums**.

AraBridge demonstrates the full stack of Arabic NLP capabilities that UAE employers look for:

- ✅ Arabic sentiment analysis with CAMeL-Lab models
- ✅ Dialect identification (Gulf, Egyptian, Levantine, MSA)
- ✅ Code-switching detection & processing
- ✅ Bilingual chatbot (Arabic + English)
- ✅ Cross-lingual benchmark comparison
- ✅ QLoRA fine-tuning pipeline for Colab

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                        │
│   Sentiment │ Dialect ID │ Code-Switching │ Chat │ Bench    │
└──────────────────────┬──────────────────────────────────────┘
                       │ REST API (httpx)
┌──────────────────────▼──────────────────────────────────────┐
│                    FastAPI Backend                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────┐  │
│  │ Arabic   │ │ English  │ │ Dialect  │ │ Code-Switching│  │
│  │ Sentiment│ │ Sentiment│ │ ID       │ │ Detector      │  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └───────┬───────┘  │
│       │            │            │               │           │
│  ┌────▼────────────▼────────────▼───────────────▼───────┐   │
│  │              Hugging Face Transformers                │   │
│  │   CAMeL-Lab · AraBERT · RoBERTa · TF-IDF Fallbacks   │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────┐  ┌────────────────────────────┐   │
│  │   DeepSeek Chatbot   │  │   Neon PostgreSQL (bench)  │   │
│  └──────────────────────┘  └────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Features

### 🎭 Sentiment Analysis
- **Arabic**: CAMeL-Lab `bert-base-arabic-camelbert-da-sentiment` — 4-class (positive, negative, neutral, mixed)
- **English**: `cardiffnlp/twitter-roberta-base-sentiment-latest` — 3-class baseline
- Graceful fallback to TF-IDF + Logistic Regression if models fail
- Side-by-side comparison in the UI

### 🗣️ Dialect Identification
- 4 dialects: **MSA**, **Gulf** (UAE/KSA/Qatar), **Egyptian**, **Levantine**
- CAMeL-Lab dialect model with TF-IDF fallback
- Confidence scores and probability distributions

### 🔄 Code-Switching Detection
- Word-level language detection (Arabic Unicode range + Latin script)
- Text segmentation into Arabic/English blocks
- Per-segment sentiment analysis
- Statistics on language mixing ratios

### 💬 Bilingual Chatbot
- Powered by **DeepSeek** (`deepseek-chat`)
- Responds in the same language(s) you use
- Perfect for code-switched input
- RTL-optimized chat interface with Arabic diacritics

### 📊 Benchmark Dashboard
- Cross-lingual accuracy & F1 comparison
- Latency benchmarking
- PostgreSQL persistence via Neon
- Interactive Plotly charts

### 🧪 Fine-Tuning Notebook
- QLoRA fine-tuning for Arabic dialect classification
- Runs on free Google Colab with T4 GPU
- 4-bit quantization: train 110M+ params with <8GB VRAM

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | FastAPI, Uvicorn |
| **ML Models** | Hugging Face Transformers, CAMeL-Lab, AraBERT, RoBERTa |
| **LLM** | DeepSeek API (`deepseek-chat`) |
| **Fine-tuning** | QLoRA, PEFT, bitsandbytes |
| **Database** | Neon (Serverless PostgreSQL) |
| **Frontend** | Streamlit, Plotly |
| **NLP** | scikit-learn (TF-IDF fallbacks), NumPy |

## Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/yourusername/arabridge.git
cd arabridge

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys:
#   DEEPSEEK_API_KEY=sk-...    (from https://platform.deepseek.com)
#   DATABASE_URL=postgresql://.. (from https://neon.tech) — optional
```

### 3. Download Models (~1GB, one-time)

```bash
python download_models.py
```

### 4. Start Backend

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API docs at: http://localhost:8000/docs

### 5. Start Frontend

```bash
cd frontend
streamlit run app.py
```

Open: http://localhost:8501

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/sentiment` | Sentiment analysis (ar/en) |
| `POST` | `/dialect` | Dialect identification |
| `POST` | `/code-switching` | Code-switching detection |
| `POST` | `/chat` | Bilingual chatbot |
| `GET` | `/benchmarks` | List saved benchmarks |
| `POST` | `/benchmarks` | Run benchmark suite |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DEEPSEEK_API_KEY` | Yes (for chatbot) | DeepSeek API key |
| `DATABASE_URL` | No | Neon PostgreSQL URL |
| `ARABRIDGE_API_URL` | No | Backend URL (default: localhost:8000) |
| `HF_HOME` | No | Hugging Face cache location |

## Project Structure

```
arabridge/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration & constants
│   ├── arabic_sentiment.py  # Arabic sentiment (CAMeL-Lab + fallback)
│   ├── english_sentiment.py # English sentiment baseline
│   ├── dialect_id.py        # Dialect identification
│   ├── code_switching.py    # Code-switching detection
│   ├── chatbot.py           # DeepSeek bilingual chatbot
│   ├── database.py          # Neon PostgreSQL operations
│   └── models.py            # Pydantic schemas
├── frontend/
│   ├── app.py               # Streamlit UI (5 tabs, RTL)
│   └── requirements.txt
├── notebooks/
│   └── fine_tuning.ipynb    # QLoRA fine-tuning (Colab)
├── download_models.py       # Pre-download all HF models
├── .env.example
├── .gitignore
└── README.md
```

## Why UAE Hiring Managers Care

The UAE is investing heavily in Arabic AI:

- **G42** — $10B+ AI conglomerate, building Arabic LLMs
- **TII (Falcon)** — UAE's sovereign AI lab, Falcon model family
- **Abu Dhabi Investment Authority** — funding Arabic NLP research
- **Dubai AI Campus** — dedicated AI free zone

Arabic NLP engineers who can demonstrate production-ready skills are **rare and highly valued**. AraBridge shows you can:

1. Work with Arabic-specific transformer models (CAMeL-Lab, AraBERT)
2. Handle the messy real-world problem of code-switching
3. Build bilingual interfaces with proper RTL support
4. Fine-tune models efficiently with modern techniques (QLoRA)
5. Compare and evaluate across languages

## Future Roadmap

- [ ] Expand dialect coverage (Maghrebi, Iraqi, Sudanese)
- [ ] Arabic text generation with DeepSeek
- [ ] Arabic NER (Named Entity Recognition)
- [ ] Arabic-English machine translation evaluation
- [ ] Integration with UAE-specific datasets
- [ ] Docker Compose deployment
- [ ] CI/CD pipeline

## License

MIT © 2024 AraBridge

---

<p align="center">
  <b>🌉 Built for the UAE AI Ecosystem</b><br>
  <i>G42 · TII · Falcon Team · Arabic NLP</i>
</p>
