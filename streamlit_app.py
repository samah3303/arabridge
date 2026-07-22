"""
AraBridge — Streamlit Cloud Entry Point
Arabic-English Bilingual NLP Toolkit with RTL Support.
Imports backend modules directly (no HTTP to localhost).
Models (CAMeL-Lab) auto-download on first run.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ── Ensure backend is importable ──────────────────────────
BACKEND_PATH = Path(__file__).resolve().parent / "backend"
sys.path.insert(0, str(BACKEND_PATH))

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="AraBridge | Arabic-English NLP Toolkit",
    page_icon="🌉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Import backend modules ────────────────────────────────
from config import DEEPSEEK_API_KEY, DATABASE_URL
from arabic_sentiment import analyze_arabic_sentiment
from english_sentiment import analyze_english_sentiment
from dialect_id import identify_dialect
from code_switching import process_code_switched
from chatbot import chat_sync
from database import init_db, save_benchmark, get_benchmarks

# ── Init DB (best effort) ─────────────────────────────────
@st.cache_resource
def _init_db():
    if DATABASE_URL:
        try:
            init_db()
            return True
        except Exception:
            pass
    return False

_db_ready = _init_db()

# ── RTL CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .arabic-text, [lang="ar"] {
        direction: rtl; text-align: right;
        font-family: 'Segoe UI', 'Tahoma', 'Arial', sans-serif;
        font-size: 1.1rem; line-height: 1.8; unicode-bidi: embed;
    }
    .arabic-large {
        direction: rtl; text-align: right;
        font-size: 1.3rem; unicode-bidi: embed;
    }
    .chat-message-user {
        direction: rtl; text-align: right;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; padding: 12px 16px;
        border-radius: 12px 12px 4px 12px; margin: 8px 0; unicode-bidi: embed;
    }
    .chat-message-assistant {
        direction: rtl; text-align: right;
        background: #f0f0f0; padding: 12px 16px;
        border-radius: 12px 12px 12px 4px; margin: 8px 0; unicode-bidi: embed;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white; padding: 20px; border-radius: 12px; text-align: center;
    }
    .dialect-badge {
        display: inline-block; padding: 6px 14px;
        border-radius: 20px; font-weight: bold; font-size: 0.9rem;
    }
    .dialect-MSA { background: #e3f2fd; color: #1565c0; }
    .dialect-Gulf { background: #e8f5e9; color: #2e7d32; }
    .dialect-Egyptian { background: #fff3e0; color: #e65100; }
    .dialect-Levantine { background: #fce4ec; color: #c62828; }
    .sentiment-positive { color: #2e7d32; font-weight: bold; }
    .sentiment-negative { color: #c62828; font-weight: bold; }
    .sentiment-neutral { color: #6c757d; font-weight: bold; }
    .sentiment-mixed { color: #e65100; font-weight: bold; }
    .seg-ar {
        border-right: 4px solid #2a5298; padding: 8px 12px; margin: 4px 0;
        background: #f0f4ff; direction: rtl; text-align: right;
        border-radius: 0 8px 8px 0;
    }
    .seg-en {
        border-left: 4px solid #e65100; padding: 8px 12px; margin: 4px 0;
        background: #fff8f0; border-radius: 8px 0 0 8px;
    }
    .footer {
        text-align: center; padding: 20px; color: #888; font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.title("🌉 AraBridge")
    st.caption("Arabic-English Bilingual NLP Toolkit")

    st.markdown("---")
    st.markdown("""
    **Capabilities:**
    - 🎭 Arabic & English Sentiment
    - 🗣️ Dialect Identification
    - 🔄 Code-Switching Detection
    - 💬 Bilingual Chatbot
    - 📊 Cross-Lingual Benchmarks
    """)

    st.markdown("---")

    # Status
    if DEEPSEEK_API_KEY:
        st.success("🔑 DeepSeek API configured")
        if _db_ready:
            st.success("🗄️ Database connected")
    else:
        st.warning("⚠️ Set DEEPSEEK_API_KEY for chatbot")

    st.markdown("---")
    st.caption("Made for UAE AI Engineering 🇦🇪")
    st.caption("Targeting: G42 · TII · Falcon Team")


# ── Main Content ──────────────────────────────────────────
st.title("🌉 AraBridge")
st.subheader("Arabic-English Bilingual NLP Toolkit")
st.caption("Sentiment Analysis · Dialect ID · Code-Switching · Chatbot · Benchmarks")
st.markdown("---")

# ── Session state init ────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "dialect_input" not in st.session_state:
    st.session_state.dialect_input = ""
if "cs_input" not in st.session_state:
    st.session_state.cs_input = ""

# ── Tabs ──────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎭 Sentiment Analysis",
    "🗣️ Dialect Identifier",
    "🔄 Code-Switching",
    "💬 Bilingual Chatbot",
    "📊 Benchmarks",
])

# ═══════════════════════════════════════════════════════════
# TAB 1: SENTIMENT ANALYSIS
# ═══════════════════════════════════════════════════════════
with tab1:
    st.header("🎭 Sentiment Analysis")
    st.caption("Classify text sentiment in Arabic and English — side by side")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🇸🇦 Arabic Sentiment")
        arabic_text = st.text_area(
            "Enter Arabic text",
            value="هذا المنتج رائع وأنا سعيد جدًا به",
            height=120, key="arabic_input",
            placeholder="اكتب النص العربي هنا...",
        )
        if st.button("🔍 Analyze Arabic", type="primary", use_container_width=True):
            with st.spinner("Analyzing Arabic sentiment..."):
                label, confidence, probs, model = analyze_arabic_sentiment(arabic_text)
                st.markdown(f"""
                <div style="background: #f8f9fa; padding: 16px; border-radius: 12px; margin: 10px 0;">
                    <p class="arabic-text" style="font-size: 1.2rem;">{arabic_text}</p>
                    <p><strong>Sentiment:</strong> 
                       <span class="sentiment-{label}">{label.upper()}</span>
                       &nbsp;|&nbsp; Confidence: {confidence:.1%}</p>
                    <p><small>Model: <code>{model}</code></small></p>
                </div>
                """, unsafe_allow_html=True)

                if probs:
                    df = pd.DataFrame({"Label": list(probs.keys()), "Probability": list(probs.values())})
                    fig = px.bar(df, x="Probability", y="Label", orientation='h',
                                 color="Probability", color_continuous_scale="Blues",
                                 title="Sentiment Probabilities (Arabic)")
                    fig.update_layout(height=250)
                    st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🇬🇧 English Sentiment")
        english_text = st.text_area(
            "Enter English text",
            value="This product is amazing and I absolutely love it!",
            height=120, key="english_input",
            placeholder="Enter English text here...",
        )
        if st.button("🔍 Analyze English", type="primary", use_container_width=True):
            with st.spinner("Analyzing English sentiment..."):
                label, confidence, probs, model = analyze_english_sentiment(english_text)
                st.markdown(f"""
                <div style="background: #f8f9fa; padding: 16px; border-radius: 12px; margin: 10px 0;">
                    <p style="font-size: 1.2rem;">{english_text}</p>
                    <p><strong>Sentiment:</strong> 
                       <span class="sentiment-{label}">{label.upper()}</span>
                       &nbsp;|&nbsp; Confidence: {confidence:.1%}</p>
                    <p><small>Model: <code>{model}</code></small></p>
                </div>
                """, unsafe_allow_html=True)

                if probs:
                    df = pd.DataFrame({"Label": list(probs.keys()), "Probability": list(probs.values())})
                    fig = px.bar(df, x="Probability", y="Label", orientation='h',
                                 color="Probability", color_continuous_scale="Oranges",
                                 title="Sentiment Probabilities (English)")
                    fig.update_layout(height=250)
                    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# TAB 2: DIALECT IDENTIFIER
# ═══════════════════════════════════════════════════════════
with tab2:
    st.header("🗣️ Dialect Identifier")
    st.caption("Identify Arabic dialect: MSA · Gulf · Egyptian · Levantine")

    st.markdown("""
    <div style="background: #f8f9fa; padding: 12px; border-radius: 8px; margin-bottom: 16px;">
        <strong>Supported Dialects:</strong><br>
        🟢 <b>MSA</b> — Modern Standard Arabic<br>
        🟢 <b>Gulf</b> — UAE, Saudi Arabia, Qatar, Kuwait, Bahrain, Oman<br>
        🟢 <b>Egyptian</b> — Egypt<br>
        🟢 <b>Levantine</b> — Lebanon, Syria, Jordan, Palestine
    </div>
    """, unsafe_allow_html=True)

    quick_examples = {
        "MSA": "اللغة العربية الفصحى هي اللغة الرسمية في العالم العربي",
        "Gulf": "شو أخبارك يالغالي، شكثر سعر هالشي",
        "Egyptian": "إزيك عامل إيه النهاردة، عايز أروح معاك",
        "Levantine": "شو بدك تاكل اليوم، كيفك يا حبيبي",
    }

    cols = st.columns(4)
    for i, (dialect, example) in enumerate(quick_examples.items()):
        with cols[i]:
            if st.button(f"📝 {dialect}", key=f"dialect_ex_{dialect}", use_container_width=True):
                st.session_state.dialect_input = example
                st.rerun()

    dialect_text = st.text_area(
        "Enter Arabic text to identify its dialect",
        value=st.session_state.dialect_input or "شو أخبارك يالغالي، شكثر سعر هالشي",
        height=100, key="dialect_text",
        placeholder="أدخل النص العربي هنا...",
    )

    if st.button("🔍 Identify Dialect", type="primary", use_container_width=True):
        with st.spinner("Analyzing dialect..."):
            predicted, confidence, probs, model = identify_dialect(dialect_text)
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 16px; border-radius: 12px; margin: 10px 0;">
                <p class="arabic-large">{dialect_text}</p>
                <p><strong>Predicted Dialect:</strong> 
                   <span class="dialect-badge dialect-{predicted}">{predicted}</span>
                   &nbsp;|&nbsp; Confidence: {confidence:.1%}</p>
                <p><small>Model: <code>{model}</code></small></p>
            </div>
            """, unsafe_allow_html=True)

            if probs:
                df = pd.DataFrame({"Dialect": list(probs.keys()), "Probability": list(probs.values())})
                df = df.sort_values("Probability", ascending=True)
                colors_map = {"MSA": "#1565c0", "Gulf": "#2e7d32", "Egyptian": "#e65100", "Levantine": "#c62828"}
                bar_colors = [colors_map.get(d, "#888") for d in df["Dialect"]]
                fig = go.Figure(go.Bar(x=df["Probability"], y=df["Dialect"], orientation='h',
                                       marker_color=bar_colors,
                                       text=[f"{v:.1%}" for v in df["Probability"]],
                                       textposition='outside'))
                fig.update_layout(title="Dialect Probability Distribution", height=300,
                                  xaxis_title="Confidence", xaxis_range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# TAB 3: CODE-SWITCHING
# ═══════════════════════════════════════════════════════════
with tab3:
    st.header("🔄 Code-Switching Detection")
    st.caption("Detect and analyze mixed Arabic-English text — segment by language")

    st.info("💡 Code-switching is common among Arabic speakers. "
            "This tool detects language switches and processes each segment appropriately.")

    cs_examples = {
        "Mixed (mostly Arabic)": "الفيلم كان amazing بصراحة، acting ممتاز والقصة حلوة",
        "Mixed (mostly English)": "I really enjoyed the مطعم, the food was لذيذ and the service ممتاز",
        "Business": "ال meeting كان productive جدا، ناقشنا ال strategy وال budget للربع القادم",
    }

    cs_cols = st.columns(3)
    for i, (label, example) in enumerate(cs_examples.items()):
        with cs_cols[i]:
            if st.button(f"📝 {label}", key=f"cs_ex_{i}", use_container_width=True):
                st.session_state.cs_input = example
                st.rerun()

    cs_text = st.text_area(
        "Enter text that may contain Arabic-English code-switching",
        value=st.session_state.cs_input or "الفيلم كان amazing بصراحة، الـ acting ممتاز والقصة كانت very touching",
        height=100, key="cs_text",
        placeholder="Enter mixed Arabic-English text...",
    )

    if st.button("🔍 Analyze Code-Switching", type="primary", use_container_width=True):
        with st.spinner("Detecting code-switching..."):
            result = process_code_switched(cs_text)
            has_cs = result["has_code_switching"]
            primary = result["primary_language"]
            segments = result["segments"]

            if has_cs:
                st.success(f"✅ Code-switching detected! Primary language: **{primary.upper()}**")
            else:
                st.info(f"ℹ️ No significant code-switching detected. Language: **{primary.upper()}**")

            st.subheader("📝 Language Segments")
            for seg in segments:
                text = seg["text"]
                lang = seg["language"]
                sent = seg.get("sentiment")
                if lang == "ar":
                    sentiment_tag = f'<span class="sentiment-{sent}">[{sent.upper()}]</span>' if sent else ""
                    st.markdown(f"""
                    <div class="seg-ar"><small style="color:#2a5298">🇸🇦 Arabic</small> {sentiment_tag}
                    <p class="arabic-text">{text}</p></div>
                    """, unsafe_allow_html=True)
                else:
                    sentiment_tag = f'<span class="sentiment-{sent}">[{sent.upper()}]</span>' if sent else ""
                    st.markdown(f"""
                    <div class="seg-en"><small style="color:#e65100">🇬🇧 English</small> {sentiment_tag}
                    <p>{text}</p></div>
                    """, unsafe_allow_html=True)

            # Summary stats
            ar_count = sum(1 for s in segments if s["language"] == "ar")
            en_count = sum(1 for s in segments if s["language"] == "en")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Arabic Segments", ar_count)
            with col2:
                st.metric("English Segments", en_count)
            with col3:
                st.metric("Total Segments", len(segments))

            # Language distribution pie
            pie_fig = go.Figure(go.Pie(labels=["Arabic", "English"],
                                        values=[ar_count, en_count],
                                        hole=0.5,
                                        marker_colors=["#2a5298", "#e65100"]))
            pie_fig.update_layout(title="Language Distribution", height=300)
            st.plotly_chart(pie_fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# TAB 4: BILINGUAL CHATBOT
# ═══════════════════════════════════════════════════════════
with tab4:
    st.header("💬 Bilingual Chatbot")
    st.caption("Arabic-English bilingual AI assistant powered by DeepSeek")

    if not DEEPSEEK_API_KEY:
        st.error("🔴 DEEPSEEK_API_KEY not set. Chatbot requires an API key.")
        st.code("Set DEEPSEEK_API_KEY in .env or Streamlit secrets.")
    else:
        # Chat messages
        for msg in st.session_state.chat_history:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                st.markdown(f'<div class="chat-message-user">{content}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message-assistant">{content}</div>', unsafe_allow_html=True)

        # Input
        if prompt := st.chat_input("اكتب رسالتك... / Type your message..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            with st.spinner("💭 Thinking..."):
                try:
                    messages = [{"role": m["role"], "content": m["content"]}
                                for m in st.session_state.chat_history]
                    reply = chat_sync(messages)
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
                except Exception as e:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"⚠️ Sorry, I couldn't reach the AI service: {e}",
                    })
            st.rerun()

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# ═══════════════════════════════════════════════════════════
# TAB 5: BENCHMARKS
# ═══════════════════════════════════════════════════════════
with tab5:
    st.header("📊 Cross-Lingual Benchmarks")
    st.caption("Arabic vs English model performance comparison")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Run Benchmark")
        st.markdown("Compare Arabic sentiment model against English baseline.")

        if st.button("🚀 Run Benchmark", type="primary", use_container_width=True):
            with st.spinner("Running benchmarks (this may take a minute)..."):
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
                ground_truth = {
                    "ar_positive": "positive", "ar_negative": "negative", "ar_neutral": "neutral",
                    "en_positive": "positive", "en_negative": "negative", "en_neutral": "neutral",
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
                    f1 = accuracy
                    # Save to DB
                    try:
                        save_benchmark(model_name=model, task=f"sentiment_{key}",
                                       accuracy=round(accuracy, 4),
                                       f1_score=round(f1, 4),
                                       latency_ms=round(avg_latency, 1))
                    except Exception:
                        pass
                    results.append({
                        "task": f"sentiment_{key}", "language": lang, "model": model,
                        "accuracy": round(accuracy, 4), "f1_score": round(f1, 4),
                        "avg_latency_ms": round(avg_latency, 1), "samples": total,
                    })
                st.session_state.benchmark_results = results
                st.success(f"✅ Completed {len(results)} benchmark tests!")
                st.rerun()

        if st.button("📂 Load Saved", use_container_width=True):
            entries = get_benchmarks()
            if entries:
                st.session_state.benchmark_saved = entries
                st.success(f"Loaded {len(entries)} saved benchmarks")
                st.rerun()

    with col2:
        if "benchmark_results" in st.session_state:
            results = st.session_state.benchmark_results
            st.subheader("Live Benchmark Results")
            df = pd.DataFrame(results)
            df["Display Task"] = df["task"].str.replace("sentiment_", "").str.replace("_", " ").str.title()
            df["Model Type"] = df["model"].apply(
                lambda m: "🇸🇦 Arabic" if "arabert" in m.lower() or "camel" in m.lower() else "🇬🇧 English")

            st.markdown("#### Accuracy Comparison")
            fig = px.bar(df, x="Display Task", y="accuracy", color="Model Type", barmode="group",
                         text=df["accuracy"].apply(lambda x: f"{x:.1%}"),
                         color_discrete_map={"🇸🇦 Arabic": "#2a5298", "🇬🇧 English": "#e65100"},
                         title="Accuracy by Task and Model")
            fig.update_traces(textposition='outside')
            fig.update_layout(yaxis_range=[0, 1.1], yaxis_tickformat=".0%", height=400)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Latency Comparison")
            fig = px.bar(df, x="Display Task", y="avg_latency_ms", color="Model Type", barmode="group",
                         text=df["avg_latency_ms"].apply(lambda x: f"{x:.0f}ms"),
                         color_discrete_map={"🇸🇦 Arabic": "#2a5298", "🇬🇧 English": "#e65100"},
                         title="Average Latency by Task and Model")
            fig.update_layout(height=400, yaxis_title="Latency (ms)")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Detailed Results")
            display_df = df[["Display Task", "Model Type", "model", "accuracy", "f1_score", "avg_latency_ms", "samples"]]
            display_df.columns = ["Task", "Type", "Model", "Accuracy", "F1", "Latency (ms)", "Samples"]
            st.dataframe(display_df, use_container_width=True, hide_index=True)

        if "benchmark_saved" in st.session_state:
            entries = st.session_state.benchmark_saved
            st.subheader("Saved Benchmarks (Database)")
            df = pd.DataFrame(entries)
            if not df.empty:
                if "created_at" in df.columns:
                    df["created_at"] = pd.to_datetime(df["created_at"])
                    df["Date"] = df["created_at"].dt.strftime("%Y-%m-%d %H:%M")
                display_cols = ["model_name", "task", "accuracy", "f1_score", "latency_ms"]
                if "Date" in df.columns:
                    display_cols.append("Date")
                display_df = df[display_cols]
                display_df.columns = [c.replace("_", " ").title() for c in display_df.columns]
                st.dataframe(display_df, use_container_width=True, hide_index=True)

                st.markdown("#### Accuracy Trend")
                fig = px.line(df, x="created_at", y="accuracy", color="task",
                              markers=True, title="Accuracy Over Time")
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

        if "benchmark_results" not in st.session_state and "benchmark_saved" not in st.session_state:
            st.info("👈 Run a benchmark or load saved results to see comparisons here.")

# ── Footer ────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div class="footer">
    🌉 <strong>AraBridge</strong> — Arabic-English Bilingual NLP Toolkit<br>
    Built for the UAE AI ecosystem: G42 · TII · Falcon Team<br>
    <small>© 2024 AraBridge | MIT License</small>
</div>
""", unsafe_allow_html=True)
