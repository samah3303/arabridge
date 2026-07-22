"""
AraBridge — Arabic-English Bilingual NLP Toolkit
Streamlit Frontend with RTL Support
"""

import streamlit as st
import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
from pathlib import Path
from dotenv import load_dotenv

# Load env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ── Page Config ─────────────────────────────────────────

st.set_page_config(
    page_title="AraBridge | Arabic-English NLP Toolkit",
    page_icon="🌉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── RTL CSS ─────────────────────────────────────────────

st.markdown("""
<style>
    /* RTL support for Arabic text */
    .arabic-text, [lang="ar"] {
        direction: rtl;
        text-align: right;
        font-family: 'Segoe UI', 'Tahoma', 'Arial', sans-serif;
        font-size: 1.1rem;
        line-height: 1.8;
        unicode-bidi: embed;
    }

    /* General Arabic styling */
    .arabic-large {
        direction: rtl;
        text-align: right;
        font-size: 1.3rem;
        unicode-bidi: embed;
    }

    /* Chat messages */
    .chat-message-user {
        direction: rtl;
        text-align: right;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 16px;
        border-radius: 12px 12px 4px 12px;
        margin: 8px 0;
        unicode-bidi: embed;
    }
    .chat-message-assistant {
        direction: rtl;
        text-align: right;
        background: #f0f0f0;
        padding: 12px 16px;
        border-radius: 12px 12px 12px 4px;
        margin: 8px 0;
        unicode-bidi: embed;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
    }
    .metric-card h3 {
        margin: 0;
        font-size: 2rem;
    }
    .metric-card p {
        margin: 4px 0 0 0;
        opacity: 0.9;
    }

    /* Flag styling for dialect */
    .dialect-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .dialect-MSA { background: #e3f2fd; color: #1565c0; }
    .dialect-Gulf { background: #e8f5e9; color: #2e7d32; }
    .dialect-Egyptian { background: #fff3e0; color: #e65100; }
    .dialect-Levantine { background: #fce4ec; color: #c62828; }

    /* Sentiment badges */
    .sentiment-positive { color: #2e7d32; font-weight: bold; }
    .sentiment-negative { color: #c62828; font-weight: bold; }
    .sentiment-neutral { color: #6c757d; font-weight: bold; }
    .sentiment-mixed { color: #e65100; font-weight: bold; }

    /* Code-switch segment cards */
    .seg-ar {
        border-right: 4px solid #2a5298;
        padding: 8px 12px;
        margin: 4px 0;
        background: #f0f4ff;
        direction: rtl;
        text-align: right;
        border-radius: 0 8px 8px 0;
    }
    .seg-en {
        border-left: 4px solid #e65100;
        padding: 8px 12px;
        margin: 4px 0;
        background: #fff8f0;
        border-radius: 8px 0 0 8px;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
        color: #888;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ── API Client ──────────────────────────────────────────

API_BASE = os.getenv("ARABRIDGE_API_URL", "http://localhost:8000")


def api_post(endpoint: str, json_data: dict) -> dict | None:
    """POST to the AraBridge API with error handling."""
    try:
        response = httpx.post(
            f"{API_BASE}{endpoint}",
            json=json_data,
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()
    except httpx.ConnectError:
        st.error("⚠️ Cannot connect to AraBridge API. "
                 "Make sure the backend is running: `uvicorn backend.main:app`")
        return None
    except httpx.HTTPStatusError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        st.error(f"⚠️ API Error: {detail}")
        return None
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
        return None


def api_get(endpoint: str, params: dict | None = None) -> dict | None:
    """GET from the AraBridge API."""
    try:
        response = httpx.get(
            f"{API_BASE}{endpoint}",
            params=params,
            timeout=10.0,
        )
        response.raise_for_status()
        return response.json()
    except httpx.ConnectError:
        st.error("⚠️ Cannot connect to AraBridge API.")
        return None
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
        return None


# ── Sidebar ─────────────────────────────────────────────

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

    api_status = st.empty()
    try:
        health = httpx.get(f"{API_BASE}/health", timeout=3.0)
        if health.status_code == 200:
            api_status.success("✅ API Connected")
        else:
            api_status.warning("⚠️ API Unhealthy")
    except Exception:
        api_status.error("❌ API Offline — start backend first")
        st.code("cd backend && uvicorn main:app --reload")

    st.markdown("---")
    st.caption("Made for UAE AI Engineering 🇦🇪")
    st.caption("Targeting: G42 · TII · Falcon Team")

# ── Main Content ────────────────────────────────────────

st.title("🌉 AraBridge")
st.subheader("Arabic-English Bilingual NLP Toolkit")
st.caption("Sentiment Analysis · Dialect ID · Code-Switching · Chatbot · Benchmarks")

st.markdown("---")

# ── Tabs ────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎭 Sentiment Analysis",
    "🗣️ Dialect Identifier",
    "🔄 Code-Switching",
    "💬 Bilingual Chatbot",
    "📊 Benchmarks",
])

# ═══════════════════════════════════════════════════════
# TAB 1: SENTIMENT ANALYSIS
# ═══════════════════════════════════════════════════════

with tab1:
    st.header("🎭 Sentiment Analysis")
    st.caption("Classify text sentiment in Arabic and English — side by side")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🇸🇦 Arabic Sentiment")
        arabic_text = st.text_area(
            "Enter Arabic text",
            value="هذا المنتج رائع وأنا سعيد جدًا به",
            height=120,
            key="arabic_input",
            placeholder="اكتب النص العربي هنا...",
        )
        if st.button("🔍 Analyze Arabic", type="primary", use_container_width=True):
            with st.spinner("Analyzing Arabic sentiment..."):
                result = api_post("/sentiment", {
                    "text": arabic_text,
                    "language": "ar",
                })
                if result:
                    sentiment = result["sentiment"]
                    confidence = result["confidence"]
                    probs = result["probabilities"]
                    model = result["model_used"]

                    st.markdown(f"""
                    <div style="background: #f8f9fa; padding: 16px; border-radius: 12px; margin: 10px 0;">
                        <p class="arabic-text" style="font-size: 1.2rem;">{arabic_text}</p>
                        <p><strong>Sentiment:</strong> 
                           <span class="sentiment-{sentiment}">{sentiment.upper()}</span>
                           &nbsp;|&nbsp; Confidence: {confidence:.1%}</p>
                        <p><small>Model: <code>{model}</code></small></p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Probabilities chart
                    if probs:
                        df = pd.DataFrame({
                            "Label": list(probs.keys()),
                            "Probability": list(probs.values()),
                        })
                        fig = px.bar(
                            df, x="Probability", y="Label",
                            orientation='h',
                            color="Probability",
                            color_continuous_scale="Blues",
                            title="Sentiment Probabilities (Arabic)",
                        )
                        fig.update_layout(height=250)
                        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🇬🇧 English Sentiment")
        english_text = st.text_area(
            "Enter English text",
            value="This product is amazing and I absolutely love it!",
            height=120,
            key="english_input",
            placeholder="Enter English text here...",
        )
        if st.button("🔍 Analyze English", type="primary", use_container_width=True):
            with st.spinner("Analyzing English sentiment..."):
                result = api_post("/sentiment", {
                    "text": english_text,
                    "language": "en",
                })
                if result:
                    sentiment = result["sentiment"]
                    confidence = result["confidence"]
                    probs = result["probabilities"]
                    model = result["model_used"]

                    st.markdown(f"""
                    <div style="background: #f8f9fa; padding: 16px; border-radius: 12px; margin: 10px 0;">
                        <p style="font-size: 1.2rem;">{english_text}</p>
                        <p><strong>Sentiment:</strong> 
                           <span class="sentiment-{sentiment}">{sentiment.upper()}</span>
                           &nbsp;|&nbsp; Confidence: {confidence:.1%}</p>
                        <p><small>Model: <code>{model}</code></small></p>
                    </div>
                    """, unsafe_allow_html=True)

                    if probs:
                        df = pd.DataFrame({
                            "Label": list(probs.keys()),
                            "Probability": list(probs.values()),
                        })
                        fig = px.bar(
                            df, x="Probability", y="Label",
                            orientation='h',
                            color="Probability",
                            color_continuous_scale="Oranges",
                            title="Sentiment Probabilities (English)",
                        )
                        fig.update_layout(height=250)
                        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════
# TAB 2: DIALECT IDENTIFIER
# ═══════════════════════════════════════════════════════

with tab2:
    st.header("🗣️ Dialect Identifier")
    st.caption("Identify Arabic dialect: MSA · Gulf · Egyptian · Levantine")

    st.markdown("""
    <div style="background: #f8f9fa; padding: 12px; border-radius: 8px; margin-bottom: 16px;">
        <strong>Supported Dialects:</strong><br>
        🟢 <b>MSA</b> — Modern Standard Arabic (اللغة العربية الفصحى), used in media & formal contexts<br>
        🟢 <b>Gulf</b> — UAE, Saudi Arabia, Qatar, Kuwait, Bahrain, Oman<br>
        🟢 <b>Egyptian</b> — Egypt and widely understood across the Arab world<br>
        🟢 <b>Levantine</b> — Lebanon, Syria, Jordan, Palestine
    </div>
    """, unsafe_allow_html=True)

    # Quick examples
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
        value=st.session_state.get("dialect_input", "شو أخبارك يالغالي، شكثر سعر هالشي"),
        height=100,
        key="dialect_text",
        placeholder="أدخل النص العربي هنا...",
    )

    if st.button("🔍 Identify Dialect", type="primary", use_container_width=True):
        with st.spinner("Analyzing dialect..."):
            result = api_post("/dialect", {"text": dialect_text})
            if result:
                predicted = result["predicted_dialect"]
                confidence = result["confidence"]
                probs = result["dialect_probabilities"]
                model = result["model_used"]

                st.markdown(f"""
                <div style="background: #f8f9fa; padding: 16px; border-radius: 12px; margin: 10px 0;">
                    <p class="arabic-large">{dialect_text}</p>
                    <p><strong>Predicted Dialect:</strong> 
                       <span class="dialect-badge dialect-{predicted}">{predicted}</span>
                       &nbsp;|&nbsp; Confidence: {confidence:.1%}</p>
                    <p><small>Model: <code>{model}</code></small></p>
                </div>
                """, unsafe_allow_html=True)

                # Dialect probabilities
                if probs:
                    df = pd.DataFrame({
                        "Dialect": list(probs.keys()),
                        "Probability": list(probs.values()),
                    }).sort_values("Probability", ascending=True)

                    colors = {
                        "MSA": "#1565c0",
                        "Gulf": "#2e7d32",
                        "Egyptian": "#e65100",
                        "Levantine": "#c62828",
                    }
                    bar_colors = [colors.get(d, "#888") for d in df["Dialect"]]

                    fig = go.Figure(go.Bar(
                        x=df["Probability"],
                        y=df["Dialect"],
                        orientation='h',
                        marker_color=bar_colors,
                        text=[f"{v:.1%}" for v in df["Probability"]],
                        textposition='outside',
                    ))
                    fig.update_layout(
                        title="Dialect Probability Distribution",
                        height=300,
                        xaxis_title="Confidence",
                        xaxis_range=[0, 1],
                    )
                    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════
# TAB 3: CODE-SWITCHING
# ═══════════════════════════════════════════════════════

with tab3:
    st.header("🔄 Code-Switching Detection")
    st.caption("Detect and analyze mixed Arabic-English text — segment by language")

    st.info(
        "💡 Code-switching is common among Arabic speakers. "
        "This tool detects language switches and processes each segment appropriately."
    )

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
        value=st.session_state.get(
            "cs_input",
            "الفيلم كان amazing بصراحة، الـ acting ممتاز والقصة كانت very touching"
        ),
        height=100,
        key="cs_text",
        placeholder="Enter mixed Arabic-English text...",
    )

    if st.button("🔍 Analyze Code-Switching", type="primary", use_container_width=True):
        with st.spinner("Detecting code-switching..."):
            result = api_post("/code-switching", {"text": cs_text})
            if result:
                has_cs = result["has_code_switching"]
                primary = result["primary_language"]
                segments = result["segments"]

                # Summary
                if has_cs:
                    st.success(f"✅ Code-switching detected! Primary language: **{primary.upper()}**")
                else:
                    st.info(f"ℹ️ No significant code-switching detected. Language: **{primary.upper()}**")

                # Segments
                st.subheader("📝 Language Segments")
                for seg in segments:
                    text = seg["text"]
                    lang = seg["language"]
                    sent = seg.get("sentiment")

                    if lang == "ar":
                        sentiment_tag = f'<span class="sentiment-{sent}">[{sent.upper()}]</span>' if sent else ""
                        st.markdown(f"""
                        <div class="seg-ar">
                            <small style="color:#2a5298;">🇸🇦 Arabic segment</small>
                            {sentiment_tag}
                            <p style="font-size:1.1rem;">{text}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif lang == "en":
                        sentiment_tag = f'<span class="sentiment-{sent}">[{sent.upper()}]</span>' if sent else ""
                        st.markdown(f"""
                        <div class="seg-en">
                            <small style="color:#e65100;">🇬🇧 English segment</small>
                            {sentiment_tag}
                            <p style="font-size:1.1rem;">{text}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="padding:8px 12px; margin:4px 0; color:#888;">
                            <small>Other</small>
                            <p>{text}</p>
                        </div>
                        """, unsafe_allow_html=True)

                # Statistics
                if result.get("primary_language") == "ar":
                    ar_pct = 100 - sum(
                        seg.get("lang_pct", 0) for seg in segments if seg["language"] != "ar"
                    )

# ═══════════════════════════════════════════════════════
# TAB 4: BILINGUAL CHATBOT
# ═══════════════════════════════════════════════════════

with tab4:
    st.header("💬 Bilingual Chatbot")
    st.caption("Powered by DeepSeek — speaks Arabic, English, and everything in between")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="chat-message-user">
                    <small>👤 You</small><br>
                    {msg['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message-assistant">
                    <small>🤖 AraBridge</small><br>
                    {msg['content']}
                </div>
                """, unsafe_allow_html=True)

    # Chat input
    st.markdown("---")
    chat_input = st.chat_input("Type your message in Arabic, English, or both...")

    if chat_input:
        # Add user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": chat_input,
        })

        with st.spinner("💭 Thinking..."):
            # Convert to API format
            messages = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in st.session_state.chat_history
            ]

            result = api_post("/chat", {"messages": messages})
            if result:
                reply = result["reply"]
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": reply,
                })
            else:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": "⚠️ Sorry, I couldn't reach the AI service. Please check your DeepSeek API key and try again.",
                })

        st.rerun()

    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ═══════════════════════════════════════════════════════
# TAB 5: BENCHMARKS
# ═══════════════════════════════════════════════════════

with tab5:
    st.header("📊 Cross-Lingual Benchmarks")
    st.caption("Arabic vs English model performance comparison")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Run Benchmark")
        st.markdown("""
        Compare Arabic sentiment model performance against the English baseline on 
        a standard test set.
        """)

        if st.button("🚀 Run Benchmark", type="primary", use_container_width=True):
            with st.spinner("Running benchmarks (this may take a minute)..."):
                result = api_post("/benchmarks", {})
                if result:
                    st.success(f"✅ Completed {result['total_tests']} benchmark tests!")

                    # Store results
                    st.session_state.benchmark_results = result["results"]

        # Load from DB
        if st.button("📂 Load Saved", use_container_width=True):
            result = api_get("/benchmarks")
            if result and result.get("entries"):
                entries = result["entries"]
                st.session_state.benchmark_saved = entries
                st.success(f"Loaded {len(entries)} saved benchmarks")

    with col2:
        # Show live results
        if "benchmark_results" in st.session_state:
            results = st.session_state.benchmark_results

            st.subheader("Live Benchmark Results")

            df = pd.DataFrame(results)
            # Add display-friendly labels
            df["Display Task"] = df["task"].str.replace("sentiment_", "").str.replace("_", " ").str.title()
            df["Model Type"] = df["model"].apply(
                lambda m: "🇸🇦 Arabic" if "arabert" in m.lower() or "camel" in m.lower() else "🇬🇧 English"
            )

            # Accuracy comparison
            st.markdown("#### Accuracy Comparison")
            fig = px.bar(
                df,
                x="Display Task",
                y="accuracy",
                color="Model Type",
                barmode="group",
                text=df["accuracy"].apply(lambda x: f"{x:.1%}"),
                color_discrete_map={"🇸🇦 Arabic": "#2a5298", "🇬🇧 English": "#e65100"},
                title="Accuracy by Task and Model",
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(
                yaxis_range=[0, 1.1],
                yaxis_tickformat=".0%",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Latency comparison
            st.markdown("#### Latency Comparison")
            fig = px.bar(
                df,
                x="Display Task",
                y="avg_latency_ms",
                color="Model Type",
                barmode="group",
                text=df["avg_latency_ms"].apply(lambda x: f"{x:.0f}ms"),
                color_discrete_map={"🇸🇦 Arabic": "#2a5298", "🇬🇧 English": "#e65100"},
                title="Average Latency by Task and Model",
            )
            fig.update_layout(height=400, yaxis_title="Latency (ms)")
            st.plotly_chart(fig, use_container_width=True)

            # Summary table
            st.markdown("#### Detailed Results")
            display_df = df[["Display Task", "Model Type", "model", "accuracy", "f1_score", "avg_latency_ms", "samples"]]
            display_df.columns = ["Task", "Type", "Model", "Accuracy", "F1", "Latency (ms)", "Samples"]
            st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Show saved benchmarks
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

                # Trends
                st.markdown("#### Accuracy Trend")
                fig = px.line(
                    df, x="created_at", y="accuracy", color="task",
                    markers=True, title="Accuracy Over Time",
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

        # Empty state
        if "benchmark_results" not in st.session_state and "benchmark_saved" not in st.session_state:
            st.info("👈 Run a benchmark or load saved results to see comparisons here.")

# ── Footer ──────────────────────────────────────────────

st.markdown("---")
st.markdown("""
<div class="footer">
    🌉 <strong>AraBridge</strong> — Arabic-English Bilingual NLP Toolkit<br>
    Built for the UAE AI ecosystem: G42 · TII · Falcon Team<br>
    <small>© 2024 AraBridge | MIT License</small>
</div>
""", unsafe_allow_html=True)
