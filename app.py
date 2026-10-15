
"""
Streamlit Web Application: Emotion Detection & Sentiment Analysis
NLP College Final Project

Frontend:
- Blue/purple gradient hero banner
- Emotion Detection with seven sample buttons
- Emotion and sentiment result cards
- Emotion probability distribution chart
- Analytics dashboard
- Existing backend and prediction logic preserved
"""

import os
import sys

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

# ---------------------------------------------------------------------
# 1. PROJECT PATHS AND EXISTING BACKEND
# ---------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocessing import preprocess_text
from src.predict import (
    predict_emotion,
    predict_sentiment,
    predict_batch,
    EMOTION_MODEL_PATH,
    EMOTION_VEC_PATH,
    SENTIMENT_MODEL_PATH,
    SENTIMENT_VEC_PATH,
)

# ---------------------------------------------------------------------
# 2. PAGE CONFIGURATION
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Emotion Detection AI",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# 3. FRONTEND STYLING
# ---------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap'
    );

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Blue and purple hero banner */
    .hero-banner {
        background: linear-gradient(
            135deg,
            #1e3a8a 0%,
            #3b82f6 50%,
            #6366f1 100%
        );
        color: white;
        padding: 2rem 2.3rem;
        border-radius: 18px;
        box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.3);
        margin-bottom: 1.5rem;
    }

    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.25;
        margin-bottom: 0.6rem;
    }

    .hero-subtitle {
        font-size: 1rem;
        opacity: 0.94;
        line-height: 1.6;
        margin-bottom: 0;
    }

    /* Dashboard metric cards */
    .stat-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.2rem 0.8rem;
        text-align: center;
        min-height: 120px;
    }

    .stat-number {
        font-size: 1.9rem;
        font-weight: 800;
        color: #2563eb;
    }

    .stat-label {
        font-size: 0.8rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* Emotion result card */
    .emotion-result-box {
        border-radius: 18px;
        padding: 1.7rem;
        text-align: center;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.12);
        min-height: 270px;
    }

    .emotion-title-tag {
        font-size: 0.8rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        font-weight: 700;
        opacity: 0.9;
    }

    .emotion-emoji {
        font-size: 3.7rem;
        line-height: 1.3;
        margin-top: 0.4rem;
    }

    .emotion-name {
        font-size: 2rem;
        font-weight: 800;
        margin: 0.2rem 0;
    }

    .emotion-conf {
        font-size: 1.05rem;
        font-weight: 600;
    }

    /* Sentiment card */
    .sentiment-result-box {
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 1.7rem 1rem;
        text-align: center;
        background: #ffffff;
        box-shadow: 0 5px 16px rgba(15, 23, 42, 0.06);
        min-height: 270px;
    }

    /* Pipeline cards */
    .flow-step {
        background: #f1f5f9;
        border-left: 4px solid #6366f1;
        padding: 0.85rem 1rem;
        border-radius: 0 9px 9px 0;
        margin-bottom: 0.7rem;
        color: #334155;
        line-height: 1.5;
    }

    /* Section headings */
    .section-subtitle {
        color: #64748b;
        font-size: 0.95rem;
        margin-top: -0.4rem;
    }

    /* Sidebar heading */
    section[data-testid="stSidebar"] h2 {
        color: #1e40af;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# 4. EMOTION AND SENTIMENT CONFIGURATION
# ---------------------------------------------------------------------

EMOTION_CONFIG = {
    "Joy": {
        "emoji": "😊",
        "color": "#10b981",
        "bg_gradient": "linear-gradient(135deg, #059669 0%, #10b981 100%)",
        "description": "Happiness, delight, celebration, or triumph.",
    },
    "Sadness": {
        "emoji": "😢",
        "color": "#3b82f6",
        "bg_gradient": "linear-gradient(135deg, #1d4ed8 0%, #3b82f6 100%)",
        "description": "Sorrow, grief, loss, or dejection.",
    },
    "Anger": {
        "emoji": "😠",
        "color": "#ef4444",
        "bg_gradient": "linear-gradient(135deg, #b91c1c 0%, #ef4444 100%)",
        "description": "Frustration, rage, unfairness, or hostility.",
    },
    "Fear": {
        "emoji": "😨",
        "color": "#8b5cf6",
        "bg_gradient": "linear-gradient(135deg, #6d28d9 0%, #8b5cf6 100%)",
        "description": "Fear, dread, vulnerability, or panic.",
    },
    "Surprise": {
        "emoji": "😲",
        "color": "#f59e0b",
        "bg_gradient": "linear-gradient(135deg, #d97706 0%, #f59e0b 100%)",
        "description": "Wonder, astonishment, or an unexpected event.",
    },
    "Disgust": {
        "emoji": "🤢",
        "color": "#84cc16",
        "bg_gradient": "linear-gradient(135deg, #4d7c0f 0%, #84cc16 100%)",
        "description": "Repulsion towards unpleasant or offensive things.",
    },
    "Neutral": {
        "emoji": "😐",
        "color": "#64748b",
        "bg_gradient": "linear-gradient(135deg, #475569 0%, #64748b 100%)",
        "description": "Factual, informational, or emotionally unbiased expression.",
    },
    "Depressed": {
        "emoji": "😔",
        "color": "#6b7280",
        "bg_gradient": "linear-gradient(135deg, #374151 0%, #6b7280 100%)",
        "description": "A low or discouraged emotional state.",
    },
}

SENTIMENT_CONFIG = {
    "Positive": {
        "emoji": "🟢",
        "color": "#10b981",
    },
    "Negative": {
        "emoji": "🔴",
        "color": "#ef4444",
    },
    "Neutral": {
        "emoji": "⚪",
        "color": "#64748b",
    },
}

# ---------------------------------------------------------------------
# 5. EXISTING MODEL LOADING
# ---------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_cached_models():
    """Load the existing trained models and TF-IDF vectorizers."""

    models_exist = (
        os.path.exists(EMOTION_MODEL_PATH)
        and os.path.exists(EMOTION_VEC_PATH)
        and os.path.exists(SENTIMENT_MODEL_PATH)
        and os.path.exists(SENTIMENT_VEC_PATH)
    )

    if not models_exist:
        return None, None, None, None

    try:
        em_model = joblib.load(EMOTION_MODEL_PATH)
        em_vec = joblib.load(EMOTION_VEC_PATH)
        se_model = joblib.load(SENTIMENT_MODEL_PATH)
        se_vec = joblib.load(SENTIMENT_VEC_PATH)

        return em_model, em_vec, se_model, se_vec

    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None, None


# ---------------------------------------------------------------------
# 6. ANALYTICS SESSION HISTORY
# ---------------------------------------------------------------------

if "history" not in st.session_state:
    # Retain the original initial sample records.
    st.session_state.history = [
        {
            "text": "I finally passed my exam! I am extremely happy.",
            "emotion": "Joy",
            "sentiment": "Positive",
            "confidence": 92.4,
        },
        {
            "text": "I feel so lonely and depressed today, nothing works.",
            "emotion": "Sadness",
            "sentiment": "Negative",
            "confidence": 88.7,
        },
        {
            "text": "How dare you speak to me with such disrespectful attitude!",
            "emotion": "Anger",
            "sentiment": "Negative",
            "confidence": 94.1,
        },
        {
            "text": "I am terrified about failing the technical viva tomorrow.",
            "emotion": "Fear",
            "sentiment": "Negative",
            "confidence": 85.3,
        },
        {
            "text": "Wow, I never expected to win the first prize trophy!",
            "emotion": "Surprise",
            "sentiment": "Positive",
            "confidence": 89.2,
        },
        {
            "text": "The rotten milk smelled foul and made me gag.",
            "emotion": "Disgust",
            "sentiment": "Negative",
            "confidence": 91.0,
        },
        {
            "text": "The seminar on data structures begins at 10 AM.",
            "emotion": "Neutral",
            "sentiment": "Neutral",
            "confidence": 95.8,
        },
    ]

# ---------------------------------------------------------------------
# 7. SIDEBAR NAVIGATION
# ---------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding:1rem 0;">
            <div style="font-size:3.2rem;">🧠</div>
            <h2 style="margin:0.3rem 0; font-weight:800; font-size:1.35rem;">
                Emotion &amp; Sentiment AI
            </h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    selected_page = st.radio(
        "Navigation Menu",
        options=[
            "😊 Emotion Detection",
            "📈 Analytics",
        ],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")

    st.markdown("### ⚙️ System Status")

    em_m, em_v, se_m, se_v = load_cached_models()

    if em_m is not None:
        st.success("✅ Models Loaded & Ready")
    else:
        st.error("⚠️ Models Not Found")
        st.caption("Please run `python train_model.py`.")

    st.markdown("---")

    st.caption("NLP · Emotion Detection · Sentiment Analysis")

# ---------------------------------------------------------------------
# 8. EMOTION DETECTION PAGE
# ---------------------------------------------------------------------

if selected_page == "😊 Emotion Detection":

    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-title">
                😊 Emotion Detection &amp; Sentiment Analysis
            </div>
            <div class="hero-subtitle">
                Understand the emotions and sentiment expressed in text
                using Natural Language Processing and Machine Learning.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    em_model, em_vec, se_model, se_vec = load_cached_models()

    if em_model is None:
        st.error("⚠️ Model files were not found.")

        st.markdown(
            """
            To generate predictions:

            1. Open your project folder in the terminal.
            2. Run `python train_model.py`.
            3. Refresh this application.
            """
        )

    else:
        st.markdown("### ✍️ Enter Your Text")
        st.markdown(
            '<p class="section-subtitle">'
            'Choose a sample comment or type your own text below.'
            '</p>',
            unsafe_allow_html=True,
        )

        samples = [
            ("Joy", "I finally passed my exam! I am extremely happy."),
            ("Sadness", "I feel so heartbroken, lonely, and hopeless today."),
            ("Anger", "Stop lying to my face, this makes my blood boil!"),
            ("Fear", "I am terrified of failing my viva examination tomorrow."),
            ("Surprise", "Wow, I never expected to see you here in this city!"),
            ("Disgust", "The spoiled milk smelled utterly rancid."),
            ("Neutral", "The library operates from nine in the morning until eight."),
        ]

        if "input_text" not in st.session_state:
            st.session_state.input_text = samples[0][1]

        st.info(
            "💬 Select an emotion button to fill in a sample, "
            "or enter your own comment."
        )

        # Seven sample emotion buttons
        sample_cols = st.columns(7)

        for i, (label, sample_val) in enumerate(samples):
            if sample_cols[i].button(
                f"{EMOTION_CONFIG[label]['emoji']} {label}",
                key=f"sample_{label}",
                use_container_width=True,
            ):
                st.session_state.input_text = sample_val
                st.rerun()

        user_text = st.text_area(
            "Your text",
            key="input_text",
            height=130,
            placeholder="Example: BROOO I ATE THAT PRESENTATION UP 🔥",
            help="Enter up to 10,000 characters.",
        )

        col_btn, col_spacer = st.columns([1.5, 5])

        with col_btn:
            analyze_button = st.button(
                "🔍 Analyze Text",
                type="primary",
                use_container_width=True,
            )

        if analyze_button:

            clean_check = user_text.strip()

            if not clean_check:
                st.warning("Please enter some text before analyzing.")

            elif len(clean_check) > 10000:
                st.error(
                    "Your input is too long. "
                    "Please keep it under 10,000 characters."
                )

            else:
                with st.spinner("Analyzing text..."):

                    # EXISTING BACKEND CALLS — UNCHANGED
                    em_result = predict_emotion(
                        user_text,
                        em_model,
                        em_vec,
                    )

                    se_result = predict_sentiment(
                        user_text,
                        se_model,
                        se_vec,
                    )

                    detected_emotion = em_result["emotion"]
                    em_conf = em_result["confidence"]
                    em_probs = em_result["probabilities"]

                    detected_sentiment = se_result["sentiment"]
                    se_conf = se_result["confidence"]

                    # Keep the existing analytics history behaviour.
                    st.session_state.history.append(
                        {
                            "text": user_text,
                            "emotion": detected_emotion,
                            "sentiment": detected_sentiment,
                            "confidence": em_conf,
                        }
                    )

                st.markdown("---")
                st.markdown("### 🎯 Prediction Results")

                res_col1, res_col2 = st.columns([1.2, 0.8])

                # Emotion result card
                with res_col1:
                    cfg = EMOTION_CONFIG.get(
                        detected_emotion,
                        EMOTION_CONFIG["Neutral"],
                    )

                    st.markdown(
                        f"""
                        <div class="emotion-result-box"
                             style="background:{cfg['bg_gradient']};">
                            <div class="emotion-title-tag">
                                DETECTED EMOTION
                            </div>
                            <div class="emotion-emoji">{cfg['emoji']}</div>
                            <div class="emotion-name">
                                {detected_emotion.upper()}
                            </div>
                            <div class="emotion-conf">
                                Confidence: {em_conf:.1f}%
                            </div>
                            <div style="font-size:0.88rem;
                                        margin-top:0.7rem;
                                        opacity:0.95;">
                                {cfg['description']}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Sentiment result card
                with res_col2:
                    sent_cfg = SENTIMENT_CONFIG.get(
                        detected_sentiment,
                        SENTIMENT_CONFIG["Neutral"],
                    )

                    st.markdown(
                        f"""
                        <div class="sentiment-result-box">
                            <div style="font-size:0.8rem;
                                        font-weight:700;
                                        color:#64748b;
                                        letter-spacing:0.1em;">
                                SENTIMENT POLARITY
                            </div>
                            <div style="font-size:3rem;
                                        margin:0.5rem 0;">
                                {sent_cfg['emoji']}
                            </div>
                            <div style="font-size:1.8rem;
                                        font-weight:800;
                                        color:{sent_cfg['color']};">
                                {detected_sentiment}
                            </div>
                            <div style="font-size:1rem;
                                        font-weight:600;
                                        color:#475569;
                                        margin-top:0.6rem;">
                                Confidence: {se_conf:.1f}%
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Emotion probability chart
                st.markdown("### 📊 Emotion Probability Distribution")

                prob_df = pd.DataFrame(
                    [
                        {
                            "Emotion": emotion,
                            "Probability (%)": probability,
                            "Color": EMOTION_CONFIG.get(
                                emotion, {}
                            ).get("color", "#64748b"),
                        }
                        for emotion, probability in em_probs.items()
                    ]
                )

                prob_df = prob_df.sort_values(
                    by="Probability (%)",
                    ascending=True,
                )

                fig = go.Figure(
                    go.Bar(
                        x=prob_df["Probability (%)"],
                        y=prob_df["Emotion"],
                        orientation="h",
                        text=[
                            f"{p:.1f}%"
                            for p in prob_df["Probability (%)"]
                        ],
                        textposition="outside",
                        marker=dict(
                            color=prob_df["Color"],
                            line=dict(
                                color="rgba(0,0,0,0.1)",
                                width=1,
                            ),
                        ),
                        hovertemplate=(
                            "<b>%{y}</b><br>"
                            "Probability: %{x:.2f}%"
                            "<extra></extra>"
                        ),
                    )
                )

                fig.update_layout(
                    xaxis_title="Probability (%)",
                    yaxis_title="",
                    xaxis=dict(
                        range=[
                            0,
                            max(
                                100,
                                prob_df["Probability (%)"].max() + 15,
                            ),
                        ]
                    ),
                    height=360,
                    margin=dict(l=20, r=40, t=20, b=30),
                    template="plotly_white",
                    font=dict(family="Inter, sans-serif"),
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

                # Preprocessing details
                with st.expander("🔍 Inspect NLP Preprocessing Details"):
                    st.markdown(f"**Raw Input Text:** {user_text}")

                    st.markdown(
                        "**Cleaned & Lemmatized Text:** "
                        f"{em_result.get('cleaned_text', 'Not available')}"
                    )

                    st.caption(
                        "This section displays the cleaned text returned "
                        "by your existing prediction function."
                    )

# ---------------------------------------------------------------------
# 9. ANALYTICS DASHBOARD
# ---------------------------------------------------------------------

elif selected_page == "📈 Analytics":

    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-title">
                📈 Prediction Analytics Dashboard
            </div>
            <div class="hero-subtitle">
                Explore emotion distribution, sentiment polarity,
                confidence scores, and recent prediction history.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    history_records = st.session_state.history

    if not history_records:
        st.info(
            "No predictions have been analyzed yet. "
            "Visit Emotion Detection to get started."
        )

    else:
        df_hist = pd.DataFrame(history_records)

        # Summary metric cards
        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-number">{len(df_hist)}</div>
                    <div class="stat-label">Texts Analyzed</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with m2:
            top_em = df_hist["emotion"].mode()[0]

            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-number">
                        {EMOTION_CONFIG.get(top_em, {}).get('emoji', '')}
                    </div>
                    <div class="stat-label">
                        Most Common Emotion<br>{top_em}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with m3:
            top_se = df_hist["sentiment"].mode()[0]

            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-number">
                        {SENTIMENT_CONFIG.get(top_se, {}).get('emoji', '')}
                    </div>
                    <div class="stat-label">
                        Most Common Sentiment<br>{top_se}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with m4:
            avg_conf = df_hist["confidence"].mean()

            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-number">{avg_conf:.1f}%</div>
                    <div class="stat-label">Average Confidence</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("")

        # Emotion and sentiment charts
        chart_c1, chart_c2 = st.columns(2)

        with chart_c1:
            st.markdown("### 🎭 Emotion Distribution")

            em_counts = (
                df_hist["emotion"]
                .value_counts()
                .rename_axis("Emotion")
                .reset_index(name="Count")
            )

            fig_em = px.bar(
                em_counts,
                x="Emotion",
                y="Count",
                color="Emotion",
                text="Count",
                color_discrete_map={
                    emotion: cfg["color"]
                    for emotion, cfg in EMOTION_CONFIG.items()
                },
            )

            fig_em.update_layout(
                showlegend=False,
                height=340,
                margin=dict(l=20, r=20, t=25, b=20),
                template="plotly_white",
                font=dict(family="Inter, sans-serif"),
            )

            st.plotly_chart(
                fig_em,
                use_container_width=True,
            )

        with chart_c2:
            st.markdown("### 🧭 Sentiment Distribution")

            se_counts = (
                df_hist["sentiment"]
                .value_counts()
                .rename_axis("Sentiment")
                .reset_index(name="Count")
            )

            fig_se = px.pie(
                se_counts,
                names="Sentiment",
                values="Count",
                hole=0.48,
                color="Sentiment",
                color_discrete_map={
                    "Positive": "#10b981",
                    "Negative": "#ef4444",
                    "Neutral": "#64748b",
                },
            )

            fig_se.update_layout(
                height=340,
                margin=dict(l=20, r=20, t=25, b=20),
                template="plotly_white",
                font=dict(family="Inter, sans-serif"),
            )

            st.plotly_chart(
                fig_se,
                use_container_width=True,
            )

        # Confidence chart and history table
        chart_c3, chart_c4 = st.columns(2)

        with chart_c3:
            st.markdown("### 📊 Average Confidence by Emotion")

            conf_by_em = (
                df_hist.groupby("emotion")["confidence"]
                .mean()
                .reset_index()
            )

            conf_by_em.columns = [
                "Emotion",
                "Mean Confidence",
            ]

            fig_conf = px.bar(
                conf_by_em,
                x="Emotion",
                y="Mean Confidence",
                color="Emotion",
                text=conf_by_em["Mean Confidence"].apply(
                    lambda value: f"{value:.1f}%"
                ),
                color_discrete_map={
                    emotion: cfg["color"]
                    for emotion, cfg in EMOTION_CONFIG.items()
                },
            )

            fig_conf.update_layout(
                yaxis=dict(range=[0, 100]),
                showlegend=False,
                height=340,
                margin=dict(l=20, r=20, t=25, b=20),
                template="plotly_white",
                font=dict(family="Inter, sans-serif"),
            )

            st.plotly_chart(
                fig_conf,
                use_container_width=True,
            )

        with chart_c4:
            st.markdown("### 📝 Recent Prediction History")

            st.dataframe(
                df_hist[
                    [
                        "emotion",
                        "sentiment",
                        "confidence",
                        "text",
                    ]
                ].tail(10),
                use_container_width=True,
                height=340,
                hide_index=True,
            )

        # Optional download of displayed analytics history
        st.markdown("---")

        st.download_button(
            label="⬇️ Download Prediction History",
            data=df_hist.to_csv(index=False).encode("utf-8"),
            file_name="emotion_prediction_history.csv",
            mime="text/csv",
        )

# ---------------------------------------------------------------------
# END OF APPLICATION
# ---------------------------------------------------------------------
