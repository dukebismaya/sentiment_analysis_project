# ==========================================================
# Copyright (c) Bismaya Jyoti Dalei 2025 All rights reserved
# ==========================================================

import streamlit as st
import joblib
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ---- Page Config ----
st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬", layout="centered")

st.markdown(
    """
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            padding: 0.5rem 1rem;
            text-align: center;
            background: rgba(17, 17, 17, 0.9);
        }
        main .block-container {
            padding-bottom: 4rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Load Model & Vectorizer ----
@st.cache_resource
def load_model():
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_model()

# ---- App Header ----
st.title("💬 Sentiment Analysis App By Bismaya")
st.markdown("This app analyzes the **sentiment** of your text using a machine learning model trained on IMDB reviews.")

# ---- Sidebar ----
st.sidebar.header(u"\u2699 Settings")
show_wordcloud = st.sidebar.checkbox("Show wordcloud of your text", value=False)
confidence_meter = st.sidebar.checkbox("Show confidence meter", value=True)

# ---- User Input ----
user_input = st.text_area("📝 Enter your review or text below:", height=150)

if st.button("🔍 Analyze Sentiment"):
    if user_input.strip():
        # Transform input
        X = vectorizer.transform([user_input])
        
        # Pedict and get probability
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X)[0][1]
        else:
            # For models without predict_proba (e.g., LinearSVC)
            prob = model.decision_function(X)
            prob = 1/ (1 + np.exp(-prob)) # sigmoid approx
            
        sentiment = "😊 Positive" if prob >= 0.5 else "😞 Negative"
        conf_percent = prob*100 if prob >= 0.5 else (1-prob) * 100
        
        # ---- Display Result ----
        st.subheader(f"Prediction: {sentiment}")
        st.write(f"Confidence: **{conf_percent:.2f}%**")
        
        # ---- Confidence Meter ----
        if confidence_meter:
            st.progress(int(conf_percent))
            
        # ---- Word Cloud ----
        if show_wordcloud:
            wc = WordCloud(width=800, height=400, background_color='white').generate(user_input)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            ax.set_title("Word Cloud of Input Text")
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.warning(u"\u26A0 Please enter some text to analyze.")
            
            
# ---- Footer ----
st.markdown(
    """
    <div class="footer">
        <small>Built using Streamlit &amp; Scikit-learn | &copy; Bismaya Jyoti Dalei</small>
    </div>
    """,
    unsafe_allow_html=True,
)