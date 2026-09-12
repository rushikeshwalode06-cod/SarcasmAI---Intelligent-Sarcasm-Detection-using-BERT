import streamlit as st
import torch
import joblib
import numpy as np
import html
import re
from transformers import BertTokenizer, BertModel


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Sarcasm AI Studio",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# HTML RENDER FUNCTION
# ============================================================

def render_html(content):
    content = re.sub(r"\n\s*", " ", content).strip()
    st.markdown(content, unsafe_allow_html=True)


# ============================================================
# PREMIUM CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Poppins:wght@500;600;700;800&display=swap');


/* ============================================================
   GLOBAL
============================================================ */

* {
    font-family: 'Inter', sans-serif;
}

.stApp {

    background:
        radial-gradient(
            circle at 8% 8%,
            rgba(124,58,237,0.28),
            transparent 25%
        ),

        radial-gradient(
            circle at 92% 8%,
            rgba(6,182,212,0.20),
            transparent 25%
        ),

        radial-gradient(
            circle at 50% 100%,
            rgba(236,72,153,0.16),
            transparent 30%
        ),

        linear-gradient(
            135deg,
            #060a16,
            #0b1020,
            #070b18
        );

    color: white;
}


/* Hide Streamlit default */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}


.block-container {
    max-width: 1450px;
    padding-top: 35px;
    padding-bottom: 50px;
}


/* ============================================================
   HERO
============================================================ */

.hero {

    position: relative;

    overflow: hidden;

    padding: 50px 55px;

    border-radius: 32px;

    margin-bottom: 28px;

    background:

        linear-gradient(
            135deg,
            rgba(124,58,237,0.34),
            rgba(37,99,235,0.16),
            rgba(6,182,212,0.14),
            rgba(236,72,153,0.10)
        );

    border: 1px solid rgba(255,255,255,0.14);

    box-shadow:
        0 25px 80px rgba(0,0,0,0.40),
        inset 0 1px 0 rgba(255,255,255,0.08);

    backdrop-filter: blur(20px);
}


.hero::before {

    content: "";

    position: absolute;

    width: 280px;
    height: 280px;

    border-radius: 50%;

    background:
        rgba(124,58,237,0.20);

    filter: blur(60px);

    top: -150px;
    right: -50px;
}


.hero::after {

    content: "";

    position: absolute;

    width: 180px;
    height: 180px;

    border-radius: 50%;

    background:
        rgba(6,182,212,0.12);

    filter: blur(50px);

    bottom: -100px;
    left: 20%;
}


.hero-content {

    position: relative;

    z-index: 2;
}


.hero-title {

    font-family: 'Poppins', sans-serif;

    font-size: 52px;

    font-weight: 800;

    line-height: 1.15;

    letter-spacing: -1.5px;

    margin: 0;

    color: white;
}


.hero-highlight {

    background:
        linear-gradient(
            90deg,
            #c4b5fd,
            #67e8f9,
            #f0abfc
        );

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;
}


.hero-subtitle {

    color: #cbd5e1;

    font-size: 17px;

    line-height: 1.7;

    max-width: 850px;

    margin-top: 15px;
}


.badge-row {

    display: flex;

    flex-wrap: wrap;

    gap: 10px;

    margin-top: 25px;
}


.badge {

    padding: 9px 16px;

    border-radius: 999px;

    background:
        rgba(255,255,255,0.075);

    border:
        1px solid rgba(255,255,255,0.12);

    color: #e2e8f0;

    font-size: 12px;

    font-weight: 700;
}


/* ============================================================
   INPUT CARD
============================================================ */

.input-card {

    padding: 30px;

    border-radius: 26px;

    background:

        linear-gradient(
            145deg,
            rgba(255,255,255,0.085),
            rgba(255,255,255,0.025)
        );

    border:
        1px solid rgba(255,255,255,0.11);

    box-shadow:
        0 20px 55px rgba(0,0,0,0.28);

    backdrop-filter: blur(18px);

    height: 100%;
}


.section-title {

    font-family: 'Poppins', sans-serif;

    color: white;

    font-size: 22px;

    font-weight: 700;

    margin-bottom: 7px;
}


.section-subtitle {

    color: #94a3b8;

    font-size: 14px;

    line-height: 1.6;

    margin-bottom: 20px;
}


/* ============================================================
   TEXT AREA
   ONLY WIDTH CHANGED HERE
============================================================ */

div[data-testid="stTextArea"] {

    width: 88% !important;

}


div[data-testid="stTextArea"] textarea {

    background:
        rgba(2,6,23,0.75) !important;

    color: #f8fafc !important;

    border:
        1px solid rgba(255,255,255,0.14) !important;

    border-radius:
        18px !important;

    font-size:
        16px !important;

    line-height:
        1.7 !important;

    padding:
        18px !important;

    width: 100% !important;

    resize: vertical !important;
}


div[data-testid="stTextArea"] textarea:focus {

    border:
        1px solid #8b5cf6 !important;

    box-shadow:
        0 0 0 2px rgba(139,92,246,0.16) !important;
}


/* ============================================================
   BUTTON
============================================================ */

div.stButton > button {

    width: 88%;

    min-height: 56px;

    border: none !important;

    border-radius: 16px !important;

    color: white !important;

    font-size: 16px !important;

    font-weight: 800 !important;

    letter-spacing: 0.3px;

    background:

        linear-gradient(
            135deg,
            #7c3aed,
            #2563eb,
            #06b6d4
        ) !important;

    box-shadow:
        0 12px 35px
        rgba(37,99,235,0.28);

    transition:
        all 0.25s ease;
}


div.stButton > button:hover {

    transform:
        translateY(-3px);

    box-shadow:
        0 18px 45px
        rgba(124,58,237,0.40);
}


/* ============================================================
   FEATURE CHIPS
============================================================ */

.feature-row {

    display: flex;

    flex-wrap: wrap;

    gap: 9px;

    margin-top: 18px;
}


.feature {

    padding:
        8px 13px;

    border-radius:
        999px;

    background:
        rgba(255,255,255,0.045);

    border:
        1px solid rgba(255,255,255,0.08);

    color:
        #cbd5e1;

    font-size:
        11px;

    font-weight:
        600;
}


/* ============================================================
   HOW IT WORKS
============================================================ */

.how-card {

    padding: 30px;

    border-radius: 26px;

    background:

        linear-gradient(
            145deg,
            rgba(255,255,255,0.075),
            rgba(255,255,255,0.025)
        );

    border:
        1px solid rgba(255,255,255,0.10);

    box-shadow:
        0 20px 55px
        rgba(0,0,0,0.25);

    height: 100%;
}


.step {

    display: flex;

    align-items: flex-start;

    gap: 14px;

    padding: 15px;

    margin-top: 13px;

    border-radius: 16px;

    background:
        rgba(255,255,255,0.045);

    border:
        1px solid rgba(255,255,255,0.07);
}


.step-number {

    min-width: 34px;

    height: 34px;

    display: flex;

    align-items: center;

    justify-content: center;

    border-radius: 10px;

    background:
        linear-gradient(
            135deg,
            #7c3aed,
            #2563eb
        );

    color: white;

    font-size: 12px;

    font-weight: 800;
}


.step-content {

    color: #cbd5e1;

    font-size: 13px;

    line-height: 1.55;
}


.step-content b {

    color: white;

    font-size: 14px;
}


/* ============================================================
   RESULT
============================================================ */

.result-card {

    margin-top: 30px;

    padding: 34px;

    border-radius: 28px;

    background:

        linear-gradient(
            135deg,
            rgba(124,58,237,0.24),
            rgba(37,99,235,0.12),
            rgba(6,182,212,0.09)
        );

    border:
        1px solid rgba(255,255,255,0.15);

    box-shadow:
        0 25px 70px
        rgba(0,0,0,0.35);
}


.result-label {

    color: #a5b4fc;

    font-size: 12px;

    font-weight: 800;

    text-transform: uppercase;

    letter-spacing: 1.5px;
}


.result-value {

    font-family: 'Poppins', sans-serif;

    color: white;

    font-size: 42px;

    font-weight: 800;

    margin-top: 7px;
}


.result-description {

    color: #cbd5e1;

    font-size: 15px;

    line-height: 1.6;

    margin-top: 5px;
}


/* ============================================================
   ENTERED TEXT
============================================================ */

.input-preview {

    margin-top: 24px;

    padding: 21px;

    border-radius: 18px;

    background:
        rgba(2,6,23,0.68);

    border:
        1px solid rgba(255,255,255,0.09);

    color: #e2e8f0;

    font-size: 15px;

    line-height: 1.7;

    word-break: break-word;
}


.input-title {

    color: #67e8f9;

    font-size: 12px;

    font-weight: 800;

    letter-spacing: 0.8px;

    margin-bottom: 9px;
}


/* ============================================================
   HAPPY MOMENT
============================================================ */

.happy-moment {

    margin-top: 20px;

    padding: 25px;

    border-radius: 22px;

    text-align: center;

    background:

        linear-gradient(
            135deg,
            rgba(236,72,153,0.20),
            rgba(124,58,237,0.20),
            rgba(6,182,212,0.12)
        );

    border:
        1px solid rgba(255,255,255,0.14);

    box-shadow:
        0 15px 45px
        rgba(124,58,237,0.18);

    animation:
        happyPulse 2s ease-in-out infinite;
}


.happy-icon {

    font-size: 42px;

    margin-bottom: 5px;
}


.happy-title {

    font-family: 'Poppins', sans-serif;

    font-size: 26px;

    font-weight: 800;

    color: white;
}


.happy-text {

    color: #cbd5e1;

    font-size: 14px;

    margin-top: 8px;

    line-height: 1.6;
}


@keyframes happyPulse {

    0% {
        transform: scale(1);
    }

    50% {
        transform: scale(1.015);
    }

    100% {
        transform: scale(1);
    }

}


/* ============================================================
   NORMAL MOMENT
============================================================ */

.normal-moment {

    margin-top: 20px;

    padding: 23px;

    border-radius: 20px;

    text-align: center;

    background:
        rgba(255,255,255,0.045);

    border:
        1px solid rgba(255,255,255,0.08);
}


.normal-icon {

    font-size: 34px;

    margin-bottom: 5px;
}


.normal-title {

    font-family: 'Poppins', sans-serif;

    font-size: 23px;

    font-weight: 700;

    color: #e2e8f0;
}


.normal-text {

    color: #94a3b8;

    font-size: 14px;

    margin-top: 7px;

    line-height: 1.6;
}


/* ============================================================
   CONFIDENCE
============================================================ */

.confidence-card {

    margin-top: 18px;

    padding: 22px;

    border-radius: 20px;

    background:
        rgba(255,255,255,0.045);

    border:
        1px solid rgba(255,255,255,0.08);
}


.confidence-title {

    color: #94a3b8;

    font-size: 12px;

    font-weight: 700;
}


.confidence-value {

    color: #67e8f9;

    font-size: 31px;

    font-weight: 800;

    margin-top: 4px;
}


/* ============================================================
   FOOTER
============================================================ */

.custom-footer {

    text-align: center;

    color: #64748b;

    margin-top: 50px;

    padding: 25px;

    font-size: 13px;
}


.footer-brand {

    color: #a78bfa;

    font-weight: 800;
}


/* ============================================================
   MOBILE
============================================================ */

@media (max-width: 768px) {

    .hero {
        padding: 35px 25px;
    }

    .hero-title {
        font-size: 38px;
    }

    .hero-subtitle {
        font-size: 14px;
    }

    div[data-testid="stTextArea"] {
        width: 100% !important;
    }

    div.stButton > button {
        width: 100%;
    }

}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HERO
# ============================================================

render_html("""
<div class="hero">

<div class="hero-content">

<div class="hero-title">
🎭 Sarcasm <span class="hero-highlight">AI Studio</span>
</div>

<div class="hero-subtitle">
An intelligent NLP application that detects sarcasm from
text using <b>BERT embeddings</b> and a
<b>Logistic Regression classifier</b>.
</div>

<div class="badge-row">

<div class="badge">🤖 AI POWERED</div>

<div class="badge">🧠 BERT NLP</div>

<div class="badge">📊 MACHINE LEARNING</div>

<div class="badge">⚡ FAST ANALYSIS</div>

</div>

</div>

</div>
""")


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource(show_spinner=False)
def load_models():

    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased"
    )

    bert_model = BertModel.from_pretrained(
        "bert-base-uncased"
    )

    bert_model.eval()

    clf = joblib.load(
        "sarcasm_classifier.pkl"
    )

    return tokenizer, bert_model, clf


try:

    tokenizer, bert_model, clf = load_models()

except Exception as e:

    st.error("❌ Model loading failed.")

    st.code(str(e))

    st.stop()


# ============================================================
# BERT EMBEDDING
# ============================================================

def get_embedding(text):

    encoded = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = bert_model(
            **encoded
        )

    embedding = outputs.last_hidden_state[
        :, 0, :
    ].numpy()

    return embedding


# ============================================================
# MAIN LAYOUT
# ============================================================

left, right = st.columns(
    [1.15, 0.85],
    gap="large"
)


# ============================================================
# INPUT SECTION
# ============================================================

with left:

    render_html("""
    <div class="input-card">

    <div class="section-title">
    ✍️ Enter your text
    </div>

    <div class="section-subtitle">
    Enter a sentence, headline, comment or statement
    and let the AI determine whether it contains sarcasm.
    </div>

    </div>
    """)


    text = st.text_area(
        "Text",

        value=st.session_state.get(
            "text_input",
            ""
        ),

        height=210,

        placeholder=
        "Example: Oh fantastic, my computer crashed again...",

        label_visibility="collapsed"
    )


    st.session_state["text_input"] = text


    render_html("""
    <div class="feature-row">

    <div class="feature">
    🧠 BERT Embeddings
    </div>

    <div class="feature">
    📊 Logistic Regression
    </div>

    <div class="feature">
    🎯 Binary Classification
    </div>

    <div class="feature">
    ⚡ Fast Prediction
    </div>

    </div>
    """)


    st.write("")


    predict = st.button(
        "🚀 ANALYZE TEXT",
        type="primary"
    )


# ============================================================
# HOW IT WORKS
# ============================================================

with right:

    render_html("""
    <div class="how-card">

    <div class="section-title">
    ⚙️ How it works
    </div>

    <div class="section-subtitle">
    Your text goes through three simple AI stages.
    </div>


    <div class="step">

    <div class="step-number">
    01
    </div>

    <div class="step-content">
    <b>Tokenization</b><br>
    Your text is converted into tokens that BERT can process.
    </div>

    </div>


    <div class="step">

    <div class="step-number">
    02
    </div>

    <div class="step-content">
    <b>BERT Embedding</b><br>
    BERT converts the sentence into meaningful contextual features.
    </div>

    </div>


    <div class="step">

    <div class="step-number">
    03
    </div>

    <div class="step-content">
    <b>AI Classification</b><br>
    Logistic Regression predicts the final sarcasm class.
    </div>

    </div>

    </div>
    """)


# ============================================================
# PREDICTION
# ============================================================

if predict:

    if not text.strip():

        st.warning(
            "⚠️ Please enter some text before analyzing."
        )

    else:

        with st.spinner(
            "🤖 AI is analyzing your text..."
        ):

            try:

                # =================================================
                # BERT EMBEDDING
                # =================================================

                embedding = get_embedding(
                    text
                )


                # =================================================
                # MODEL PREDICTION
                # =================================================

                pred = clf.predict(
                    embedding
                )[0]


                # =================================================
                # MODEL CONFIDENCE
                # =================================================

                probability = None

                if hasattr(
                    clf,
                    "predict_proba"
                ):

                    probabilities = clf.predict_proba(
                        embedding
                    )[0]

                    probability = float(
                        np.max(probabilities)
                    )


                # =================================================
                # RESULT
                # =================================================

                if pred == 1:

                    label = "😏 SARCASTIC"

                    description = (
                        "The AI detected a sarcastic tone "
                        "in the entered text."
                    )


                    # =============================================
                    # HAPPY MOMENT
                    # =============================================

                    moment_html = """
                    <div class="happy-moment">

                    <div class="happy-icon">
                    🎉😂✨
                    </div>

                    <div class="happy-title">
                    Haha! Sarcasm Detected! 😏
                    </div>

                    <div class="happy-text">
                    Looks like someone just added a little
                    sarcasm to the conversation! 🎭
                    <br>
                    Time for a happy moment! 🥳
                    </div>

                    </div>
                    """


                    # Balloons
                    st.balloons()


                else:

                    label = "🙂 NOT SARCASTIC"

                    description = (
                        "The AI did not detect a sarcastic "
                        "tone in the entered text."
                    )


                    # =============================================
                    # NORMAL MOMENT
                    # =============================================

                    moment_html = """
                    <div class="normal-moment">

                    <div class="normal-icon">
                    🌿🙂✨
                    </div>

                    <div class="normal-title">
                    Calm & Normal Moment
                    </div>

                    <div class="normal-text">
                    No sarcasm detected.
                    Everything looks straightforward and normal! 😊
                    </div>

                    </div>
                    """


                # =================================================
                # SAFE USER TEXT
                # =================================================

                safe_text = html.escape(
                    text
                )


                # =================================================
                # RESULT CARD
                # =================================================

                render_html(f"""
                <div class="result-card">

                <div class="result-label">
                🤖 AI PREDICTION
                </div>

                <div class="result-value">
                {label}
                </div>

                <div class="result-description">
                {description}
                </div>


                <div class="input-preview">

                <div class="input-title">
                📝 YOUR ENTERED TEXT
                </div>

                {safe_text}

                </div>

                </div>
                """)


                # =================================================
                # HAPPY / NORMAL MOMENT
                # =================================================

                render_html(
                    moment_html
                )


                # =================================================
                # CONFIDENCE
                # =================================================

                if probability is not None:

                    percentage = (
                        probability * 100
                    )


                    render_html(f"""
                    <div class="confidence-card">

                    <div class="confidence-title">
                    🎯 MODEL CONFIDENCE
                    </div>

                    <div class="confidence-value">
                    {percentage:.2f}%
                    </div>

                    </div>
                    """)


                    st.progress(
                        probability
                    )


                st.success(
                    "✨ Analysis completed successfully!"
                )


            except Exception as e:

                st.error(
                    "❌ Prediction failed."
                )

                st.code(
                    str(e)
                )


# ============================================================
# FOOTER
# ============================================================

render_html("""
<div class="custom-footer">

🎭 <span class="footer-brand">
Sarcasm AI Studio
</span>

&nbsp; • &nbsp;

Powered by BERT + Logistic Regression

<br><br>

AI-powered NLP • Built with ❤️ using Streamlit

</div>
""")