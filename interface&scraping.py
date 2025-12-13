%%writefile app.py
import streamlit as st
import joblib
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
import pandas as pd
from urllib.parse import urlparse
from scipy.sparse import hstack
from scipy.special import expit   # sigmoid

# Load trained artifacts
model = joblib.load("final_fake_job_detector.pkl")
tfidf = joblib.load("final_tfidf_vectorizer.pkl")
scaler = joblib.load("final_numeric_scaler.pkl")
numeric_feature_names = joblib.load("final_numeric_feature_names.pkl")
suspicious_keywords = joblib.load("final_suspicious_keywords.pkl")

# Helpers — preprocessing
def clean_text(text):
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text.lower()).strip()

# Numeric feature extraction
def extract_numeric_features(text):
    text_low = text.lower()
    features = {
        'char_len': len(text),
        'word_count': len(text.split()),
        'avg_word_len': (sum(len(w) for w in text.split()) / (len(text.split()) or 1)),
        'num_urls': len(re.findall(r'http\S+', text)),
        'num_emails': len(re.findall(r'\S+@\S+', text)),
        'num_currency': len(re.findall(r'[\$\€\£₹]', text)),
        'num_digits': sum(c.isdigit() for c in text),
        'num_phone_like': 1 if re.search(r'\b\d{6,}\b', text) else 0,
        'suspicious_kw_count': sum(1 for kw in suspicious_keywords if kw in text_low),
        'unique_char_frac': len(set(text_low)) / (len(text_low) + 1)
    }
    return [features[f] for f in numeric_feature_names]

# scraping function 
def fetch_job_from_url(url):
    import socket
    from requests.exceptions import RequestException
    from selenium.common.exceptions import WebDriverException, TimeoutException
    try:
        url = url.strip()
        if not url.startswith("http"):
            url = "http://" + url

        hostname = urlparse(url).hostname
        try:
            socket.gethostbyname(hostname)
        except Exception as e:
            return None, f"DNS lookup failed for host '{hostname}': {e}"

        try:
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                text = " ".join([p.get_text(separator=" ") for p in soup.find_all(
                    ["p","div","span","h1","h2","h3","li"] )])
                text = clean_text(text)
                if len(text) > 80:
                    return text, None
        except RequestException:
            pass

        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            import time

            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")

            driver = webdriver.Chrome(options=chrome_options)
            driver.set_page_load_timeout(20)
            try:
                driver.get(url)
            except TimeoutException:
                try:
                    driver.get(url)
                except Exception:
                    driver.quit()
                    return None, "Page load timed out."

            time.sleep(2)
            html = driver.page_source
            driver.quit()

            soup = BeautifulSoup(html, "html.parser")
            text = " ".join([p.get_text(separator=" ") for p in soup.find_all(
                ["p","div","span","h1","h2","h3","li"] )])
            text = clean_text(text)
            if len(text) < 30:
                return None, "Fetched page but extracted text is too short."
            return text, None

        except WebDriverException as e:
            msg = str(e)
            if "ERR_NAME_NOT_RESOLVED" in msg:
                return None, f"Chrome DNS error: {hostname}"
            return None, f"Headless browser error: {msg}"

    except Exception as e:
        return None, f"Unexpected error: {e}"

# Explainability — linear SVM coefficients
def explain_prediction(text, tfidf, model, top_k=6):
    cleaned = clean_text(text)
    words = cleaned.split()

    vocab = {w: idx for idx, w in enumerate(tfidf.get_feature_names_out())}
    coef = model.coef_[0]

    scores = {}
    for w in set(words):
        if w in vocab:
            scores[w] = float(coef[vocab[w]])

    if not scores:
        return [], []

    top_fake = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    top_real = sorted(scores.items(), key=lambda x: x[1])[:top_k]

    return top_fake, top_real

# Predict probability wrapper
def safe_predict_proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    score = model.decision_function(X)
    score = np.array(score).reshape(-1)
    return expit(score)

# Prediction pipeline
def predict_job(text):
    cleaned = clean_text(text)

    X_tfidf = tfidf.transform([cleaned])
    X_num = np.array([extract_numeric_features(text)])
    X_num_scaled = scaler.transform(X_num)

    X_final = hstack([X_tfidf, X_num_scaled])

    model_prob = float(safe_predict_proba(model, X_final)[0])
    rule_score = sum(0.05 for kw in suspicious_keywords if kw in cleaned)

    combined = min(1.0, model_prob + rule_score)

    if combined > 0.8:
        verdict = "🚨 This job posting seems FAKE."
    elif combined > 0.5:
        verdict = "⚠️ Possibly fake — verify with official sources."
    else:
        verdict = "✅ This job posting seems REAL."

    return {
    "combined_prob": combined,
    "model_prob": model_prob,
    "rule_score": rule_score,
    "verdict": verdict
}

# Streamlit UI 
import streamlit as st
from streamlit.components.v1 import html
import matplotlib.pyplot as plt
from math import pi
st.set_page_config(page_title="Fake Job Posting Detector", page_icon="🕵️‍♂️", layout="centered")

# Glassmorphic background
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f1724 0%, #1e293b 100%);
        color: #e0e0e0;
        font-family: 'Segoe UI', sans-serif;
    }
    .glass-card {
        background: rgba(255,255,255,0.05);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 255, 255, 0.2);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 20px;
    }
    .neon-metric {
        color: #00ffff;
        font-weight: bold;
        font-size: 1.3em;
    }
    .risk-badge {
        display: inline-block;
        padding: 0.2em 0.7em;
        border-radius: 12px;
        font-weight: bold;
        color: #fff;
        margin-right: 10px;
    }
    .high {background-color:#ff4d4d;}
    .moderate {background-color:#facc15;}
    .safe {background-color:#22c55e;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🕵️‍♂️ Fake Job Posting Detector")
st.markdown("Choose how you want to analyze a job posting:")

# 2 modes
mode = st.radio(
    "Select input mode",
    ["🔤 Manual Text Input", "🌐 URL-based Detection"],
    index=0,
    horizontal=True
)

text_to_use = None

# MODE 1 — MANUAL TEXT INPUT
if mode == "🔤 Manual Text Input":
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        manual_text = st.text_area("Paste the full job description here", height=250)
        if st.button("Analyze Text"):
            if not manual_text.strip():
                st.error("Please paste a job description first.")
            else:
                text_to_use = manual_text.strip()
        st.markdown('</div>', unsafe_allow_html=True)

# MODE 2 — URL-BASED EXTRACTION
if mode == "🌐 URL-based Detection":
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        url = st.text_input("Paste the job posting URL here")
        if st.button("Fetch & Analyze URL"):
            if not url.strip():
                st.error("Please enter a valid URL.")
            else:
                with st.spinner("Fetching job posting from URL..."):
                    scraped, err = fetch_job_from_url(url)

                if scraped:
                    text_to_use = scraped
                    st.success("Successfully extracted text from URL!")
                    with st.expander("Show extracted text (first 5000 chars)"):
                        st.write(scraped[:5000])
                else:
                    st.error(f"❌ Failed to fetch URL: {err}")
                    st.info("You can switch to **Manual Mode** and paste the text manually.")
        st.markdown('</div>', unsafe_allow_html=True)

# RUN PREDICTION IF WE HAVE TEXT
if text_to_use:
    with st.spinner("Analyzing posting..."):
        res = predict_job(text_to_use)

    combined_prob = res["combined_prob"]
    model_prob = res["model_prob"]
    rule_score = res["rule_score"]
    verdict = res["verdict"]

    # Verdict Neon Badge
    risk_class = "safe"
    if combined_prob > 0.8:
        risk_class = "high"
    elif combined_prob > 0.5:
        risk_class = "moderate"

    st.markdown(f"""
    <div class="glass-card">
    <h2 style="color:#00ffff;">{verdict}</h2>
    <div class="neon-metric">Fake probability: {round(combined_prob*100,2)}%</div>
    <div style="margin-top:10px;">
        <span class="risk-badge {risk_class}">{risk_class.upper()}</span>
        Model prob: {model_prob:.3f} • Rule score: {rule_score:.3f}
    </div>
    </div>
    """, unsafe_allow_html=True)


    # Compute radar chart dimensions (heuristic proxies)
    def compute_radar_scores(text, model_prob, rule_score, url_present=False):
        t = text.lower()
        # 1. ML Model Score (0..1)
        ml_score = float(model_prob)

        # 2. Suspicious Keyword Score 
        suspicious_score = min(1.0, float(rule_score))

        # 3. URL Trust Score (0..1) 
        # If URL present, penalize suspicious TLD patterns or numeric in domain
        url_score = 0.5
        if url_present:
            try:
                domain = urlparse(text).netloc.lower()
                if domain == "":
                    url_score = 0.3
                elif any(x in domain for x in ["-", "cheap", "jobs-", "apply-", "free", "earn", "offer"]):
                    url_score = 0.25
                elif any(x in domain for x in [".gov", ".edu", ".org"]):
                    url_score = 0.9
                else:
                    # presence of many digits -> suspicious
                    digits = sum(c.isdigit() for c in domain)
                    url_score = 0.4 if digits > 2 else 0.6
            except Exception:
                url_score = 0.5

        # 4. Salary Sanity Score (0..1) 
        salary_sus = 0.0
        if any(k in t for k in ["per day", "daily", "instant", "join bonus", "join now"]):
            salary_sus = 0.2
        if any(sym in t for sym in ["₹", "$", "€"]) and any(k in t for k in ["per day","daily"]):
            salary_sus = 0.1
        # default sanity = 1 - salary_sus 
        salary_score = 1.0 - salary_sus

        # 5. Grammar / Fluency Score (0..1) 
        words = [w for w in t.split() if w.isalpha()]
        avg_wlen = (sum(len(w) for w in words) / len(words)) if words else 0
        # short gibberish words reduce fluency
        grammar_score = min(1.0, max(0.2, (avg_wlen - 3.0) / 5.0 + 0.5))  # maps typical 3-8 to 0-1

        # 6. Company Authenticity Score (0..1)
        org_signals = sum(1 for k in ["team","manager","department","engineer","developer","hr","office","head"] if k in t)
        scam_signals = sum(1 for k in ["whatsapp","telegram","bank","registration fee","click here","apply now","earn"] if k in t)
        auth_score = min(1.0, max(0.0, (org_signals * 0.6 - scam_signals * 0.6 + 1.0) / 2.0))

        # Return scaled values 
        return {
            "ML Model Score": ml_score,
            "Suspicious Keyword Trust": 1.0 - suspicious_score,
            "URL Trust": url_score,
            "Salary Sanity": salary_score,
            "Grammar / Fluency": grammar_score,
            "Company Authenticity": auth_score
        }

    # Detect whether input was URL (simple heuristic)
    # Radar scores
    url_present_flag = text_to_use.startswith("http") or text_to_use.startswith("www.")
    radar_scores = compute_radar_scores(text_to_use, model_prob, rule_score, url_present_flag)
    labels = list(radar_scores.keys())
    values = list(radar_scores.values())


    # Radar / Spider chart using matplotlib
    def plot_radar(labels, values, style="dark"):
        N = len(labels)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        vals = values[:]
        vals += vals[:1]

        plt.figure(figsize=(5,5))
        ax = plt.subplot(111, polar=True)
        if style == "dark":
            plt.style.use('dark_background')
            facecolor = "#0f1724"
            line_color = "#ff4d4d"
            fill_color = "#ff4d4d33"
        else:
            # gradient/clean
            plt.style.use('default')
            facecolor = "#ffffff"
            line_color = "#3b82f6"
            fill_color = "#3b82f633"

        ax.set_facecolor(facecolor)
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)

        plt.xticks(angles[:-1], labels, color="white" if style=="dark" else "black", size=10)
        ax.set_rlabel_position(0)
        max_val = 1.0
        yticks = [0.2,0.4,0.6,0.8,1.0]
        ax.set_yticks(yticks)
        ax.set_ylim(0, max_val)
        if style == "dark":
            ax.tick_params(colors="white")
            for y in yticks:
                ax.yaxis.label.set_color("white")
        else:
            ax.tick_params(colors="black")

        ax.plot(angles, vals, color=line_color, linewidth=2, linestyle='solid')
        ax.fill(angles, vals, color=fill_color, alpha=0.6)
        plt.tight_layout()
        return plt

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.write("### 🔐 Fraud Risk Radar")
    style = "dark"
    plt_radar = plot_radar(labels, values, style="dark")  # call function and get plt object
    st.pyplot(plt_radar)
    st.markdown('</div>', unsafe_allow_html=True)

    # Breakdown
    st.markdown(f"""
    <div class="glass-card">
    <h4>🔹 Detailed Breakdown</h4>
    <ul>
        <li>ML model probability: {model_prob:.3f}</li>
        <li>Rule-based boost: {rule_score:.3f}</li>
        <li>Final combined fraud probability: {combined_prob:.3f}</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

