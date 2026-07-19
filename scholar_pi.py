import os
import sqlite3
import json
import hashlib
from datetime import datetime, timedelta
from decimal import Decimal, getcontext
import requests
import streamlit as st
import fitz  # PyMuPDF
from groq import Groq

# --- 1. CONFIGURATION & ENVIRONMENT ---
st.set_page_config(page_title="Scholarπ Paper Evaluator", page_icon="🎓", layout="wide")

MODEL_NAME = "llama-3.3-70b-versatile"
SEED_NUMBER = 42
MAX_TEXT_TOKENS_FOR_LLM = 4000

# Set up local directory storage
BASE_DIR = os.path.abspath('./ScholarPi_System_Cloud')
os.makedirs(BASE_DIR, exist_ok=True)

DB_PATH = os.path.join(BASE_DIR, 'scholar_pi_hashed.db')
CRITERIA_PATH = os.path.join(BASE_DIR, 'criteria.txt')

# --- 2. API & PERSISTENT DATABASE SETUP ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    st.error("⚠️ Groq API Key not found! Please add it to your Streamlit Advanced Settings (Secrets).")
    st.stop()
    
client = Groq(api_key=GROQ_API_KEY)

@st.cache_resource
def init_system():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute('''CREATE TABLE IF NOT EXISTS papers 
                    (file_hash TEXT PRIMARY KEY, filename TEXT, pi_index REAL, justifications TEXT, timestamp TEXT)''')
    conn.commit()

    criteria_content = """The Core Criteria (S1–S13)
S1: CharDensity – Measures the depth and complexity of the text.
S2: NumDensity – Evaluates the presence of hard data, statistics, and empirical measurements.
S3: Reasoning – Assesses the strength of the arguments and the analytical deductions.
S4: CitationIntegration – Quality of references and academic linkage within the text.
S4b: CitationVolume – Quantitative score based on the total citation count of the paper (higher is better).
S5: AuthorDiversity – Evaluates the collaborative spread of the authors.
S6: Expertise – Gauges the domain knowledge demonstrated.
S7: Novelty – Assessment of whether the findings are a new contribution.
S8: Suggestions – Checks for actionable future research directions.
S9: Fees – Identifies transparency regarding funding/grants.
S10: Recency – Timeliness of the topic and references.
S11: FieldDiversity – Interdisciplinary nature of the research.
S12: Validation – Rigor of methodology and claims.
S13: LogicalCoherence – Flow, structure, and readability.

The External Discovery Metrics (S14–S15)
S14: WebGroundedUniqueness – Objective score on how pioneering the topic is globally.
S15: AuthorHIndex – Quantitative score based on the lead/senior author's cumulative H-Index."""

    with open(CRITERIA_PATH, 'w') as f:
        f.write(criteria_content)
    return conn

conn = init_system()

# --- 3. CORE PROCESSING FUNCTIONS ---
def get_file_hash_from_bytes(file_bytes):
    return hashlib.sha256(file_bytes).hexdigest()

def calculate_pi_index(base_scores, uniqueness_score_10pt, delta_t=0):
    getcontext().prec = 10
    pi = Decimal('3.1415926535')
    avg_score = sum(base_scores) / len(base_scores) 
    u_score = uniqueness_score_10pt / 10.0 
    
    drift = (pi / Decimal('3.14')) ** Decimal(str(delta_t))
    u_multiplier = Decimal('0.5') + (Decimal(str(u_score)) * Decimal('0.5'))
    return float(Decimal(str(avg_score)) * drift * u_multiplier)

def process_paper(file_bytes, filename):
    file_hash = get_file_hash_from_bytes(file_bytes)
    
    # Cache Check
    cursor = conn.cursor()
    cursor.execute("SELECT pi_index, justifications, timestamp FROM papers WHERE file_hash=?", (file_hash,))
    cached_row = cursor.fetchone()

    if cached_row:
        cached_pi, cached_justifications, cached_time_str = cached_row
        cached_time = datetime.fromisoformat(cached_time_str)
        if datetime.now() - cached_time < timedelta(days=30):
            return cached_pi, json.loads(cached_justifications), True

    # Extract Text
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = " ".join([page.get_text() for page in doc])

    # Keyword Extraction using Groq Cloud
    kw_response = client.chat.completions.create(
        messages=[{"role": "user", "content": f"Extract the 5 most critical research keywords. Return JSON: {{'keywords': []}}. Text: {text[:MAX_TEXT_TOKENS_FOR_LLM]}"}],
        model=MODEL_NAME, temperature=0, seed=SEED_NUMBER, response_format={"type": "json_object"}
    )
    keywords = json.loads(kw_response.choices[0].message.content).get('keywords', [])
    
    # Semantic Scholar Sweep
    query = " ".join(keywords)
    try:
        res = requests.get("https://api.semanticscholar.org/graph/v1/paper/search", params={"query": query, "limit": 100, "fields": "title,year"})
        data = res.json().get('data', [])
        if data:
            search_results = f"Total related papers found in top sweep: {len(data)}\n" + "\n".join([f"- {item['title']} ({item.get('year', 'N/A')})" for item in data])
        else:
            search_results = "No external data found. This topic appears completely pioneering."
    except:
        search_results = "No external data found due to API error."

    # Unified Evaluation using Groq Cloud
    with open(CRITERIA_PATH, 'r') as f: criteria = f.read()
    eval_prompt = f"""Evaluate the paper based on these criteria: {criteria}. 
    For S14 (WebGroundedUniqueness), gauge how saturated the topic is by reviewing this list of similar research:
    {search_results}
    Return ONLY a JSON object with keys S1-S13, S4b, S14, and S15. Each key must contain a nested object with 'score' (a number 0-10) and 'reason' (a concise 1-2 sentence explanation).
    Paper Text: {text[:MAX_TEXT_TOKENS_FOR_LLM]}"""
    
    scores_json = client.chat.completions.create(messages=[{"role": "user", "content": eval_prompt}], model=MODEL_NAME, temperature=0, seed=SEED_NUMBER, response_format={"type": "json_object"}).choices[0].message.content
    scores_data = json.loads(scores_json)
    
    # Analytics
    score_keys = ['S1', 'S2', 'S3', 'S4', 'S4b', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S15']
    base_score_list = [float(scores_data.get(k, {}).get('score', 5.0)) for k in score_keys]
    uniqueness_s14 = float(scores_data.get('S14', {}).get('score', 5.0))
    
    pi = calculate_pi_index(base_score_list, uniqueness_s14)
    
    conn.execute("INSERT OR REPLACE INTO papers (file_hash, filename, pi_index, justifications, timestamp) VALUES (?,?,?,?,?)",
                 (file_hash, filename, pi, json.dumps(scores_data), datetime.now().isoformat()))
    conn.commit()
    
    return pi, scores_data, False

# --- 4. STREAMLIT WEB UI ---
st.title("🎓 Scholarπ (ScholarPi) System")
st.subheader("Automated Multi-Criteria Academic Rigor Analytics")
st.info("⚡ Powered by Groq Cloud Llama-3.3-70B")

uploaded_file = st.file_uploader("Upload an Academic Paper (PDF)", type=["pdf"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    
    if st.button("Run Full Evaluation Pipeline", type="primary"):
        with st.spinner("Analyzing text, parsing literature indices, and generating π-Index metrics..."):
            pi, justifications, from_cache = process_paper(file_bytes, uploaded_file.name)
            
        if from_cache:
            st.success("ℹ️ Retrieved evaluation metrics from persistent system cache.")
        else:
            st.success("✅ Analysis completed successfully!")

        # High level score presentation
        st.metric(label="Calculated Final π-Index", value=f"{pi:.4f}")
        
        st.markdown("### Detailed Evaluation Matrix Reports")
        ordered_keys = ['S1', 'S2', 'S3', 'S4', 'S4b', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15']
        
        # Display nicely in an accordion style format
        for key in ordered_keys:
            data = justifications.get(key, {})
            score = data.get('score', 'N/A')
            reason = data.get('reason', 'No explanation provided.')
            
            with st.expander(f"Metric {key} — Score: {score}/10"):
                st.write(reason)
