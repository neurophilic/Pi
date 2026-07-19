import os
import sqlite3
import json
import hashlib
import time
from datetime import datetime, timedelta
from decimal import Decimal, getcontext
import requests
import streamlit as st
import fitz  # PyMuPDF
from groq import Groq, RateLimitError

# --- 1. CONFIGURATION & ENVIRONMENT ---
st.set_page_config(page_title="Scholarπ Paper Evaluator", page_icon="🎓", layout="wide")

PRIMARY_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"

SEED_NUMBER = 42
MAX_TEXT_TOKENS_FOR_LLM = 4000

BASE_DIR = os.path.abspath('./ScholarPi_System_Cloud')
os.makedirs(BASE_DIR, exist_ok=True)
DB_PATH = os.path.join(BASE_DIR, 'scholar_pi_hashed.db')

# --- 2. API & DATABASE SETUP ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    st.error("⚠️ Groq API Key not found! Please add it to your Streamlit Secrets.")
    st.stop()
    
client = Groq(api_key=GROQ_API_KEY)

@st.cache_resource
def init_system():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute('''CREATE TABLE IF NOT EXISTS papers 
                    (file_hash TEXT PRIMARY KEY, filename TEXT, pi_index REAL, justifications TEXT, timestamp TEXT)''')
    conn.commit()
    return conn

conn = init_system()

# --- 3. ALGORITHMIC SCORING LOGIC ---
def calculate_algorithmic_scores(ai_data, search_results_count):
    """Takes the raw variables extracted by the AI and calculates strict 0-10 scores."""
    computed = {}
    
    def safe_get(d, key, default=0):
        val = d.get(key, default)
        return val if val is not None else default

    for k in ['S1', 'S2', 'S3', 'S4', 'S4b', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15']:
        computed[k] = {"score": 5.0, "reason": ai_data.get(k, {}).get("reason", "No reason provided.")}
    
    try:
        # S1: CharDensity
        s1_data = ai_data.get('S1', {})
        computed['S1']['score'] = min(10.0, float(safe_get(s1_data, 'vocab_level_1_to_5', 2.5) + safe_get(s1_data, 'sentence_complexity_1_to_5', 2.5)))

        # S2: NumDensity
        s2_data = ai_data.get('S2', {})
        s2_score = (3.0 if safe_get(s2_data, 'has_data_tables', False) else 0) + \
                   min(5.0, float(safe_get(s2_data, 'statistical_tests_count', 0))) + \
                   (2.0 if safe_get(s2_data, 'empirical_claims_supported', False) else 0)
        computed['S2']['score'] = min(10.0, s2_score)

        # S3: Reasoning
        s3_data = ai_data.get('S3', {})
        s3_score = (float(safe_get(s3_data, 'logical_flow_1_to_5', 3)) * 1.5) + \
                   (2.5 if safe_get(s3_data, 'addresses_counter_arguments', False) else 0)
        computed['S3']['score'] = min(10.0, s3_score)

        # S4: CitationIntegration
        s4_data = ai_data.get('S4', {})
        computed['S4']['score'] = min(10.0, float(safe_get(s4_data, 'lit_review_depth_1_to_5', 2.5) + safe_get(s4_data, 'citation_support_1_to_5', 2.5)))

        # S4b: CitationVolume
        s4b_data = ai_data.get('S4b', {})
        computed['S4b']['score'] = min(10.0, float(safe_get(s4b_data, 'total_citations_count', 25)) / 5.0)

        # S5: AuthorDiversity
        s5_data = ai_data.get('S5', {})
        computed['S5']['score'] = min(10.0, min(5.0, float(safe_get(s5_data, 'number_of_authors', 1))) + min(5.0, float(safe_get(s5_data, 'number_of_institutions', 1))))

        # S6: Expertise
        s6_data = ai_data.get('S6', {})
        computed['S6']['score'] = min(10.0, float(safe_get(s6_data, 'terminology_1_to_5', 2.5) + safe_get(s6_data, 'methodological_rigor_1_to_5', 2.5)))

        # S7: Novelty
        s7_data = ai_data.get('S7', {})
        computed['S7']['score'] = min(10.0, float(safe_get(s7_data, 'innovation_level_1_to_8', 4)) + (2.0 if safe_get(s7_data, 'explicit_new_contribution', False) else 0))

        # S8: Suggestions
        s8_data = ai_data.get('S8', {})
        computed['S8']['score'] = min(10.0, float(safe_get(s8_data, 'actionability_1_to_8', 4)) + (2.0 if safe_get(s8_data, 'mentions_future_work', False) else 0))

        # S9: Fees
        s9_data = ai_data.get('S9', {})
        computed['S9']['score'] = min(10.0, (5.0 if safe_get(s9_data, 'mentions_funding', False) else 0) + (5.0 if safe_get(s9_data, 'states_coi', False) else 0))

        # S10: Recency
        s10_data = ai_data.get('S10', {})
        computed['S10']['score'] = min(10.0, float(safe_get(s10_data, 'percent_recent_citations', 50)) / 10.0)

        # S11: FieldDiversity
        s11_data = ai_data.get('S11', {})
        computed['S11']['score'] = min(10.0, float(safe_get(s11_data, 'disciplines_mentioned', 1)) * 2.5)

        # S12: Validation
        s12_data = ai_data.get('S12', {})
        s12_score = (2.0 if safe_get(s12_data, 'validation_method_used', False) else 0) + \
                    (2.0 if safe_get(s12_data, 'sample_size_stated', False) else 0) + \
                    float(safe_get(s12_data, 'reproducible_1_to_6', 3))
        computed['S12']['score'] = min(10.0, s12_score)

        # S13: LogicalCoherence
        s13_data = ai_data.get('S13', {})
        computed['S13']['score'] = min(10.0, float(safe_get(s13_data, 'structure_1_to_5', 2.5) + safe_get(s13_data, 'readability_1_to_5', 2.5)))

        # S14: WebGroundedUniqueness
        computed['S14']['score'] = max(0.0, 10.0 - (float(search_results_count) / 10.0))
        computed['S14']['reason'] = f"Calculated algorithmically. Semantic Scholar sweep found {search_results_count} similar papers."

        # S15: AuthorHIndex
        s15_data = ai_data.get('S15', {})
        computed['S15']['score'] = min(10.0, float(safe_get(s15_data, 'author_prominence_1_to_10', 5)))

    except Exception as e:
        print(f"Algorithm mapping error: {e}")
        # Failsafe clamping ensures the app never crashes
        for k in computed:
            computed[k]['score'] = max(0.0, min(10.0, float(computed[k].get('score', 5.0))))

    return computed

def calculate_pi_index(base_scores, uniqueness_score_10pt, delta_t=0):
    getcontext().prec = 10
    pi = Decimal('3.1415926535')
    avg_score = sum(base_scores) / len(base_scores) 
    u_score = uniqueness_score_10pt / 10.0 
    
    drift = (pi / Decimal('3.14')) ** Decimal(str(delta_t))
    u_multiplier = Decimal('0.5') + (Decimal(str(u_score)) * Decimal('0.5'))
    return float(Decimal(str(avg_score)) * drift * u_multiplier)

# --- 4. CORE PROCESSING FUNCTIONS ---
def process_paper(file_bytes, filename):
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    
    cursor = conn.cursor()
    cursor.execute("SELECT pi_index, justifications, timestamp FROM papers WHERE file_hash=?", (file_hash,))
    cached_row = cursor.fetchone()

    if cached_row:
        cached_time = datetime.fromisoformat(cached_row[2])
        if datetime.now() - cached_time < timedelta(days=30):
            return cached_row[0], json.loads(cached_row[1]), True, False

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = " ".join([page.get_text() for page in doc])

    def evaluate_with_model(target_model):
        kw_response = client.chat.completions.create(
            messages=[{"role": "user", "content": f"Extract 5 critical research keywords. Return JSON: {{'keywords': []}}. Text: {text[:MAX_TEXT_TOKENS_FOR_LLM]}"}],
            model=target_model, temperature=0, seed=SEED_NUMBER, response_format={"type": "json_object"}
        )
        keywords = json.loads(kw_response.choices[0].message.content).get('keywords', [])
        time.sleep(2) 
        
        search_results_count = 0
        try:
            res = requests.get("https://api.semanticscholar.org/graph/v1/paper/search", params={"query": " ".join(keywords), "limit": 100})
            search_results_count = len(res.json().get('data', []))
        except:
            pass

        # NEW PROMPT: Asking for Variables instead of Scores
        eval_prompt = f"""Read the following academic text. Extract the specific variables required for our algorithmic scoring system.
        Return ONLY a JSON object exactly matching this structure, along with a 1-sentence 'reason' explaining your findings for each section:
        {{
            "S1": {{"vocab_level_1_to_5": int, "sentence_complexity_1_to_5": int, "reason": str}},
            "S2": {{"has_data_tables": bool, "statistical_tests_count": int, "empirical_claims_supported": bool, "reason": str}},
            "S3": {{"logical_flow_1_to_5": int, "addresses_counter_arguments": bool, "reason": str}},
            "S4": {{"lit_review_depth_1_to_5": int, "citation_support_1_to_5": int, "reason": str}},
            "S4b": {{"total_citations_count": int, "reason": str}},
            "S5": {{"number_of_authors": int, "number_of_institutions": int, "reason": str}},
            "S6": {{"terminology_1_to_5": int, "methodological_rigor_1_to_5": int, "reason": str}},
            "S7": {{"innovation_level_1_to_8": int, "explicit_new_contribution": bool, "reason": str}},
            "S8": {{"actionability_1_to_8": int, "mentions_future_work": bool, "reason": str}},
            "S9": {{"mentions_funding": bool, "states_coi": bool, "reason": str}},
            "S10": {{"percent_recent_citations": int, "reason": str}},
            "S11": {{"disciplines_mentioned": int, "reason": str}},
            "S12": {{"validation_method_used": bool, "sample_size_stated": bool, "reproducible_1_to_6": int, "reason": str}},
            "S13": {{"structure_1_to_5": int, "readability_1_to_5": int, "reason": str}},
            "S15": {{"author_prominence_1_to_10": int, "reason": str}}
        }}
        Paper Text: {text[:MAX_TEXT_TOKENS_FOR_LLM]}"""
        
        scores_json = client.chat.completions.create(
            messages=[{"role": "user", "content": eval_prompt}], 
            model=target_model, temperature=0, seed=SEED_NUMBER, response_format={"type": "json_object"}
        ).choices[0].message.content
        
        raw_ai_data = json.loads(scores_json)
        
        # Apply the Algorithmic Math!
        return calculate_algorithmic_scores(raw_ai_data, search_results_count)

    used_fallback = False
    try:
        final_scores_data = evaluate_with_model(PRIMARY_MODEL)
    except RateLimitError:
        used_fallback = True
        st.toast("⚠️ Rate limit reached! Switching to Instant Model...", icon="🔄")
        st.warning(f"Free-tier limits reached. Computing algorithms using fallback `{FALLBACK_MODEL}` model...")
        time.sleep(3) 
        final_scores_data = evaluate_with_model(FALLBACK_MODEL)
    
    score_keys = ['S1', 'S2', 'S3', 'S4', 'S4b', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S15']
    base_score_list = [final_scores_data[k]['score'] for k in score_keys]
    uniqueness_s14 = final_scores_data['S14']['score']
    
    pi = calculate_pi_index(base_score_list, uniqueness_s14)
    
    conn.execute("INSERT OR REPLACE INTO papers (file_hash, filename, pi_index, justifications, timestamp) VALUES (?,?,?,?,?)",
                 (file_hash, filename, pi, json.dumps(final_scores_data), datetime.now().isoformat()))
    conn.commit()
    
    return pi, final_scores_data, False, used_fallback

# --- 5. STREAMLIT WEB UI ---
st.title("🎓 Scholarπ (ScholarPi) System")
st.subheader("Algorithmic Multi-Criteria Academic Rigor Analytics")

uploaded_file = st.file_uploader("Upload an Academic Paper (PDF)", type=["pdf"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    
    if st.button("Run Algorithmic Evaluation", type="primary"):
        with st.spinner("Extracting variables, crunching algorithms, and generating metrics..."):
            pi, justifications, from_cache, used_fallback = process_paper(file_bytes, uploaded_file.name)
            
        if from_cache:
            st.success("ℹ️ Retrieved calculated metrics from cache.")
        else:
            model_used = FALLBACK_MODEL if used_fallback else PRIMARY_MODEL
            st.success(f"✅ Algorithms executed successfully via `{model_used}` data extraction!")

        st.metric(label="Calculated Final π-Index", value=f"{pi:.4f}")
        st.markdown("### Algorithmic Evaluation Matrix")
        
        ordered_keys = ['S1', 'S2', 'S3', 'S4', 'S4b', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15']
        METRIC_NAMES = {'S1': 'CharDensity', 'S2': 'NumDensity', 'S3': 'Reasoning', 'S4': 'CitationIntegration', 'S4b': 'CitationVolume', 'S5': 'AuthorDiversity', 'S6': 'Expertise', 'S7': 'Novelty', 'S8': 'Suggestions', 'S9': 'Fees', 'S10': 'Recency', 'S11': 'FieldDiversity', 'S12': 'Validation', 'S13': 'LogicalCoherence', 'S14': 'WebGroundedUniqueness', 'S15': 'AuthorHIndex'}
        
        for key in ordered_keys:
            data = justifications.get(key, {})
            # We format the float cleanly to 1 decimal point for display
            score = f"{data.get('score', 0):.1f}" 
            reason = data.get('reason', 'No explanation provided.')
            metric_name = METRIC_NAMES.get(key, 'Unknown Metric')
            
            with st.expander(f"Metric {key} ({metric_name}) — Calculated Score: {score}/10"):
                st.write(reason)
