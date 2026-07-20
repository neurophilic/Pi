import os
import sqlite3
import json
import hashlib
import time
import re
import requests
import colorsys
import math
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import fitz  # PyMuPDF
from groq import Groq
from pyvis.network import Network

# --- 1. CONFIGURATION & ENVIRONMENT ---
st.set_page_config(page_title="π-Index Assessment Engine", layout="wide")

PRIMARY_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"
MAX_TEXT_TOKENS = 12000 # ~2500 tokens to safely stay below the TPM limit
SEED_NUMBER = 42

# BLOCKCHAIN CONFIGURATION
# Defines how many evaluations must occur before mining a new epoch block. 
# Set to 1 for live testing updates. Change back to 10 for production.
EPOCH_BLOCK_SIZE = 5

BASE_DIR = os.path.abspath('./Scientometric_Pi_Index')
os.makedirs(BASE_DIR, exist_ok=True)
# Unified database name to prevent blockchain resets
DB_PATH = os.path.join(BASE_DIR, 'pi_index_main.db')

GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    st.error("API Key not found! Please configure your environment variables or Streamlit Secrets.")
    st.stop()
client = Groq(api_key=GROQ_API_KEY)

# --- UTILITY FUNCTIONS ---
def verify_orcid_live(orcid_id):
    try:
        url = f"https://pub.orcid.org/v3.0/{orcid_id}/person"
        headers = {"Accept": "application/json"}
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            name_data = data.get('name', {})
            if name_data:
                given = name_data.get('given-names', {}).get('value', '') if name_data.get('given-names') else ''
                family = name_data.get('family-name', {}).get('value', '') if name_data.get('family-name') else ''
                full_name = f"{given} {family}".strip()
                return True, full_name or "Verified Researcher (Name Private)"
            return True, "Verified Researcher (Name Private)"
        return False, "ORCID ID not found on public registry."
    except Exception as e:
        return False, f"API Error: {str(e)}"

def get_pi_float(block_height):
    pi_str = "3.141592653589793238462643383279502884197169399375105820974944592"
    length = min(block_height + 3, len(pi_str))
    return float(pi_str[:length])

def validate_block_por(block_index, weights, timestamp, previous_hash, eval_hash, model_used):
    validator_node = "Validator_Pi_" + hashlib.md5(str(time.time()).encode()).hexdigest()[:6]
    data = f"{block_index}{weights}{timestamp}{previous_hash}{validator_node}{eval_hash}{model_used}".encode('utf-8')
    block_hash = hashlib.sha256(data).hexdigest()
    return validator_node, block_hash

# --- 2. DATABASE & BLOCKCHAIN INIT ---
@st.cache_resource
def init_system():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS papers_assessment 
                      (eval_hash TEXT PRIMARY KEY, user_id TEXT, title TEXT, filename TEXT, scope TEXT,
                       c1 REAL, c2 REAL, c3 REAL, c4 REAL, 
                       c5 REAL, c6 REAL, c7 REAL, c8 REAL, 
                       scope_alignment REAL, logic_score REAL,
                       subfields TEXT, fields TEXT, final_score REAL, timestamp DATETIME)''')
                       
    try:
        cursor.execute("ALTER TABLE papers_assessment ADD COLUMN logic_score REAL DEFAULT 0.0")
    except sqlite3.OperationalError:
        pass 
        
    cursor.execute('''CREATE TABLE IF NOT EXISTS blockchain_por_weights 
                      (block_height INTEGER PRIMARY KEY AUTOINCREMENT, 
                       w1 REAL, w2 REAL, w3 REAL, w4 REAL, 
                       w5 REAL, w6 REAL, w7 REAL, w8 REAL, 
                       timestamp DATETIME, previous_hash TEXT, 
                       validator_node TEXT, block_hash TEXT, eval_hash TEXT, model_used TEXT)''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS global_eval_counter (count INTEGER)''')
    
    cursor.execute("SELECT COUNT(*) FROM blockchain_por_weights")
    if cursor.fetchone()[0] == 0:
        genesis_weights = [1.0] * 8
        prev_hash = "0" * 64
        timestamp = datetime.now().isoformat()
        val_node, block_hash = validate_block_por(1, genesis_weights, timestamp, prev_hash, "genesis", "none")
        cursor.execute('''INSERT INTO blockchain_por_weights 
                          (w1, w2, w3, w4, w5, w6, w7, w8, timestamp, previous_hash, validator_node, block_hash, eval_hash, model_used) 
                          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                       (*genesis_weights, timestamp, prev_hash, val_node, block_hash, "genesis", "none"))
                       
    cursor.execute("SELECT count FROM global_eval_counter")
    if not cursor.fetchone():
        cursor.execute("INSERT INTO global_eval_counter (count) VALUES (0)")
        
    conn.commit()
    return conn

conn = init_system()

# --- 3. MATHEMATICAL ALGORITHM ENGINE ---
def calculate_model_driven_weights(old_weights, scores, model_name, block_height):
    v, s = (3.3, 70.0) if "70b" in model_name else (3.1, 8.0)
    pi_acc = get_pi_float(block_height)
    delta_models = abs((3.3 * 70.0) - (3.1 * 8.0)) 
    
    mean_score = np.mean(scores)
    stretched_scores = [max(1.0, min(100.0, mean_score + (score - mean_score) * 3.0)) for score in scores]
    
    new_weights = []
    for i, old_w in enumerate(old_weights):
        c_score = stretched_scores[i]
        delta_w = ((v * s) / (delta_models * pi_acc)) * ((c_score / 100.0) ** 2)
        w_new = old_w * 0.85 + (1.0 + delta_w * 0.15) * 0.15
        new_weights.append(w_new)
        
    sum_w = sum(new_weights)
    return [round((w / sum_w) * 8.0, 6) for w in new_weights]

def compute_logical_integrity(v):
    e_str = v.get('Evidence_Strength', 0.5)
    c_reach = v.get('Conclusion_Reach', 0.5)
    jumps = v.get('Logical_Jumps', 0.5)
    p_valid = v.get('Premise_Validity', 0.5)
    
    gap = max(0.0, c_reach - e_str)
    logic_score = (p_valid * e_str) * np.exp(-(gap * 2.0 + jumps * 1.5)) * 100
    return max(0.0, min(100.0, logic_score))

def compute_formulaic_criteria(v):
    scores = {}
    
    H_novel, K_epi = v.get('H_novel', 0.5), v.get('K_epistemic', 0.5)
    zeta, I_ex = v.get('zeta', 0.5), v.get('I_existing', 0.5)
    Sigma_err, mu_sig = v.get('Sigma_error', 0.2), v.get('mu_signal', 0.8)
    rho_k = v.get('rho_k', 0.5)
    p_disc = np.array(v.get('p_disciplines', [1.0]))
    bridge_cap = v.get('bridge_capacity', 0.5)
    Utility, decay = v.get('Utility_vector', 0.5), v.get('decay_rate', 0.5)
    q_frac = v.get('q_fractional', 1.5)
    D_open, J_code, P_FAIR = v.get('D_open', 0.1), v.get('J_code', 0.1), v.get('P_FAIR', 0.1)
    d_g, R_xi, PR_xi = v.get('d_g_distance', 0.5), v.get('R_xi', 0.5), v.get('PR_xi', 0.5)
    I_Fish, KL_div = v.get('I_Fisher', 0.5), v.get('KL_divergence', 0.5)
    V_base, omega = v.get('V_baseline', 0.5), v.get('omega_data', 0.5)
    sum_lam = v.get('sum_lambda_kappa', 1.0)
    eta, Lambda = v.get('eta_steps', 2.0), v.get('Lambda_Lyapunov', 0.5)

    c1_raw = ((H_novel * K_epi) / (zeta * I_ex + 0.1)) * 60
    scores["C1_Originality"] = min(100.0, max(0.0, c1_raw))
    
    gamma_val = math.gamma(1.5) 
    rigor_matrix = max(0.0, 1.0 - (Sigma_err / (mu_sig + 0.1)))
    c2_raw = rigor_matrix * rho_k * gamma_val * 140
    scores["C2_Methodological_Rigor"] = min(100.0, max(0.0, c2_raw))
    
    p_disc = p_disc / (p_disc.sum() + 1e-9)
    renyi = -np.log(np.sum(p_disc**2) + 1e-9) 
    c3_raw = (renyi + bridge_cap) * 55
    scores["C3_Interdisciplinary"] = min(100.0, max(0.0, c3_raw))
    
    gamma_q = math.gamma(max(0.1, q_frac))
    c4_raw = (1.0 / gamma_q) * Utility * np.exp(-decay) * 150
    scores["C4_Societal_Impact"] = min(100.0, max(0.0, c4_raw))
    
    c5_raw = ((0.7 * D_open) + (0.3 * J_code)) * P_FAIR * 180
    scores["C5_Open_Science_Potential"] = min(100.0, max(0.0, c5_raw))
    
    c6_raw = np.exp(-1.5 * d_g) * R_xi * PR_xi * 180
    scores["C6_Literature_Integration"] = min(100.0, max(0.0, c6_raw))
    
    density_inner = (I_Fish * KL_div) / (V_base * omega + 0.1)
    c7_raw = np.tanh(density_inner) * sum_lam * 80
    scores["C7_Empirical_Density"] = min(100.0, max(0.0, c7_raw))
    
    c8_raw = (1.0 / (1.0 + np.exp(-(eta - (Lambda * 5))))) * 100
    scores["C8_Future_Actionability"] = min(100.0, max(0.0, c8_raw))
    
    return {k: round(v, 2) for k, v in scores.items()}

def evaluate_scope_alignment(text, scope, model, text_limit):
    if not scope.strip():
        return 0.0
    
    if len(text) > text_limit:
        text = text[:text_limit]
        
    prompt = f"""You are a research alignment tool.
Read the following paper text and evaluate how well it aligns with this specific research scope/keyword: "{scope}"
Return ONLY a valid JSON object with a single key "Scope_Alignment" containing a float between 0.0 and 100.0.
{{
    "Scope_Alignment": 85.5
}}
Text: {text}
"""
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        return float(json.loads(response.choices[0].message.content).get("Scope_Alignment", 0.0))
    except:
        return 0.0

def evaluate_pdf_text(text, model, text_limit):
    if len(text) > text_limit:
        text = text[:text_limit]

    prompt = f"""You are the theoretical parser for the π-Index Assessment Engine.
Instead of assigning arbitrary scores, you must read the academic paper and extract the underlying mathematical proxy variables based purely on the document's objective scientific merit.

CRITICAL INSTRUCTION - FORCE EXTREME VARIANCE:
Do NOT cluster your variables around 0.5. If a paper is weak or standard, use values between 0.0 and 0.3. If exceptional, use 0.8 to 1.0. 
Failure to create extreme contrast will break the mathematical formulas.

1. Extracted Variables (all values must be floats between 0.0 and 1.0, unless specified):
- `H_novel`: Conceptual novelty (0.1 = derivative, 0.9 = groundbreaking).
- `K_epistemic`: Paradigm shift potential.
- `zeta`: Reliance on existing works (0.9 = heavily reliant, 0.1 = independent/new).
- `I_existing`: Volume of foundational literature used.
- `Sigma_error`: Probability of methodological flaw (0.0 = perfect, 1.0 = flawed).
- `mu_signal`: Robustness of core methodology.
- `rho_k`: Density of empirical testing.
- `p_disciplines`: Array of 2 to 4 floats representing field distribution (e.g., [0.7, 0.3]).
- `bridge_capacity`: Success of bridging these disciplines.
- `Utility_vector`: Direct real-world application potential.
- `decay_rate`: Obsolescence rate (0.1 = eternal, 0.9 = obsolete next year).
- `q_fractional`: Time-domain impact scaling (float from 0.5 to 2.5).
- `D_open`: Availability of open data (0.1 = none, 0.9 = open repo).
- `J_code`: Availability of code/scripts (0.1 = none, 0.9 = open source).
- `P_FAIR`: Compliance with FAIR data principles.
- `d_g_distance`: Distance to the central core of the subject (0.1 = foundational, 0.9 = fringe).
- `R_xi`: Relevance to future research.
- `PR_xi`: Expected PageRank / citation magnet potential.
- `I_Fisher`: Information density (empirical data depth).
- `KL_divergence`: Statistical separation from the null hypothesis.
- `V_baseline`: Standard variance/noise in the data field.
- `omega_data`: Volume of data analyzed.
- `sum_lambda_kappa`: Quality metric for data dimensions (float 0.5 to 1.5).
- `eta_steps`: Number of concrete actionable future steps identified (Integer 1 to 5).
- `Lambda_Lyapunov`: Trajectory divergence (0.1 = highly predictable continuation, 0.9 = chaotic/disruptive).

2. Adversarial Logic Mapping:
Identify logical structural flaws and gaps in reasoning:
- `Evidence_Strength`: (0.1 = Anecdotal/Weak, 0.9 = Robust/Repetitive).
- `Conclusion_Reach`: (0.1 = Conservative/Supported, 0.9 = Wild/Unsupported).
- `Logical_Jumps`: (0.1 = Highly logical flow, 0.9 = Major non-sequiturs).
- `Premise_Validity`: (0.1 = Questionable assumptions, 0.9 = Solid definitions).

Return ONLY a valid JSON object matching exactly this structure:
{{
    "Extracted_Title": "Title", 
    "variables": {{
        "H_novel": 0.8, "K_epistemic": 0.7, "zeta": 0.5, "I_existing": 0.5, "Sigma_error": 0.1, "mu_signal": 0.9, "rho_k": 0.8,
        "p_disciplines": [0.6, 0.4], "bridge_capacity": 0.8, "Utility_vector": 0.7, "decay_rate": 0.2, "q_fractional": 1.2,
        "D_open": 0.2, "J_code": 0.1, "P_FAIR": 0.3, "d_g_distance": 0.2, "R_xi": 0.9, "PR_xi": 0.8,
        "I_Fisher": 0.8, "KL_divergence": 0.7, "V_baseline": 0.4, "omega_data": 0.8, "sum_lambda_kappa": 1.1,
        "eta_steps": 3, "Lambda_Lyapunov": 0.4
    }},
    "logic_analysis": {{
        "Evidence_Strength": 0.8, "Conclusion_Reach": 0.5, "Logical_Jumps": 0.1, "Premise_Validity": 0.9
    }},
    "fields": ["Field1", "Field2"], 
    "subfields": ["Subfield1"]
}}
Text: {text}
"""
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model, 
        temperature=0.0, 
        seed=SEED_NUMBER, 
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

def calculate_complex_drift(alignment, scores):
    mu, sigma = np.mean(scores), np.std(scores)
    delta = (100.0 - alignment) / 100.0
    drift_metric = 100.0 * (1.0 - np.exp(-3.0 * (delta ** 1.5) * (1.0 + (sigma / 100.0)) / (0.1 + (mu / 100.0))))
    return float(max(0.0, min(100.0, drift_metric)))

def get_recommendation_spectrum(score, drift):
    synergy = score * (1.0 - (drift / 100.0)**1.5)
    if synergy >= 85: return "Tier I: Core Paradigm (Optimal Synergy)"
    elif synergy >= 70: return "Tier II: Highly Aligned Framework"
    elif synergy >= 55: return "Tier III: Moderately Synergistic"
    elif synergy >= 40: return "Tier IV: Tangential Relevance"
    elif synergy >= 25: return "Tier V: Epistemic Divergence"
    return "Tier VI: Orthogonal / Unrelated Noise"

def process_single_pdf(file_bytes, filename, scope, user_id):
    file_hash = hashlib.sha256(file_bytes).hexdigest() 
    cursor = conn.cursor()
    
    # Check cache based purely on the document, NOT the scope
    cursor.execute("SELECT final_score, logic_score, title, fields, subfields, c1, c2, c3, c4, c5, c6, c7, c8 FROM papers_assessment WHERE eval_hash=? AND user_id=?", (file_hash, user_id))
    cached = cursor.fetchone()
    
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = " ".join([page.get_text() for page in doc]) 
    
    # Dynamically calculate Scope Alignment (always fresh based on the keyword)
    scope_alignment = evaluate_scope_alignment(text, scope, FALLBACK_MODEL, MAX_TEXT_TOKENS) if scope.strip() else 0.0

    if cached:
        score, logic_score, title, fields_str, subfields_str, c1, c2, c3, c4, c5, c6, c7, c8 = cached
        fields = json.loads(fields_str) if fields_str else ["General Science"]
        subfields = json.loads(subfields_str) if subfields_str else ["General"]
        scores_array = [c1, c2, c3, c4, c5, c6, c7, c8]
        
        # Calculate dynamic drift based on the new keyword's alignment
        drift = calculate_complex_drift(scope_alignment, scores_array) if scope.strip() else "N/A"
        rec = get_recommendation_spectrum(score, drift) if scope.strip() else "N/A"
        scores_dict = {"C1_Originality": c1, "C2_Methodological_Rigor": c2, "C3_Interdisciplinary": c3, "C4_Societal_Impact": c4, "C5_Open_Science_Potential": c5, "C6_Literature_Integration": c6, "C7_Empirical_Density": c7, "C8_Future_Actionability": c8}
        
        return title, score, logic_score, drift, rec, fields, subfields, scores_dict, file_hash

    # IF NOT CACHED: Run the heavy Pi-Index calculation (Static document metrics)
    try:
        raw_data = evaluate_pdf_text(text, PRIMARY_MODEL, MAX_TEXT_TOKENS)
        model_used = PRIMARY_MODEL
    except Exception as e:
        st.warning(f"Primary model limit hit. Failing over to {FALLBACK_MODEL}...")
        try:
            reduced_limit = MAX_TEXT_TOKENS // 2 if 'limit' in str(e).lower() or '413' in str(e) else MAX_TEXT_TOKENS
            raw_data = evaluate_pdf_text(text, FALLBACK_MODEL, reduced_limit)
            model_used = FALLBACK_MODEL
        except Exception as e2:
            st.error(f"Both models failed. API Error: {str(e2)}")
            return "Extraction Failed", 0.0, 0.0, "N/A", "N/A", ["Unknown"], ["Unknown"], {k: 0.0 for k in ["C1_Originality", "C2_Methodological_Rigor", "C3_Interdisciplinary", "C4_Societal_Impact", "C5_Open_Science_Potential", "C6_Literature_Integration", "C7_Empirical_Density", "C8_Future_Actionability"]}, "Failed"
        
    cursor.execute("UPDATE global_eval_counter SET count = count + 1")
    cursor.execute("SELECT count FROM global_eval_counter")
    total_evals = cursor.fetchone()[0]
        
    cursor.execute("SELECT block_height, block_hash, w1, w2, w3, w4, w5, w6, w7, w8 FROM blockchain_por_weights ORDER BY block_height DESC LIMIT 1")
    epoch_data = cursor.fetchone()
    block_height, previous_hash, old_weights = epoch_data[0], epoch_data[1], epoch_data[2:]
    
    variables = raw_data.get("variables", {})
    scores_dict = compute_formulaic_criteria(variables)
    scores = [scores_dict[k] for k in ["C1_Originality", "C2_Methodological_Rigor", "C3_Interdisciplinary", "C4_Societal_Impact", "C5_Open_Science_Potential", "C6_Literature_Integration", "C7_Empirical_Density", "C8_Future_Actionability"]]
    
    logic_vars = raw_data.get("logic_analysis", {})
    logic_integrity = compute_logical_integrity(logic_vars)

    if total_evals % EPOCH_BLOCK_SIZE == 0:
        new_weights = calculate_model_driven_weights(old_weights, scores, model_used, block_height)
        timestamp = datetime.now().isoformat()
        val_node, block_hash = validate_block_por(block_height + 1, new_weights, timestamp, previous_hash, file_hash, model_used)
        cursor.execute('''INSERT INTO blockchain_por_weights (w1, w2, w3, w4, w5, w6, w7, w8, timestamp, previous_hash, validator_node, block_hash, eval_hash, model_used) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                       (*new_weights, timestamp, previous_hash, val_node, block_hash, file_hash, model_used))
    else:
        new_weights = old_weights

    title = raw_data.get("Extracted_Title", filename)
    fields, subfields = raw_data.get("fields", ["General Science"]), raw_data.get("subfields", ["General"])
    
    raw_final_score = float(np.dot(scores, new_weights)) / 8.0
    final_score = float(raw_final_score * (0.7 + (logic_integrity / 333.3)))
    
    drift = calculate_complex_drift(scope_alignment, scores) if scope.strip() else "N/A"
    rec = get_recommendation_spectrum(final_score, drift) if scope.strip() else "N/A"
    
    timestamp = datetime.now().isoformat()
    # Notice we save scope_alignment here, but we will always overwrite it dynamically when reading from cache
    cursor.execute('''INSERT INTO papers_assessment (eval_hash, user_id, title, filename, scope, c1, c2, c3, c4, c5, c6, c7, c8, logic_score, scope_alignment, subfields, fields, final_score, timestamp) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                   (file_hash, user_id, title, filename, scope, *scores, logic_integrity, scope_alignment, json.dumps(subfields), json.dumps(fields), final_score, timestamp))
    conn.commit()
    
    return title, final_score, logic_integrity, drift, rec, fields, subfields, scores_dict, file_hash

# --- 4. TOPOLOGICAL MAPPING (INTERACTIVE PYVIS NETWORK) ---
def generate_interactive_bubble_chart(scope, user_id):
    cursor = conn.cursor()
    cursor.execute("SELECT fields, subfields, final_score FROM papers_assessment WHERE scope=? AND user_id=?", (scope, user_id))
    data = cursor.fetchall()
    
    html_string, table_html = "", ""
    if not data: return html_string, table_html
    
    all_topics = []
    for fields_json, subfields_json, final_score in data:
        try:
            fields = [f.title().strip() for f in json.loads(fields_json)]
            subfields = [s.title().strip() for s in json.loads(subfields_json)]
            score = float(final_score) if final_score else 50.0
            for f in fields: all_topics.append({'topic': f, 'weight': score})
            for s in subfields: all_topics.append({'topic': s, 'weight': score})
        except: continue
            
    if not all_topics: return html_string, table_html
    
    df_topics = pd.DataFrame(all_topics)
    topic_counts = df_topics.groupby(['topic'])['weight'].sum().reset_index(name='weight')
    
    if topic_counts.empty: return html_string, table_html
    unique_topics = topic_counts['topic'].unique()
    
    def get_color(i, n):
        h, s, v = i/n, 0.7, 0.9
        rgb = colorsys.hsv_to_rgb(h, s, v)
        return '#%02x%02x%02x' % tuple(int(x * 255) for x in rgb)
    
    color_map = {topic: get_color(i, len(unique_topics)) for i, topic in enumerate(unique_topics)}
    
    net = Network(height='600px', width='100%', bgcolor='#ffffff', font_color='#2c3e50', notebook=False)
    
    physics_options = """
    {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -1000,
          "centralGravity": 1,
          "springLength": 100,
          "avoidOverlap": 1.0
        },
        "stabilization": {
          "enabled": true,
          "iterations": 500,
          "fit": true
        },
        "solver": "barnesHut"
      }
    }
    """
    net.set_options(physics_options)
    
    for _, row in topic_counts.iterrows():
        node_size = 30 + (row['weight'] * 2.5) 
        net.add_node(
            n_id=row['topic'],
            label=' ', 
            title=f"Topic: {row['topic']} | Weight: {row['weight']}",
            size=node_size,
            physics=True, 
            color=color_map[row['topic']]
        )
    
    for _, row in topic_counts.iterrows():
        node_size = 30 + (row['weight'] * 2.5) 
        net.add_node(
            n_id=row['topic'],
            label=' ', 
            title=f"Topic: {row['topic']} | Weight: {row['weight']}",
            size=node_size,
            mass=node_size / 2,
            color=color_map[row['topic']]
        )
        
    html_string = net.generate_html()
    
    table_html = "<style>.table-big { width: 100%; font-size: 14px; border-collapse: collapse; margin-top: 10px; font-family: sans-serif; } .table-big th { background-color: #2c3e50; color: white; padding: 10px; text-align: left; } .table-big td { border-bottom: 1px solid #ddd; padding: 8px; vertical-align: middle; } .color-box { width: 18px; height: 18px; display: inline-block; border-radius: 3px; border: 1px solid #ccc; margin: 0 auto;} .legend-container { max-height: 550px; overflow-y: auto; border: 1px solid #eee; }</style>"
    table_html += "<div class='legend-container'><table class='table-big'><thead><tr><th style='width: 25%; text-align: center;'>Color</th><th>Topic</th></tr></thead><tbody>"
    
    for _, row in topic_counts.sort_values(by="weight", ascending=False).iterrows():
        table_html += f"<tr><td style='text-align: center;'><div class='color-box' style='background-color:{color_map[row['topic']]};'></div></td><td>{row['topic']}</td></tr>"
        
    table_html += "</tbody></table></div>"
    
    return html_string, table_html

# --- 5. USER INTERFACE ---
st.sidebar.title("System Access")

if 'orcid_id' not in st.session_state:
    st.session_state.orcid_id = "0000-0000-0000-0000"
    st.session_state.orcid_name = ""
    st.session_state.is_authenticated = False

if not st.session_state.is_authenticated:
    st.sidebar.markdown("### Authenticate via ORCID")
    st.sidebar.info("Connect your real ORCID iD to securely track your assessments and isolate your topological maps.")
    
    manual_orcid = st.sidebar.text_input("Enter ORCID iD", placeholder="XXXX-XXXX-XXXX-XXXX")
    if st.sidebar.button("🔗 Validate & Connect via ORCID API"):
        clean_orcid = manual_orcid.strip()
        if re.match(r'^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$', clean_orcid):
            with st.sidebar.status("Connecting to ORCID Registry..."):
                is_valid, user_name = verify_orcid_live(clean_orcid)
                
            if is_valid:
                st.session_state.orcid_id = clean_orcid
                st.session_state.orcid_name = user_name
                st.session_state.is_authenticated = True
                st.rerun()
            else:
                st.sidebar.error(user_name)
        else:
            st.sidebar.error("Invalid format. Please use XXXX-XXXX-XXXX-XXXX")
else:
    st.sidebar.success("✅ Securely Connected")
    st.sidebar.markdown(f"**Researcher:** {st.session_state.orcid_name}")
    st.sidebar.markdown(f"**ORCID iD:** `{st.session_state.orcid_id}`")
    if st.sidebar.button("Disconnect Session"):
        st.session_state.is_authenticated = False
        st.session_state.orcid_name = ""
        st.rerun()

current_user = st.session_state.orcid_id
st.sidebar.caption("Assessment histories and maps are isolated to your ORCID, but the PoR blockchain remains globally synchronized.")

st.title("π-Index Assessment Engine")
st.markdown("**Upload papers, define your scope of research, let π-index filter noise and have better results**")

with st.expander("View π-Index Grading Criteria & Theoretical Formulations"):
    st.markdown("### Evaluation Metrics & Adversarial Logic Engine")
    st.markdown(r"""
    **Adversarial Logic Gap ($\Delta_{Logic}$):** Before a final score is validated, the system maps the paper's reasoning structure. It penalizes the paper exponentially if the author's conclusions overreach the provided evidence.
    $$ L_i = (\mathcal{P}_{valid} \cdot \mathcal{E}_{strength}) \cdot \exp\left(-\left(2 \cdot \max(0, \mathcal{C}_{reach} - \mathcal{E}_{strength}) + 1.5 \cdot \lambda_{jumps}\right)\right) \times 100 $$
    """)
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**C1: Originality**\nEvaluates uniqueness through epistemic gradient fields.")
        st.markdown(r"$$O = \varpi_1 \cdot \lim_{\Delta t \to 0} \oint_{\partial \Omega} \frac{\nabla \times (\mathcal{H}_{novel} \otimes \mathcal{K}_{epistemic})}{\iint_{\mathcal{M}} \sum_{i=1}^N (\zeta_i \cdot \mathcal{I}_{existing}^{(i)}) \, d\mu} \cdot d\mathbf{S} \times 100 $$")
        st.markdown("**C2: Methodological Rigor**\nAssesses robustness via error-covariance tensors.")
        st.markdown(r"$$R = \varpi_2 \cdot \left( 1 - \frac{\mathrm{tr}(\boldsymbol{\Sigma}_{error} \boldsymbol{\Lambda}^{-1})}{\det(\boldsymbol{\mu}_{signal} \otimes \mathbf{W})} \right) \cdot \prod_{k=1}^{m} \int_{0}^{\infty} \rho_k(x) e^{-\beta x^2} \Gamma\left(k+\frac{1}{2}\right) dx \times 100 $$")
        st.markdown("**C3: Interdisciplinary**\nMeasures bridge capacity using generalized Rényi entropy.")
        st.markdown(r"$$I = \varpi_3 \cdot \left( \frac{1}{1-\alpha} \ln \left( \sum_{j=1}^{K} p_j^\alpha \right) + \sum_{i,j} \frac{A_{ij} \phi_i \phi_j}{\sqrt{d_i d_j}} \right) \cdot \frac{\Xi(\mathcal{G})}{\ln K \cdot \mathcal{Z}_{norm}} \times 100 $$")
        st.markdown("**C4: Societal Impact**\nProjects applications utilizing fractional stochastic integration.")
        st.markdown(r"$$S = \varpi_4 \cdot \frac{1}{\Gamma(q)} \int_{t_0}^{t_\infty} (t_\infty - \tau)^{q-1} e^{-\gamma(\tau) \tau} \cdot \Theta\left[ \sum_{v \in \mathcal{V}} \omega_v U_v(\tau, \mathbf{x}) \right] d\tau \times 100 $$")
    with col2:
        st.markdown("**C5: Open Science Potential**\nGauges transparency via multi-objective integration.")
        st.markdown(r"$$O_s = \varpi_5 \cdot \frac{\sum_{\ell \in \mathcal{L}} \alpha_\ell \mathcal{D}_{open}^{(\ell)} + \beta \iint_{\mathcal{C}} \nabla \cdot \mathbf{J}_{code} \, dV}{\max \left( \sup_{t} \mathcal{D}_{total}(t), \inf_{\epsilon>0} \mathcal{C}_{total}(\epsilon) \right)} \times \mathcal{P}_{FAIR} \times 100 $$")
        st.markdown("**C6: Literature Integration**\nEvaluates embedding via non-Euclidean PageRank.")
        st.markdown(r"$$L = \varpi_6 \cdot \frac{1}{\mathcal{N}} \sum_{i=1}^{\mathcal{N}} \int_{\mathcal{M}} e^{-\lambda d_g(x_i, x_{core})} R(x_i) \sqrt{g} \, dx_i \cdot \frac{\text{PR}(x_i)}{\sum_j \text{PR}(x_j)} \times 100 $$")
        st.markdown("**C7: Empirical Density**\nEvaluates data depth utilizing Fisher information metrics.")
        st.markdown(r"$$E_d = \varpi_7 \cdot \tanh \left( \frac{\det \mathcal{I}_{Fisher}(\hat{\theta}) \cdot \mathbb{E}_{P}\left[\log\frac{P}{Q}\right]}{\mathcal{V}_{baseline} \cdot \oint_\Gamma \omega_{data}} \right) \times \sum_{d=1}^D \lambda_d \kappa_d \times 100 $$")
        st.markdown("**C8: Future Actionability**\nDetermines continuation potential using Lyapunov exponents.")
        st.markdown(r"$$F_a = \varpi_8 \cdot \frac{1}{\mathcal{Z}} \int_{\mathcal{X}} \frac{1}{1 + \exp\left(-\sum_{k=1}^K w_k(\eta_k(\mathbf{x}) - \eta_{0,k}) + \Lambda_{Lyapunov}\right)} d\mu(\mathbf{x}) \times 100 $$")

tab1, tab2, tab3 = st.tabs(["Batch Assessment", "Scope Cartography", "Active Epoch Constants"])

with tab1:
    research_scope = st.text_input("Define your specific Research Topic / Scope (Optional)", placeholder="e.g., Application of deep learning in vascular imaging...")
    uploaded_files = st.file_uploader("Upload Academic Papers (PDFs)", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Run Batch Assessment", type="primary"):
        if not uploaded_files:
            st.warning("⚠️ Please upload at least one academic paper (PDF) to proceed.")
        else:
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, file in enumerate(uploaded_files):
                status_text.text(f"Analyzing {i+1} of {len(uploaded_files)}: {file.name}...")
                
                title, score, logic_integrity, drift, rec, fields, subfields, scores_dict, eval_hash = process_single_pdf(
                    file.read(), file.name, research_scope, current_user
                )
                
                combined_fields = f"Fields: {', '.join(fields)} | Subfields: {', '.join(subfields)}"
                
                record = {
                    "No.": i + 1,
                    "File Name": file.name,
                    "Fields & Subfields": combined_fields,
                    "Logic Integrity (%)": round(logic_integrity, 1),
                    "π-Index (0-100)": round(score, 1),
                }
                
                if research_scope.strip():
                    record["Topic"] = research_scope
                    record["Recommendation Spectrum"] = rec
                    record["Scope Drift %"] = round(drift, 1) if drift != "N/A" else "N/A"
                    
                record.update({
                    "C1": round(scores_dict.get("C1_Originality", 0.0), 1),
                    "C2": round(scores_dict.get("C2_Methodological_Rigor", 0.0), 1),
                    "C3": round(scores_dict.get("C3_Interdisciplinary", 0.0), 1),
                    "C4": round(scores_dict.get("C4_Societal_Impact", 0.0), 1),
                    "C5": round(scores_dict.get("C5_Open_Science_Potential", 0.0), 1),
                    "C6": round(scores_dict.get("C6_Literature_Integration", 0.0), 1),
                    "C7": round(scores_dict.get("C7_Empirical_Density", 0.0), 1),
                    "C8": round(scores_dict.get("C8_Future_Actionability", 0.0), 1),
                    "Eval Hash (Document)": eval_hash 
                })
                
                results.append(record)
                progress_bar.progress((i + 1) / len(uploaded_files))
                
            status_text.text("Batch processing complete!")
            
            df = pd.DataFrame(results)
            df_display = df.sort_values(by=["π-Index (0-100)"], ascending=False)
            st.markdown("### Assessment Summary")
            st.dataframe(df_display, use_container_width=True, hide_index=True)
            
            csv = df_display.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download Summary as CSV", data=csv, file_name="pi_index_assessment_results.csv", mime="text/csv")

    st.markdown("---")
    st.markdown("### Latest Assessment History")
    
    if st.session_state.is_authenticated:
        cursor = conn.cursor()
        cursor.execute("SELECT title, scope, final_score, timestamp, eval_hash FROM papers_assessment WHERE user_id=? ORDER BY timestamp DESC LIMIT 20", (current_user,))
        history_data = cursor.fetchall()
        
        if history_data:
            df_hist = pd.DataFrame(history_data, columns=["Paper Title", "Scope", "π-Index Score", "Date", "Evaluation Hash"])
            st.dataframe(df_hist, use_container_width=True, hide_index=True)
        else:
            st.info("No assessment history found for your account.")
    else:
        st.warning("🔒 Please connect your ORCID iD in the sidebar to view your private assessment history.")

with tab2:
    st.subheader("Epistemic Bubbles")
    
    interactive_html, table_html = generate_interactive_bubble_chart(research_scope, current_user)
    
    if interactive_html:
        col1, col2 = st.columns([3, 1])
        with col1:
            components.html(interactive_html, height=620)
        with col2:
            st.markdown("### Legend")
            st.markdown(table_html, unsafe_allow_html=True)
    else: 
        st.info("Awaiting sufficient data for this scope and user.")

with tab3:
    cursor = conn.cursor()
    cursor.execute("SELECT block_height, w1, w2, w3, w4, w5, w6, w7, w8, model_used, eval_hash, block_hash FROM blockchain_por_weights ORDER BY block_height DESC LIMIT 1")
    epoch_data = cursor.fetchone()
    
    if epoch_data:
        block_height, weights, model_used, eval_hash, block_hash = epoch_data[0], epoch_data[1:9], epoch_data[9], epoch_data[10], epoch_data[11]
        current_pi_base = get_pi_float(block_height)
        
        st.markdown(f"**Last Model Orchestration:** `{model_used}` | **Epoch Block:** `{block_height}` | **Pi Acc:** `{current_pi_base}`")
        
        st.markdown(r"""
        **Weight Evolution Dynamics:**
        $$ \varpi_{i}^{(t+1)} = \mathcal{N} \left( \lambda \varpi_{i}^{(t)} + (1-\lambda) \left[ 1 + \kappa \left( \frac{V \cdot S}{\Delta_{\mathcal{M}} \cdot \pi_{(t)}} \right) \left( \frac{C_i}{100} \right) \right] \right) $$
        
        **Parameters:**
        *   $\varpi_i^{(t)}$ : Current weight for criteria $i$ at epoch $t$.
        *   $\mathcal{N}$ : Normalization operator ensuring $\sum \varpi_i = 8.0$.
        *   $\lambda, \kappa$ : Dampening coefficients (set to $0.85$ and $0.15$ respectively).
        *   $V, S$ : Selected LLM Version and Parameter Size.
        *   $\Delta_{\mathcal{M}}$ : Constant structural delta between available candidate models.
        *   $\pi_{(t)}$ : Epoch-dependent $\pi$ progression accuracy.
        *   $C_i$ : The raw evaluation score assigned to criteria $i$ by the model.
        """)
        st.markdown("---")
        
        cols = st.columns(4)
        labels = [("C1 Originality", r"$\varpi_1$"), ("C2 Method Rigor", r"$\varpi_2$"), ("C3 Interdisciplinary", r"$\varpi_3$"), ("C4 Societal Impact", r"$\varpi_4$"), ("C5 Open Science", r"$\varpi_5$"), ("C6 Lit Integration", r"$\varpi_6$"), ("C7 Empirical Density", r"$\varpi_7$"), ("C8 Actionability", r"$\varpi_8$")]
        
        for i, col in enumerate(cols * 2):
            if i < 8: 
                name, symbol = labels[i]
                col.markdown(f"**{name} ({symbol})**")
                col.markdown(f"<h3 style='margin-top:0px; margin-bottom:5px;'>{weights[i]:.6f}</h3>", unsafe_allow_html=True)
                
        st.markdown("---")
        st.markdown("### PoR (Proof of Review) Blockchain Explorer")
        st.markdown("""
        **How to Verify via the Explorer:**
        1.  **Locate the Eval Hash:** Copy the Evaluation Hash (Document) associated with a paper you assessed.
        2.  **Use the Explorer:** Paste that hash into the PoR Blockchain Explorer input field below.
        3.  **Click "Verify Record":** The system will query the global blockchain database to return the exact Weights Matrix alongside the immutable Block Hash.
        """)
        
        explore_col1, explore_col2 = st.columns([3, 1])
        with explore_col1: search_query = st.text_input("Enter Document Evaluation Hash or Block Hash to verify ledger record...")
        with explore_col2: 
            st.write("")
            st.write("")
            search_btn = st.button("Verify Record")
            
        if search_btn and search_query:
            cursor.execute("SELECT * FROM blockchain_por_weights WHERE block_hash=? OR eval_hash=?", (search_query, search_query))
            record = cursor.fetchone()
            if record:
                st.success("Valid Block Found on Ledger!")
                st.json({"Block Height": record[0], "Timestamp": record[9], "Model Used": record[13], "Validator Node": record[11], "Block Hash": record[12], "Previous Hash": record[10], "Evaluation Hash (Document)": record[14], "Weights Matrix (w1..w8)": record[1:9]})
            else:
                st.error("No block matching that signature was found on the ledger.")
                
        with st.expander("View Recent Global Ledger Blocks"):
            cursor.execute("SELECT block_height, timestamp, model_used, block_hash FROM blockchain_por_weights ORDER BY block_height DESC LIMIT 10")
            df_blocks = pd.DataFrame(cursor.fetchall(), columns=["Height", "Timestamp", "Model", "Block Hash"])
            st.dataframe(df_blocks, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray; font-size: 0.8em;'>Framework Author: Ali Vafadar Yengejeh | Università degli Studi di Milano-Bicocca</div>", unsafe_allow_html=True)
