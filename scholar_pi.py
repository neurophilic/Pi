import os
import sqlite3
import json
import hashlib
import time
import re
import requests
import colorsys
import math
import tempfile
import threading
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import fitz  # PyMuPDF
from pyvis.network import Network
from groq import Groq
from tenacity import retry, wait_exponential, stop_after_attempt

# --- Machine Learning Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --- 1. CONFIGURATION & ENVIRONMENT SETUP ---
st.set_page_config(page_title="π-Index Assessment Engine", layout="wide")

PRIMARY_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"
MAX_TEXT_TOKENS = 12000 
SEED_NUMBER = 42
EPOCH_BLOCK_SIZE = 5

BASE_DIR = os.path.abspath('./Scientometric_Pi_Index')
os.makedirs(BASE_DIR, exist_ok=True)
DB_PATH = os.path.join(BASE_DIR, 'pi_index_main.db')

GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    st.error("API Key not found! Please configure your environment variables or Streamlit Secrets.")
    st.stop()
client = Groq(api_key=GROQ_API_KEY)

# Use a threading lock to prevent SQLite database locked errors during batch parallel writes
db_lock = threading.Lock()

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
    data_string = f"{block_index}{weights}{timestamp}{previous_hash}{validator_node}{eval_hash}{model_used}"
    block_hash = hashlib.sha256(data_string.encode('utf-8')).hexdigest()
    return validator_node, block_hash

# --- 2. DATABASE INITIALIZATION ---
@st.cache_resource
def init_system():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS papers_assessment 
                      (eval_hash TEXT PRIMARY KEY, user_id TEXT, title TEXT, filename TEXT, scope TEXT,
                       c1 REAL, c2 REAL, c3 REAL, c4 REAL, 
                       c5 REAL, c6 REAL, c7 REAL, c8 REAL, 
                       scope_alignment REAL, logic_score REAL,
                       subfields TEXT, fields TEXT, author_name TEXT, final_score REAL, timestamp DATETIME)''')
                       
    try: cursor.execute("ALTER TABLE papers_assessment ADD COLUMN logic_score REAL DEFAULT 0.0")
    except sqlite3.OperationalError: pass 

    try: cursor.execute("ALTER TABLE papers_assessment ADD COLUMN author_name TEXT DEFAULT 'Unknown Author'")
    except sqlite3.OperationalError: pass 
        
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

# --- 3. MATHEMATICAL EVALUATION ENGINE ---
def calculate_model_driven_weights(old_weights, scores, model_name, block_height):
    model_version, model_size = (3.3, 70.0) if "70b" in model_name else (3.1, 8.0)
    pi_accuracy = get_pi_float(block_height)
    delta_models = abs((3.3 * 70.0) - (3.1 * 8.0)) 
    mean_score = np.mean(scores)
    
    new_weights = []
    for i, old_w in enumerate(old_weights):
        stretched_score = max(1.0, min(100.0, mean_score + (scores[i] - mean_score) * 3.0))
        weight_shift = ((model_version * model_size) / (delta_models * pi_accuracy)) * ((stretched_score / 100.0) ** 2)
        w_new = old_w * 0.85 + (1.0 + weight_shift * 0.15) * 0.15
        new_weights.append(w_new)
        
    sum_of_weights = sum(new_weights)
    return [round((w / sum_of_weights) * 8.0, 6) for w in new_weights]

def compute_logical_integrity(extracted_logic_vars):
    evidence = float(extracted_logic_vars.get('Evidence_Strength', 0.5))
    conclusion_reach = float(extracted_logic_vars.get('Conclusion_Reach', 0.5))
    jumps = float(extracted_logic_vars.get('Logical_Jumps', 0.5))
    premise = float(extracted_logic_vars.get('Premise_Validity', 0.5))
    
    logic_gap = max(0.0, conclusion_reach - evidence)
    logic_score = (premise * evidence) * np.exp(-(logic_gap * 2.0 + jumps * 1.5)) * 100
    return max(0.0, min(100.0, logic_score))

def compute_formulaic_criteria(v):
    scores = {}
    
    # 1. Originality
    c1_raw = ((float(v.get('H_novel', 0.5)) * float(v.get('K_epistemic', 0.5))) / (float(v.get('zeta', 0.5)) * float(v.get('I_existing', 0.5)) + 0.1)) * 60
    scores["C1_Originality"] = min(100.0, max(0.0, c1_raw))
    
    # 2. Methodological Rigor
    rigor_matrix = max(0.0, 1.0 - (float(v.get('Sigma_error', 0.2)) / (float(v.get('mu_signal', 0.8)) + 0.1)))
    c2_raw = rigor_matrix * float(v.get('rho_k', 0.5)) * math.gamma(1.5) * 140
    scores["C2_Methodological_Rigor"] = min(100.0, max(0.0, c2_raw))
    
    # 3. Interdisciplinary
    p_disc = np.array(v.get('p_disciplines', [1.0]), dtype=float)
    p_disc = p_disc / (p_disc.sum() + 1e-9)
    renyi_entropy = -np.log(np.sum(p_disc**2) + 1e-9) 
    c3_raw = (renyi_entropy + float(v.get('bridge_capacity', 0.5))) * 55
    scores["C3_Interdisciplinary"] = min(100.0, max(0.0, c3_raw))
    
    # 4. Societal Impact
    gamma_q = math.gamma(max(0.1, float(v.get('q_fractional', 1.5))))
    c4_raw = (1.0 / gamma_q) * float(v.get('Utility_vector', 0.5)) * np.exp(-float(v.get('decay_rate', 0.5))) * 150
    scores["C4_Societal_Impact"] = min(100.0, max(0.0, c4_raw))
    
    # 5. Open Science Potential
    c5_raw = ((0.7 * float(v.get('D_open', 0.1))) + (0.3 * float(v.get('J_code', 0.1)))) * float(v.get('P_FAIR', 0.1)) * 180
    scores["C5_Open_Science_Potential"] = min(100.0, max(0.0, c5_raw))
    
    # 6. Literature Integration
    c6_raw = np.exp(-1.5 * float(v.get('d_g_distance', 0.5))) * float(v.get('R_xi', 0.5)) * float(v.get('PR_xi', 0.5)) * 180
    scores["C6_Literature_Integration"] = min(100.0, max(0.0, c6_raw))
    
    # 7. Empirical Density
    density_inner = (float(v.get('I_Fisher', 0.5)) * float(v.get('KL_divergence', 0.5))) / (float(v.get('V_baseline', 0.5)) * float(v.get('omega_data', 0.5)) + 0.1)
    c7_raw = np.tanh(density_inner) * float(v.get('sum_lambda_kappa', 1.0)) * 80
    scores["C7_Empirical_Density"] = min(100.0, max(0.0, c7_raw))
    
    # 8. Future Actionability
    eta = float(v.get('eta_steps', 2.0))
    lambda_lyapunov = float(v.get('Lambda_Lyapunov', 0.5))
    c8_raw = (1.0 / (1.0 + np.exp(-(eta - (lambda_lyapunov * 5))))) * 100
    scores["C8_Future_Actionability"] = min(100.0, max(0.0, c8_raw))
    
    return {k: round(v, 2) for k, v in scores.items()}

@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(4))
def evaluate_scope_alignment(text, scope, model, text_limit):
    if not scope.strip(): return 0.0
    
    prompt = f"""You are a research alignment tool.
Read the following paper text and evaluate how well it aligns with this specific research scope/keyword: "{scope}"
Return ONLY a valid JSON object with a single key "Scope_Alignment" containing a float between 0.0 and 100.0.
{{
    "Scope_Alignment": 85.5
}}
Text: {text}"""
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    return float(json.loads(response.choices[0].message.content).get("Scope_Alignment", 0.0))

@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(4))
def evaluate_pdf_text(text, model):
    prompt = f"""You are the theoretical parser for the π-Index Assessment Engine.
Extract the underlying mathematical proxy variables based purely on the document's objective scientific merit. Look at the first page of the text to find the actual names of the human authors written below the title.

1. Extracted Metadata:
- `Extracted_Title`: The full title of the paper.
- `Extracted_Author`: The primary author name(s) (e.g., "Jane Doe et al.").

2. Extracted Variables (Floats 0.0 - 1.0 unless specified):
- `H_novel`: Conceptual novelty.
- `K_epistemic`: Paradigm shift potential.
- `zeta`: Reliance on existing works.
- `I_existing`: Volume of foundational literature used.
- `Sigma_error`: Probability of methodological flaw.
- `mu_signal`: Robustness of core methodology.
- `rho_k`: Density of empirical testing.
- `p_disciplines`: Array of 2 to 4 floats representing field distribution (e.g., [0.7, 0.3]).
- `bridge_capacity`: Success of bridging these disciplines.
- `Utility_vector`: Direct real-world application potential.
- `decay_rate`: Obsolescence rate.
- `q_fractional`: Time-domain impact scaling (float from 0.5 to 2.5).
- `D_open`: Availability of open data.
- `J_code`: Availability of code/scripts.
- `P_FAIR`: Compliance with FAIR data principles.
- `d_g_distance`: Distance to the central core of the subject.
- `R_xi`: Relevance to future research.
- `PR_xi`: Expected PageRank.
- `I_Fisher`: Information density.
- `KL_divergence`: Statistical separation from null hypothesis.
- `V_baseline`: Standard variance/noise.
- `omega_data`: Volume of data analyzed.
- `sum_lambda_kappa`: Quality metric for data dimensions (float 0.5 to 1.5).
- `eta_steps`: Number of actionable future steps (Integer 1 to 5).
- `Lambda_Lyapunov`: Trajectory divergence.

3. Adversarial Logic Mapping (Floats 0.0 - 1.0):
- `Evidence_Strength`
- `Conclusion_Reach`
- `Logical_Jumps`
- `Premise_Validity`

Return ONLY a valid JSON object matching exactly this structure:
{{
    "Extracted_Title": "Title", 
    "Extracted_Author": "Author Name",
    "variables": {{"H_novel": 0.8, "K_epistemic": 0.7, "zeta": 0.5, "I_existing": 0.5, "Sigma_error": 0.1, "mu_signal": 0.9, "rho_k": 0.8, "p_disciplines": [0.6, 0.4], "bridge_capacity": 0.8, "Utility_vector": 0.7, "decay_rate": 0.2, "q_fractional": 1.2, "D_open": 0.2, "J_code": 0.1, "P_FAIR": 0.3, "d_g_distance": 0.2, "R_xi": 0.9, "PR_xi": 0.8, "I_Fisher": 0.8, "KL_divergence": 0.7, "V_baseline": 0.4, "omega_data": 0.8, "sum_lambda_kappa": 1.1, "eta_steps": 3, "Lambda_Lyapunov": 0.4}},
    "logic_analysis": {{"Evidence_Strength": 0.8, "Conclusion_Reach": 0.5, "Logical_Jumps": 0.1, "Premise_Validity": 0.9}},
    "fields": ["Field1", "Field2"], 
    "subfields": ["Subfield1"]
}}
Text: {text}"""
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model, 
        temperature=0.0, 
        seed=SEED_NUMBER, 
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

def calculate_complex_drift(alignment, scores):
    alignment_gap = (100.0 - alignment) / 100.0
    drift_metric = 100.0 * (1.0 - np.exp(-3.0 * (alignment_gap ** 1.5) * (1.0 + (np.std(scores) / 100.0)) / (0.1 + (np.mean(scores) / 100.0))))
    return float(max(0.0, min(100.0, drift_metric)))

def get_recommendation_spectrum(score, drift):
    synergy = score * (1.0 - (drift / 100.0)**1.5)
    if synergy >= 85: return "Tier I: Core Paradigm (Optimal Synergy)"
    elif synergy >= 70: return "Tier II: Highly Aligned Framework"
    elif synergy >= 55: return "Tier III: Moderately Synergistic"
    elif synergy >= 40: return "Tier IV: Tangential Relevance"
    elif synergy >= 25: return "Tier V: Epistemic Divergence"
    else: return "Tier VI: Orthogonal / Unrelated Noise"

def process_single_pdf(file_bytes, filename, scope, user_id):
    file_hash = hashlib.sha256(file_bytes).hexdigest() 
    cursor = conn.cursor()
    
    cursor.execute("SELECT final_score, logic_score, title, fields, subfields, author_name, c1, c2, c3, c4, c5, c6, c7, c8 FROM papers_assessment WHERE eval_hash=? AND user_id=?", (file_hash, user_id))
    cached = cursor.fetchone()
    
    # Memory-optimized PDF Reading
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pdf_meta_author = doc.metadata.get("author", "").strip()
    
    full_text = ""
    for page in doc:
        full_text += page.get_text() + " "
        if len(full_text) > MAX_TEXT_TOKENS:
            full_text = full_text[:MAX_TEXT_TOKENS]
            break

    scope_alignment = evaluate_scope_alignment(full_text, scope, FALLBACK_MODEL, MAX_TEXT_TOKENS) if scope.strip() else 0.0

    if cached:
        score, logic_score, title, fields_str, subfields_str, author_name, c1, c2, c3, c4, c5, c6, c7, c8 = cached
        fields = json.loads(fields_str) if fields_str else ["General Science"]
        subfields = json.loads(subfields_str) if subfields_str else ["General"]
        if not author_name or author_name == "Unknown Author": author_name = pdf_meta_author or "Research Scholar"
        scores_array = [c1, c2, c3, c4, c5, c6, c7, c8]
        drift = calculate_complex_drift(scope_alignment, scores_array) if scope.strip() else "N/A"
        rec = get_recommendation_spectrum(score, drift) if scope.strip() else "N/A"
        return title, author_name, score, logic_score, drift, rec, fields, subfields, {"C1_Originality": c1, "C2_Methodological_Rigor": c2, "C3_Interdisciplinary": c3, "C4_Societal_Impact": c4, "C5_Open_Science_Potential": c5, "C6_Literature_Integration": c6, "C7_Empirical_Density": c7, "C8_Future_Actionability": c8}, file_hash

    try:
        raw_data = evaluate_pdf_text(full_text, PRIMARY_MODEL)
        model_used = PRIMARY_MODEL
    except Exception as e:
        try:
            raw_data = evaluate_pdf_text(full_text, FALLBACK_MODEL)
            model_used = FALLBACK_MODEL
        except Exception as e2:
            st.error(f"Models failed due to rate limits or API errors. {str(e2)}")
            empty_scores = {k: 0.0 for k in ["C1_Originality", "C2_Methodological_Rigor", "C3_Interdisciplinary", "C4_Societal_Impact", "C5_Open_Science_Potential", "C6_Literature_Integration", "C7_Empirical_Density", "C8_Future_Actionability"]}
            return "Extraction Failed", pdf_meta_author or "Research Scholar", 0.0, 0.0, "N/A", "N/A", ["Unknown"], ["Unknown"], empty_scores, "Failed"
        
    # Thread-safe database transactions for the global counter
    with db_lock:
        cursor.execute("BEGIN IMMEDIATE")
        cursor.execute("UPDATE global_eval_counter SET count = count + 1")
        cursor.execute("SELECT count FROM global_eval_counter")
        total_evals = cursor.fetchone()[0]
        
        cursor.execute("SELECT block_height, block_hash, w1, w2, w3, w4, w5, w6, w7, w8 FROM blockchain_por_weights ORDER BY block_height DESC LIMIT 1")
        epoch_data = cursor.fetchone()
        conn.commit()
    
    block_height = epoch_data[0]
    previous_hash = epoch_data[1]
    old_weights = epoch_data[2:]
    
    variables = raw_data.get("variables", {})
    scores_dict = compute_formulaic_criteria(variables)
    scores = [scores_dict["C1_Originality"], scores_dict["C2_Methodological_Rigor"], scores_dict["C3_Interdisciplinary"], scores_dict["C4_Societal_Impact"], scores_dict["C5_Open_Science_Potential"], scores_dict["C6_Literature_Integration"], scores_dict["C7_Empirical_Density"], scores_dict["C8_Future_Actionability"]]
    
    logic_vars = raw_data.get("logic_analysis", {})
    logic_integrity = compute_logical_integrity(logic_vars)

    if total_evals % EPOCH_BLOCK_SIZE == 0:
        new_weights = calculate_model_driven_weights(old_weights, scores, model_used, block_height)
        timestamp = datetime.now().isoformat()
        val_node, block_hash = validate_block_por(block_height + 1, new_weights, timestamp, previous_hash, file_hash, model_used)
        with db_lock:
            cursor.execute('''INSERT INTO blockchain_por_weights (w1, w2, w3, w4, w5, w6, w7, w8, timestamp, previous_hash, validator_node, block_hash, eval_hash, model_used) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                           (*new_weights, timestamp, previous_hash, val_node, block_hash, file_hash, model_used))
            conn.commit()
        active_weights = new_weights
    else:
        active_weights = old_weights

    title = raw_data.get("Extracted_Title", filename)
    extracted_author = raw_data.get("Extracted_Author", "").strip()
    if not extracted_author or extracted_author.lower() in ["unknown", "unknown author", "none", "n/a"]: extracted_author = pdf_meta_author or "Research Scholar"

    fields = raw_data.get("fields", ["General Science"])
    subfields = raw_data.get("subfields", ["General"])
    
    raw_final_score = float(np.dot(scores, active_weights)) / 8.0
    final_score = float(raw_final_score * (0.7 + (logic_integrity / 333.3)))
    drift = calculate_complex_drift(scope_alignment, scores) if scope.strip() else "N/A"
    rec = get_recommendation_spectrum(final_score, drift) if scope.strip() else "N/A"
    
    with db_lock:
        cursor.execute('''INSERT INTO papers_assessment (eval_hash, user_id, title, filename, scope, c1, c2, c3, c4, c5, c6, c7, c8, logic_score, scope_alignment, subfields, fields, author_name, final_score, timestamp) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                       (file_hash, user_id, title, filename, scope, *scores, logic_integrity, scope_alignment, json.dumps(subfields), json.dumps(fields), extracted_author, final_score, datetime.now().isoformat()))
        conn.commit()
    
    return title, extracted_author, final_score, logic_integrity, drift, rec, fields, subfields, scores_dict, file_hash

# --- 4. TOPOLOGICAL MAPPING (INTERACTIVE PYVIS NETWORK) ---
def generate_interactive_bubble_chart(user_id, target_author=None):
    cursor = conn.cursor()
    if target_author and target_author != "All Authors":
        cursor.execute("SELECT fields, subfields, final_score FROM papers_assessment WHERE user_id=? AND author_name LIKE ?", (user_id, f"%{target_author}%"))
    else:
        cursor.execute("SELECT fields, subfields, final_score FROM papers_assessment WHERE user_id=?", (user_id,))
        
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
    unique_topics = topic_counts['topic'].unique()
    
    def get_color(i, n):
        h, s, v = i/n if n > 0 else 0, 0.7, 0.9
        rgb = colorsys.hsv_to_rgb(h, s, v)
        return '#%02x%02x%02x' % tuple(int(x * 255) for x in rgb)
    
    color_map = {topic: get_color(i, len(unique_topics)) for i, topic in enumerate(unique_topics)}
    
    net = Network(height='600px', width='100%', bgcolor='#ffffff', font_color='#2c3e50', notebook=False)
    net.set_options('{"physics": {"barnesHut": {"gravitationalConstant": -1000, "springLength": 100}, "stabilization": {"enabled": true, "iterations": 500}}}')
    
    for _, row in topic_counts.iterrows():
        net.add_node(n_id=row['topic'], label=' ', title=f"Topic: {row['topic']} | Weight: {row['weight']}", size=30 + (row['weight'] * 2.5), physics=True, color=color_map[row['topic']])
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp_file:
        net.save_graph(tmp_file.name)
        with open(tmp_file.name, 'r', encoding='utf-8') as f: html_string = f.read()
    os.remove(tmp_file.name)

    html_string = html_string.replace('mynetwork', f"pi_network_{int(time.time() * 1000)}")
    table_html = "<style>.table-big { width: 100%; font-size: 14px; border-collapse: collapse; margin-top: 10px; font-family: sans-serif; } .table-big th { background-color: #2c3e50; color: white; padding: 10px; text-align: left; } .table-big td { border-bottom: 1px solid #ddd; padding: 8px; vertical-align: middle; } .color-box { width: 18px; height: 18px; display: inline-block; border-radius: 3px; border: 1px solid #ccc; margin: 0 auto;} .legend-container { max-height: 550px; overflow-y: auto; border: 1px solid #eee; }</style>"
    table_html += "<div class='legend-container'><table class='table-big'><thead><tr><th style='width: 25%; text-align: center;'>Color</th><th>Topic</th></tr></thead><tbody>"
    for _, row in topic_counts.sort_values(by="weight", ascending=False).iterrows():
        table_html += f"<tr><td style='text-align: center;'><div class='color-box' style='background-color:{color_map[row['topic']]};'></div></td><td>{row['topic']}</td></tr>"
    table_html += "</tbody></table></div>"
    
    return html_string, table_html

# --- 5. NEURAL NETWORK CLASSES ---
class PiBlockchainDataset(Dataset):
    def __init__(self, data_matrix, lookback):
        self.data = data_matrix
        self.lookback = lookback
    def __len__(self): return len(self.data) - self.lookback
    def __getitem__(self, idx): return torch.tensor(self.data[idx : idx + self.lookback], dtype=torch.float32), torch.tensor(self.data[idx + self.lookback], dtype=torch.float32)

class PiBrainLSTM(nn.Module):
    def __init__(self, input_size=8, hidden_layer_size=32, output_size=8):
        super(PiBrainLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Sequential(nn.Linear(hidden_layer_size, 16), nn.ReLU(), nn.Linear(16, output_size))
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return torch.softmax(self.linear(lstm_out[:, -1, :]), dim=-1) * 8.0

@st.cache_data(show_spinner="Training PoR Meta-Model in background...")
def train_and_predict_lstm(weight_data, lookback_window, current_block_count):
    dataset = PiBlockchainDataset(weight_data, lookback_window)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    model = PiBrainLSTM()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.MSELoss()
    
    model.train()
    for _ in range(200):
        for seq, target in dataloader:
            optimizer.zero_grad()
            loss = loss_function(model(seq), target)
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        next_weights = model(torch.tensor(weight_data[-lookback_window:], dtype=torch.float32).unsqueeze(0)).squeeze().numpy()
    return next_weights, weight_data[-1]

# --- 6. USER INTERFACE (STREAMLIT) ---
st.sidebar.title("System Access")
if 'assessment_update_token' not in st.session_state: st.session_state['assessment_update_token'] = time.time()
if 'orcid_id' not in st.session_state:
    st.session_state.orcid_id, st.session_state.orcid_name, st.session_state.is_authenticated = "0000-0000-0000-0000", "", False

if not st.session_state.is_authenticated:
    st.sidebar.markdown("### Authenticate via ORCID")
    manual_orcid = st.sidebar.text_input("Enter ORCID iD", placeholder="XXXX-XXXX-XXXX-XXXX")
    if st.sidebar.button("🔗 Validate & Connect via ORCID API"):
        clean_orcid = manual_orcid.strip()
        if re.match(r'^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$', clean_orcid):
            with st.sidebar.status("Connecting to Registry..."): is_valid, user_name = verify_orcid_live(clean_orcid)
            if is_valid:
                st.session_state.orcid_id, st.session_state.orcid_name, st.session_state.is_authenticated = clean_orcid, user_name, True
                st.rerun()
            else: st.sidebar.error(user_name)
        else: st.sidebar.error("Invalid format.")
else:
    st.sidebar.success("Securely Connected")
    st.sidebar.markdown(f"**Researcher:** {st.session_state.orcid_name}\n**ORCID iD:** `{st.session_state.orcid_id}`")
    if st.sidebar.button("Disconnect Session"):
        st.session_state.is_authenticated, st.session_state.orcid_name = False, ""
        st.rerun()

current_user = st.session_state.orcid_id
st.title("π-Index Assessment Engine")

tab1, tab2, tab3, tab4 = st.tabs(["Batch Assessment", "Scope Cartography", "Active Epoch Constants", "π-Brain Neural Network"])

with tab1:
    research_scope = st.text_input("Define your specific Research Topic / Scope (Optional)")
    uploaded_files = st.file_uploader("Upload Academic Papers (PDFs)", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Run Batch Assessment", type="primary") and uploaded_files:
        results_list, progress_bar, status_text = [], st.progress(0), st.empty()
        
        for i, file in enumerate(uploaded_files):
            status_text.text(f"Analyzing {i+1} of {len(uploaded_files)}: {file.name}...")
            title, author_name, score, logic_integrity, drift, rec, fields, subfields, scores_dict, eval_hash = process_single_pdf(file.read(), file.name, research_scope, current_user)
            
            record = {"No.": i + 1, "File Name": file.name, "Primary Author": author_name, "Logic Integrity (%)": round(logic_integrity, 1), "π-Index (0-100)": round(score, 1)}
            if research_scope.strip(): record.update({"Topic": research_scope, "Recommendation Spectrum": rec, "Scope Drift %": round(drift, 1) if drift != "N/A" else "N/A"})
            record.update({f"C{j}": round(scores_dict.get(k, 0.0), 1) for j, k in enumerate(scores_dict.keys(), 1)})
            
            results_list.append(record)
            progress_bar.progress((i + 1) / len(uploaded_files))
            
        status_text.success("Batch processing complete!")
        st.session_state['latest_assessment_results'] = pd.DataFrame(results_list)
        st.session_state['assessment_update_token'] = time.time()
        
    if 'latest_assessment_results' in st.session_state: st.dataframe(st.session_state['latest_assessment_results'], use_container_width=True, hide_index=True)

with tab2:
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT author_name FROM papers_assessment WHERE user_id=?", (current_user,))
    user_authors = sorted(list(set([row[0].strip() for row in cursor.fetchall() if row[0] and row[0].strip()])))
    
    selected_author = None
    if user_authors:
        filter_choice = st.selectbox("Filter Cartography by Primary Author:", ["All Authors"] + user_authors, key=f"author_filter_dropdown_{st.session_state['assessment_update_token']}")
        if filter_choice != "All Authors": selected_author = filter_choice

    interactive_html, table_html = generate_interactive_bubble_chart(current_user, target_author=selected_author)
    if interactive_html:
        col1, col2 = st.columns([3, 1])
        with col1: components.html(interactive_html, height=620, scrolling=True)
        with col2:
            st.markdown("### Legend")
            st.markdown(table_html, unsafe_allow_html=True)
    else: st.info("Awaiting sufficient data for this selection.")

with tab3:
    cursor = conn.cursor()
    cursor.execute("SELECT block_height, w1, w2, w3, w4, w5, w6, w7, w8, model_used, eval_hash, block_hash FROM blockchain_por_weights ORDER BY block_height DESC LIMIT 1")
    epoch_data = cursor.fetchone()
    if epoch_data:
        st.markdown(f"**Epoch Block:** `{epoch_data[0]}` | **Last Model:** `{epoch_data[9]}` | **Block Hash:** `{epoch_data[11]}`")
        cols = st.columns(4)
        for i, col in enumerate(cols * 2):
            if i < 8: 
                col.markdown(f"**C{i+1}**")
                col.markdown(f"<h3 style='margin-top:0px;'>{epoch_data[1:9][i]:.6f}</h3>", unsafe_allow_html=True)

with tab4:
    cursor = conn.cursor()
    cursor.execute("SELECT w1, w2, w3, w4, w5, w6, w7, w8 FROM blockchain_por_weights ORDER BY block_height ASC")
    historical_rows = cursor.fetchall()
    lookback_window = 5
    
    if len(historical_rows) < lookback_window + 2:
        st.warning(f"Not enough blockchain data to train the meta-model. You need at least {lookback_window + 2} blocks.")
    else:
        weight_data = np.array(historical_rows, dtype=np.float32)
        next_weights, curr_weights = train_and_predict_lstm(weight_data, lookback_window, len(historical_rows))
        
        st.markdown("### Next Epoch Prediction vs. Current Epoch")
        df_compare = pd.DataFrame({"Current Active Weights": curr_weights, "Predicted Next Epoch": next_weights}, index=[f"C{i+1}" for i in range(8)])
        st.bar_chart(df_compare, height=400)
