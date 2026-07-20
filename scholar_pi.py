
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
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import fitz  # PyMuPDF
from pyvis.network import Network
from groq import Groq

# --- Machine Learning Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --- 1. CONFIGURATION & ENVIRONMENT SETUP ---
st.set_page_config(page_title="π-Index Assessment Engine", layout="wide")

# Define which LLM models to use for the text analysis
PRIMARY_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"
MAX_TEXT_TOKENS = 12000 # Keep the text size small enough to avoid API limits
SEED_NUMBER = 42

# We track papers in blocks, like a mini-blockchain
EPOCH_BLOCK_SIZE = 5

# Set up the folder and database file
BASE_DIR = os.path.abspath('./Scientometric_Pi_Index')
os.makedirs(BASE_DIR, exist_ok=True)
DB_PATH = os.path.join(BASE_DIR, 'pi_index_main.db')

# Securely grab the API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    st.error("API Key not found! Please configure your environment variables or Streamlit Secrets.")
    st.stop()
client = Groq(api_key=GROQ_API_KEY)

# --- UTILITY FUNCTIONS ---

def verify_orcid_live(orcid_id):
    """Check if the provided ORCID iD actually exists on the public registry."""
    try:
        url = f"https://pub.orcid.org/v3.0/{orcid_id}/person"
        headers = {"Accept": "application/json"}
        response = requests.get(url, headers=headers, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            name_data = data.get('name', {})
            
            # Extract first and last name if they exist
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
    """Gradually reveal more digits of Pi based on how many blocks we have processed."""
    pi_str = "3.141592653589793238462643383279502884197169399375105820974944592"
    length = min(block_height + 3, len(pi_str))
    return float(pi_str[:length])

def validate_block_por(block_index, weights, timestamp, previous_hash, eval_hash, model_used):
    """Create a unique hash signature for our blockchain records."""
    validator_node = "Validator_Pi_" + hashlib.md5(str(time.time()).encode()).hexdigest()[:6]
    data_string = f"{block_index}{weights}{timestamp}{previous_hash}{validator_node}{eval_hash}{model_used}"
    block_hash = hashlib.sha256(data_string.encode('utf-8')).hexdigest()
    return validator_node, block_hash

# --- 2. DATABASE INITIALIZATION ---

@st.cache_resource
def init_system():
    """Set up the SQLite database tables if they don't exist yet."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()
    
    # Main table for storing evaluated papers
    cursor.execute('''CREATE TABLE IF NOT EXISTS papers_assessment 
                      (eval_hash TEXT PRIMARY KEY, user_id TEXT, title TEXT, filename TEXT, scope TEXT,
                       c1 REAL, c2 REAL, c3 REAL, c4 REAL, 
                       c5 REAL, c6 REAL, c7 REAL, c8 REAL, 
                       scope_alignment REAL, logic_score REAL,
                       subfields TEXT, fields TEXT, author_name TEXT, final_score REAL, timestamp DATETIME)''')
                       
    # Try adding columns in case the database is from an older version
    try: cursor.execute("ALTER TABLE papers_assessment ADD COLUMN logic_score REAL DEFAULT 0.0")
    except sqlite3.OperationalError: pass 

    try: cursor.execute("ALTER TABLE papers_assessment ADD COLUMN author_name TEXT DEFAULT 'Unknown Author'")
    except sqlite3.OperationalError: pass 
        
    # Table for storing the weighting rules (the "blockchain")
    cursor.execute('''CREATE TABLE IF NOT EXISTS blockchain_por_weights 
                      (block_height INTEGER PRIMARY KEY AUTOINCREMENT, 
                       w1 REAL, w2 REAL, w3 REAL, w4 REAL, 
                       w5 REAL, w6 REAL, w7 REAL, w8 REAL, 
                       timestamp DATETIME, previous_hash TEXT, 
                       validator_node TEXT, block_hash TEXT, eval_hash TEXT, model_used TEXT)''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS global_eval_counter (count INTEGER)''')
    
    # Create the very first "Genesis" block if the blockchain is empty
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
                       
    # Initialize the total paper counter
    cursor.execute("SELECT count FROM global_eval_counter")
    if not cursor.fetchone():
        cursor.execute("INSERT INTO global_eval_counter (count) VALUES (0)")
        
    conn.commit()
    return conn

# Connect to database globally
conn = init_system()

# --- 3. MATHEMATICAL EVALUATION ENGINE ---

def calculate_model_driven_weights(old_weights, scores, model_name, block_height):
    """Adjust the grading weights slightly based on the average scores of the recent batch."""
    if "70b" in model_name:
        model_version = 3.3
        model_size = 70.0
    else:
        model_version = 3.1
        model_size = 8.0
        
    pi_accuracy = get_pi_float(block_height)
    delta_models = abs((3.3 * 70.0) - (3.1 * 8.0)) 
    
    mean_score = np.mean(scores)
    
    # Calculate new weights
    new_weights = []
    for i, old_w in enumerate(old_weights):
        # Stretch the score to exaggerate differences
        stretched_score = max(1.0, min(100.0, mean_score + (scores[i] - mean_score) * 3.0))
        
        # Calculate how much to shift the weight based on our formula
        weight_shift = ((model_version * model_size) / (delta_models * pi_accuracy)) * ((stretched_score / 100.0) ** 2)
        
        # Blend the old weight with the new shift (85% old, 15% new)
        w_new = old_w * 0.85 + (1.0 + weight_shift * 0.15) * 0.15
        new_weights.append(w_new)
        
    # Make sure all weights sum up to exactly 8.0
    sum_of_weights = sum(new_weights)
    final_normalized_weights = []
    for w in new_weights:
        normalized = round((w / sum_of_weights) * 8.0, 6)
        final_normalized_weights.append(normalized)
        
    return final_normalized_weights

def compute_logical_integrity(extracted_logic_vars):
    """Check if the paper's conclusions jump too far ahead of its evidence."""
    evidence = extracted_logic_vars.get('Evidence_Strength', 0.5)
    conclusion_reach = extracted_logic_vars.get('Conclusion_Reach', 0.5)
    jumps = extracted_logic_vars.get('Logical_Jumps', 0.5)
    premise = extracted_logic_vars.get('Premise_Validity', 0.5)
    
    # Calculate the gap between evidence and conclusion
    logic_gap = max(0.0, conclusion_reach - evidence)
    
    # Calculate penalty
    logic_score = (premise * evidence) * np.exp(-(logic_gap * 2.0 + jumps * 1.5)) * 100
    
    # Ensure it stays within 0-100 range
    return max(0.0, min(100.0, logic_score))

def compute_formulaic_criteria(vars_dict):
    """Calculate the 8 main grading criteria based on the variables extracted by the LLM."""
    scores = {}
    
    # 1. Originality
    c1_raw = ((vars_dict.get('H_novel', 0.5) * vars_dict.get('K_epistemic', 0.5)) / (vars_dict.get('zeta', 0.5) * vars_dict.get('I_existing', 0.5) + 0.1)) * 60
    scores["C1_Originality"] = min(100.0, max(0.0, c1_raw))
    
    # 2. Methodological Rigor
    rigor_matrix = max(0.0, 1.0 - (vars_dict.get('Sigma_error', 0.2) / (vars_dict.get('mu_signal', 0.8) + 0.1)))
    c2_raw = rigor_matrix * vars_dict.get('rho_k', 0.5) * math.gamma(1.5) * 140
    scores["C2_Methodological_Rigor"] = min(100.0, max(0.0, c2_raw))
    
    # 3. Interdisciplinary
    p_disc = np.array(vars_dict.get('p_disciplines', [1.0]))
    p_disc = p_disc / (p_disc.sum() + 1e-9) # Normalize array
    renyi_entropy = -np.log(np.sum(p_disc**2) + 1e-9) 
    c3_raw = (renyi_entropy + vars_dict.get('bridge_capacity', 0.5)) * 55
    scores["C3_Interdisciplinary"] = min(100.0, max(0.0, c3_raw))
    
    # 4. Societal Impact
    gamma_q = math.gamma(max(0.1, vars_dict.get('q_fractional', 1.5)))
    c4_raw = (1.0 / gamma_q) * vars_dict.get('Utility_vector', 0.5) * np.exp(-vars_dict.get('decay_rate', 0.5)) * 150
    scores["C4_Societal_Impact"] = min(100.0, max(0.0, c4_raw))
    
    # 5. Open Science Potential
    c5_raw = ((0.7 * vars_dict.get('D_open', 0.1)) + (0.3 * vars_dict.get('J_code', 0.1))) * vars_dict.get('P_FAIR', 0.1) * 180
    scores["C5_Open_Science_Potential"] = min(100.0, max(0.0, c5_raw))
    
    # 6. Literature Integration
    c6_raw = np.exp(-1.5 * vars_dict.get('d_g_distance', 0.5)) * vars_dict.get('R_xi', 0.5) * vars_dict.get('PR_xi', 0.5) * 180
    scores["C6_Literature_Integration"] = min(100.0, max(0.0, c6_raw))
    
    # 7. Empirical Density
    density_inner = (vars_dict.get('I_Fisher', 0.5) * vars_dict.get('KL_divergence', 0.5)) / (vars_dict.get('V_baseline', 0.5) * vars_dict.get('omega_data', 0.5) + 0.1)
    c7_raw = np.tanh(density_inner) * vars_dict.get('sum_lambda_kappa', 1.0) * 80
    scores["C7_Empirical_Density"] = min(100.0, max(0.0, c7_raw))
    
    # 8. Future Actionability
    eta = vars_dict.get('eta_steps', 2.0)
    lambda_lyapunov = vars_dict.get('Lambda_Lyapunov', 0.5)
    c8_raw = (1.0 / (1.0 + np.exp(-(eta - (lambda_lyapunov * 5))))) * 100
    scores["C8_Future_Actionability"] = min(100.0, max(0.0, c8_raw))
    
    # Round everything neatly
    for key in scores:
        scores[key] = round(scores[key], 2)
        
    return scores

def evaluate_scope_alignment(text, scope, model, text_limit):
    """Ask the LLM how well the paper matches the user's defined research scope."""
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
        result = json.loads(response.choices[0].message.content)
        return float(result.get("Scope_Alignment", 0.0))
    except Exception:
        return 0.0

def evaluate_pdf_text(text, model, text_limit):
    """Send the PDF text to the LLM to extract the core mathematical proxy variables."""
    if len(text) > text_limit:
        text = text[:text_limit]

    prompt = f"""You are the theoretical parser for the π-Index Assessment Engine.
Instead of assigning arbitrary scores, you must read the academic paper and extract the underlying mathematical proxy variables based purely on the document's objective scientific merit.

CRITICAL INSTRUCTION - AUTHOR EXTRACTION:
Carefully look at the first page of the text to find the actual names of the human authors written below the title. Look for names formatted like "Firstname Lastname" or "Author Name". Do NOT use the file name, do NOT use university names, do NOT use journal names, and do NOT write "Unknown Author". If multiple authors exist, provide the primary/first author name followed by "et al." (e.g. "Jane Doe et al.").

1. Extracted Metadata:
- `Extracted_Title`: The full title of the paper.
- `Extracted_Author`: The primary author name(s).

2. Extracted Variables (all values must be floats between 0.0 and 1.0, unless specified):
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

3. Adversarial Logic Mapping:
Identify logical structural flaws and gaps in reasoning:
- `Evidence_Strength`: (0.1 = Anecdotal/Weak, 0.9 = Robust/Repetitive).
- `Conclusion_Reach`: (0.1 = Conservative/Supported, 0.9 = Wild/Unsupported).
- `Logical_Jumps`: (0.1 = Highly logical flow, 0.9 = Major non-sequiturs).
- `Premise_Validity`: (0.1 = Questionable assumptions, 0.9 = Solid definitions).

Return ONLY a valid JSON object matching exactly this structure:
{{
    "Extracted_Title": "Title", 
    "Extracted_Author": "Author Name",
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
    """Calculate how far the paper drifts from the user's specific research scope."""
    average_score = np.mean(scores)
    standard_deviation = np.std(scores)
    
    # Calculate the gap
    alignment_gap = (100.0 - alignment) / 100.0
    
    # Apply our formula to find the final drift metric
    drift_metric = 100.0 * (1.0 - np.exp(-3.0 * (alignment_gap ** 1.5) * (1.0 + (standard_deviation / 100.0)) / (0.1 + (average_score / 100.0))))
    return float(max(0.0, min(100.0, drift_metric)))

def get_recommendation_spectrum(score, drift):
    """Translate the numerical score and drift into a plain-English recommendation."""
    synergy = score * (1.0 - (drift / 100.0)**1.5)
    
    if synergy >= 85: return "Tier I: Core Paradigm (Optimal Synergy)"
    elif synergy >= 70: return "Tier II: Highly Aligned Framework"
    elif synergy >= 55: return "Tier III: Moderately Synergistic"
    elif synergy >= 40: return "Tier IV: Tangential Relevance"
    elif synergy >= 25: return "Tier V: Epistemic Divergence"
    else: return "Tier VI: Orthogonal / Unrelated Noise"

def process_single_pdf(file_bytes, filename, scope, user_id):
    """Main workflow to process a single PDF file and grade it."""
    file_hash = hashlib.sha256(file_bytes).hexdigest() 
    cursor = conn.cursor()
    
    # Check if we already evaluated this exact file
    cursor.execute("SELECT final_score, logic_score, title, fields, subfields, author_name, c1, c2, c3, c4, c5, c6, c7, c8 FROM papers_assessment WHERE eval_hash=? AND user_id=?", (file_hash, user_id))
    cached_result = cursor.fetchone()
    
    # Open the PDF and read its text
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pdf_meta_author = doc.metadata.get("author", "").strip()
    
    # Gather all text from all pages
    full_text_pages = []
    for page in doc:
        full_text_pages.append(page.get_text())
    full_text = " ".join(full_text_pages) 
    
    # Check alignment if the user provided a scope
    if scope.strip():
        scope_alignment = evaluate_scope_alignment(full_text, scope, FALLBACK_MODEL, MAX_TEXT_TOKENS)
    else:
        scope_alignment = 0.0

    # If we found it in the database, return the saved data instead of calling the AI again
    if cached_result:
        score, logic_score, title, fields_str, subfields_str, author_name, c1, c2, c3, c4, c5, c6, c7, c8 = cached_result
        
        fields = json.loads(fields_str) if fields_str else ["General Science"]
        subfields = json.loads(subfields_str) if subfields_str else ["General"]
        
        if not author_name or author_name == "Unknown Author" or author_name == os.path.splitext(filename)[0]:
            author_name = pdf_meta_author or "Research Scholar"

        scores_array = [c1, c2, c3, c4, c5, c6, c7, c8]
        drift = calculate_complex_drift(scope_alignment, scores_array) if scope.strip() else "N/A"
        rec = get_recommendation_spectrum(score, drift) if scope.strip() else "N/A"
        scores_dict = {"C1_Originality": c1, "C2_Methodological_Rigor": c2, "C3_Interdisciplinary": c3, "C4_Societal_Impact": c4, "C5_Open_Science_Potential": c5, "C6_Literature_Integration": c6, "C7_Empirical_Density": c7, "C8_Future_Actionability": c8}
        
        return title, author_name, score, logic_score, drift, rec, fields, subfields, scores_dict, file_hash

    # If it's a new file, analyze it with the LLM
    try:
        raw_data = evaluate_pdf_text(full_text, PRIMARY_MODEL, MAX_TEXT_TOKENS)
        model_used = PRIMARY_MODEL
    except Exception as e:
        st.warning(f"Primary model hit a limit. Trying fallback model ({FALLBACK_MODEL})...")
        try:
            reduced_limit = MAX_TEXT_TOKENS // 2 if 'limit' in str(e).lower() or '413' in str(e) else MAX_TEXT_TOKENS
            raw_data = evaluate_pdf_text(full_text, FALLBACK_MODEL, reduced_limit)
            model_used = FALLBACK_MODEL
        except Exception as e2:
            st.error(f"Both models failed. API Error: {str(e2)}")
            fallback_author = pdf_meta_author or "Research Scholar"
            empty_scores = {k: 0.0 for k in ["C1_Originality", "C2_Methodological_Rigor", "C3_Interdisciplinary", "C4_Societal_Impact", "C5_Open_Science_Potential", "C6_Literature_Integration", "C7_Empirical_Density", "C8_Future_Actionability"]}
            return "Extraction Failed", fallback_author, 0.0, 0.0, "N/A", "N/A", ["Unknown"], ["Unknown"], empty_scores, "Failed"
        
    # Update global counters
    cursor.execute("UPDATE global_eval_counter SET count = count + 1")
    cursor.execute("SELECT count FROM global_eval_counter")
    total_evals = cursor.fetchone()[0]
        
    # Fetch the most recent epoch weights from our blockchain table
    cursor.execute("SELECT block_height, block_hash, w1, w2, w3, w4, w5, w6, w7, w8 FROM blockchain_por_weights ORDER BY block_height DESC LIMIT 1")
    epoch_data = cursor.fetchone()
    
    block_height = epoch_data[0]
    previous_hash = epoch_data[1]
    old_weights = epoch_data[2:]
    
    # Calculate scores from the raw AI data
    variables = raw_data.get("variables", {})
    scores_dict = compute_formulaic_criteria(variables)
    
    # Keep scores in order for array math
    scores = [
        scores_dict["C1_Originality"], scores_dict["C2_Methodological_Rigor"], 
        scores_dict["C3_Interdisciplinary"], scores_dict["C4_Societal_Impact"], 
        scores_dict["C5_Open_Science_Potential"], scores_dict["C6_Literature_Integration"], 
        scores_dict["C7_Empirical_Density"], scores_dict["C8_Future_Actionability"]
    ]
    
    logic_vars = raw_data.get("logic_analysis", {})
    logic_integrity = compute_logical_integrity(logic_vars)

    # Check if we reached a new Epoch. If so, update the weights!
    if total_evals % EPOCH_BLOCK_SIZE == 0:
        new_weights = calculate_model_driven_weights(old_weights, scores, model_used, block_height)
        timestamp = datetime.now().isoformat()
        val_node, block_hash = validate_block_por(block_height + 1, new_weights, timestamp, previous_hash, file_hash, model_used)
        
        # Save new block to database
        cursor.execute('''INSERT INTO blockchain_por_weights (w1, w2, w3, w4, w5, w6, w7, w8, timestamp, previous_hash, validator_node, block_hash, eval_hash, model_used) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                       (*new_weights, timestamp, previous_hash, val_node, block_hash, file_hash, model_used))
        active_weights = new_weights
    else:
        # Not a new epoch yet, keep using old weights
        active_weights = old_weights

    # Clean up names and metadata
    title = raw_data.get("Extracted_Title", filename)
    extracted_author = raw_data.get("Extracted_Author", "").strip()
    
    if not extracted_author or extracted_author.lower() in ["unknown", "unknown author", "none", "n/a"] or extracted_author == os.path.splitext(filename)[0]:
        extracted_author = pdf_meta_author or "Research Scholar"

    fields = raw_data.get("fields", ["General Science"])
    subfields = raw_data.get("subfields", ["General"])
    
    # Calculate Final Score
    raw_final_score = float(np.dot(scores, active_weights)) / 8.0
    final_score = float(raw_final_score * (0.7 + (logic_integrity / 333.3)))
    
    drift = calculate_complex_drift(scope_alignment, scores) if scope.strip() else "N/A"
    rec = get_recommendation_spectrum(final_score, drift) if scope.strip() else "N/A"
    
    # Save the new assessment to the database
    timestamp = datetime.now().isoformat()
    cursor.execute('''INSERT INTO papers_assessment (eval_hash, user_id, title, filename, scope, c1, c2, c3, c4, c5, c6, c7, c8, logic_score, scope_alignment, subfields, fields, author_name, final_score, timestamp) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                   (file_hash, user_id, title, filename, scope, *scores, logic_integrity, scope_alignment, json.dumps(subfields), json.dumps(fields), extracted_author, final_score, timestamp))
    conn.commit()
    
    return title, extracted_author, final_score, logic_integrity, drift, rec, fields, subfields, scores_dict, file_hash


# --- 4. TOPOLOGICAL MAPPING (INTERACTIVE PYVIS NETWORK) ---

def generate_interactive_bubble_chart(user_id, target_author=None):
    """Generate the HTML code for the PyVis interactive network map."""
    cursor = conn.cursor()
    
    # Fix: Use LIKE for a more robust match in case there are invisible spaces stored in the DB
    if target_author and target_author != "All Authors":
        cursor.execute("SELECT fields, subfields, final_score FROM papers_assessment WHERE user_id=? AND author_name LIKE ?", (user_id, f"%{target_author}%"))
    else:
        cursor.execute("SELECT fields, subfields, final_score FROM papers_assessment WHERE user_id=?", (user_id,))
        
    data = cursor.fetchall()
    
    html_string, table_html = "", ""
    if not data: 
        return html_string, table_html
    
    # Compile a list of all topics found in the user's papers
    all_topics = []
    for fields_json, subfields_json, final_score in data:
        try:
            fields = [f.title().strip() for f in json.loads(fields_json)]
            subfields = [s.title().strip() for s in json.loads(subfields_json)]
            score = float(final_score) if final_score else 50.0
            
            for f in fields: 
                all_topics.append({'topic': f, 'weight': score})
            for s in subfields: 
                all_topics.append({'topic': s, 'weight': score})
        except: 
            continue
            
    if not all_topics: 
        return html_string, table_html
    
    # Group by topic and sum the weights
    df_topics = pd.DataFrame(all_topics)
    topic_counts = df_topics.groupby(['topic'])['weight'].sum().reset_index(name='weight')
    
    if topic_counts.empty: 
        return html_string, table_html
        
    unique_topics = topic_counts['topic'].unique()
    
    # Generate unique colors for each topic
    def get_color(i, n):
        h, s, v = i/n if n > 0 else 0, 0.7, 0.9
        rgb = colorsys.hsv_to_rgb(h, s, v)
        return '#%02x%02x%02x' % tuple(int(x * 255) for x in rgb)
    
    color_map = {topic: get_color(i, len(unique_topics)) for i, topic in enumerate(unique_topics)}
    
    # Build the network graph
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
    
    # Save network to a temporary file, read the HTML, and clean up
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp_file:
        net.save_graph(tmp_file.name)
        with open(tmp_file.name, 'r', encoding='utf-8') as f:
            html_string = f.read()
    
    os.remove(tmp_file.name)

    # ----------------------------------------------------------------------
    # THE ULTIMATE CACHE BUSTER FOR STREAMLIT IFRAMES
    # ----------------------------------------------------------------------
    # PyVis hardcodes the CSS ID of the graph container as 'mynetwork'. 
    # By replacing this with a randomly generated ID on every single click,
    # the browser DOM physically changes, forcing Streamlit to completely 
    # wipe the old iframe and draw the updated data.
    unique_network_id = f"pi_network_{int(time.time() * 1000)}"
    html_string = html_string.replace('mynetwork', unique_network_id)

    # Build an HTML legend table
    table_html = "<style>.table-big { width: 100%; font-size: 14px; border-collapse: collapse; margin-top: 10px; font-family: sans-serif; } .table-big th { background-color: #2c3e50; color: white; padding: 10px; text-align: left; } .table-big td { border-bottom: 1px solid #ddd; padding: 8px; vertical-align: middle; } .color-box { width: 18px; height: 18px; display: inline-block; border-radius: 3px; border: 1px solid #ccc; margin: 0 auto;} .legend-container { max-height: 550px; overflow-y: auto; border: 1px solid #eee; }</style>"
    table_html += "<div class='legend-container'><table class='table-big'><thead><tr><th style='width: 25%; text-align: center;'>Color</th><th>Topic</th></tr></thead><tbody>"
    
    for _, row in topic_counts.sort_values(by="weight", ascending=False).iterrows():
        table_html += f"<tr><td style='text-align: center;'><div class='color-box' style='background-color:{color_map[row['topic']]};'></div></td><td>{row['topic']}</td></tr>"
        
    table_html += "</tbody></table></div>"
    
    return html_string, table_html

# --- 5. NEURAL NETWORK CLASSES ---

class PiBlockchainDataset(Dataset):
    """Formats the blockchain weight history into a time-series dataset for training."""
    def __init__(self, data_matrix, lookback):
        self.data = data_matrix
        self.lookback = lookback

    def __len__(self):
        return len(self.data) - self.lookback

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.lookback]
        y = self.data[idx + self.lookback]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class PiBrainLSTM(nn.Module):
    """A small LSTM neural network to predict how the grading weights will shift next."""
    def __init__(self, input_size=8, hidden_layer_size=32, output_size=8):
        super(PiBrainLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(hidden_layer_size, 16),
            nn.ReLU(),
            nn.Linear(16, output_size)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :]
        predictions = self.linear(last_time_step)
        
        # Softmax ensures they sum to 1.0, then we scale to our 8.0 budget
        normalized_predictions = torch.softmax(predictions, dim=-1) * 8.0
        return normalized_predictions

# --- 6. USER INTERFACE (STREAMLIT) ---

st.sidebar.title("System Access")

# Track if the user is logged in
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
        
        # Make sure format matches XXXX-XXXX-XXXX-XXXX
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
    st.sidebar.success("Securely Connected")
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

# Theoretical Formulas Expander
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

# --- APP TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["Batch Assessment", "Scope Cartography", "Active Epoch Constants", "π-Brain Neural Network"])

with tab1:
    research_scope = st.text_input("Define your specific Research Topic / Scope (Optional)", placeholder="e.g., Application of deep learning in vascular imaging...")
    uploaded_files = st.file_uploader("Upload Academic Papers (PDFs)", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Run Batch Assessment", type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one academic paper (PDF) to proceed.")
        else:
            results_list = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Loop through all uploaded files and grade them
            for i, file in enumerate(uploaded_files):
                status_text.text(f"Analyzing {i+1} of {len(uploaded_files)}: {file.name}...")
                
                title, author_name, score, logic_integrity, drift, rec, fields, subfields, scores_dict, eval_hash = process_single_pdf(
                    file.read(), file.name, research_scope, current_user
                )
                
                combined_fields = f"Fields: {', '.join(fields)} | Subfields: {', '.join(subfields)}"
                
                record = {
                    "No.": i + 1,
                    "File Name": file.name,
                    "Primary Author": author_name,
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
                    "Eval Hash": eval_hash 
                })
                
                results_list.append(record)
                progress_bar.progress((i + 1) / len(uploaded_files))
                
            status_text.success("Batch processing complete! Here are your results:")
            
            # Show the results directly on the screen so the user can see them!
            results_df = pd.DataFrame(results_list)
            st.dataframe(results_df, use_container_width=True, hide_index=True)
            
            # Trigger meta-learning model to retrain next time Tab 4 is opened
            st.session_state['last_trained_blocks'] = -1

    st.markdown("---")
    st.markdown("### Latest Assessment History")
    
    if st.session_state.is_authenticated:
        cursor = conn.cursor()
        cursor.execute("SELECT title, author_name, scope, final_score, timestamp, eval_hash FROM papers_assessment WHERE user_id=? ORDER BY timestamp DESC LIMIT 20", (current_user,))
        history_data = cursor.fetchall()
        
        if history_data:
            df_hist = pd.DataFrame(history_data, columns=["Paper Title", "Primary Author", "Scope", "π-Index Score", "Date", "Evaluation Hash"])
            st.dataframe(df_hist, use_container_width=True, hide_index=True)
        else:
            st.info("No assessment history found for your account.")
    else:
        st.warning("Please connect your ORCID iD in the sidebar to view your private assessment history.")


with tab2:
    st.subheader("Epistemic Bubbles (Author & Portfolio Cartography)")
    st.write("Filter the topological network map below by the extracted primary author names of your evaluated papers.")
    
    cursor = conn.cursor()
    # Fix: Get all authors, clean up whitespace, remove empty entries, and sort alphabetically
    cursor.execute("SELECT DISTINCT author_name FROM papers_assessment WHERE user_id=?", (current_user,))
    raw_authors = cursor.fetchall()
    
    # Use a set to remove accidental duplicates after stripping spaces
    user_authors = sorted(list(set([row[0].strip() for row in raw_authors if row[0] and row[0].strip()])))
    
    selected_author = None
    if user_authors:
        # Fix: Add a unique key so Streamlit remembers the selection when you switch tabs
        filter_choice = st.selectbox(
            "Filter Cartography by Primary Author:", 
            ["All Authors"] + user_authors,
            key="author_filter_dropdown"
        )
        if filter_choice != "All Authors":
            selected_author = filter_choice

    # Generate the map HTML string
    interactive_html, table_html = generate_interactive_bubble_chart(current_user, target_author=selected_author)
    
    if interactive_html:
        col1, col2 = st.columns([3, 1])
        with col1:
            components.html(interactive_html, height=620, scrolling=True)
        with col2:
            st.markdown("### Legend")
            st.markdown(table_html, unsafe_allow_html=True)
    else: 
        st.info("Awaiting sufficient data for this selection. Upload and process papers to build your map.")


with tab3:
    cursor = conn.cursor()
    cursor.execute("SELECT block_height, w1, w2, w3, w4, w5, w6, w7, w8, model_used, eval_hash, block_hash FROM blockchain_por_weights ORDER BY block_height DESC LIMIT 1")
    epoch_data = cursor.fetchone()
    
    if epoch_data:
        block_height = epoch_data[0]
        weights = epoch_data[1:9]
        model_used = epoch_data[9]
        eval_hash = epoch_data[10]
        block_hash = epoch_data[11]
        
        current_pi_base = get_pi_float(block_height)
        
        cursor.execute("SELECT COUNT(DISTINCT eval_hash) FROM blockchain_por_weights WHERE eval_hash != 'genesis'")
        total_papers_processed = cursor.fetchone()[0]

        st.markdown(f"**Total Papers Processed (Blockchain Ledger):** `{total_papers_processed}` | **Block Size:** `{EPOCH_BLOCK_SIZE} papers/block` | **Epoch Frequency:** `Every {EPOCH_BLOCK_SIZE} evaluations` | **Last Model:** `{model_used}` | **Epoch Block:** `{block_height}` | **Pi Acc:** `{current_pi_base}`")
        
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
        1.  **Locate the Eval Hash:** Copy the Evaluation Hash associated with a paper you assessed.
        2.  **Use the Explorer:** Paste that hash into the PoR Blockchain Explorer input field below.
        3.  **Click "Verify Record":** The system will query the global database to return the exact Weights Matrix alongside the immutable Block Hash.
        """)
        
        explore_col1, explore_col2 = st.columns([3, 1])
        with explore_col1: 
            search_query = st.text_input("Enter Document Evaluation Hash or Block Hash to verify ledger record...")
        with explore_col2: 
            st.write("")
            st.write("")
            search_btn = st.button("Verify Record")
            
        if search_btn and search_query:
            cursor.execute("SELECT * FROM blockchain_por_weights WHERE block_hash=? OR eval_hash=?", (search_query, search_query))
            record = cursor.fetchone()
            
            if record:
                st.success("Valid Block Found on Ledger!")
                st.json({
                    "Block Height": record[0], 
                    "Timestamp": record[9], 
                    "Model Used": record[13], 
                    "Validator Node": record[11], 
                    "Block Hash": record[12], 
                    "Previous Hash": record[10], 
                    "Evaluation Hash (Document)": record[14], 
                    "Weights Matrix (w1..w8)": record[1:9]
                })
            else:
                st.error("No block matching that signature was found on the ledger.")
                
        with st.expander("View Recent Global Ledger Blocks"):
            cursor.execute("SELECT block_height, timestamp, model_used, block_hash FROM blockchain_por_weights ORDER BY block_height DESC LIMIT 10")
            df_blocks = pd.DataFrame(cursor.fetchall(), columns=["Height", "Timestamp", "Model", "Block Hash"])
            st.dataframe(df_blocks, use_container_width=True, hide_index=True)


with tab4:
    st.subheader("π-Brain: Meta-Learning on the PoR Blockchain")
    st.info("""
    How it works: Instead of relying on language models to process the next weight shift, this LSTM Neural Network treats your Proof of Review (PoR) blockchain as a time-series dataset. 
    It trains on the historical evolution of previous epochs and predicts the exact mathematical trajectory of the next unmined epoch.
    """)
    
    cursor = conn.cursor()
    cursor.execute("SELECT w1, w2, w3, w4, w5, w6, w7, w8 FROM blockchain_por_weights ORDER BY block_height ASC")
    historical_rows = cursor.fetchall()
    
    lookback_window = 5
    
    if len(historical_rows) < lookback_window + 2:
        st.warning(f"Not enough blockchain data to train the meta-model. Current blocks: {len(historical_rows)}. You need at least {lookback_window + 2} blocks to form a training sequence. Please assess more papers.")
    else:
        st.success(f"Ready for training. {len(historical_rows)} blocks successfully extracted from the ledger.")
        
        current_block_count = len(historical_rows)
        
        # Only re-train if new blocks have been added since our last training
        if 'last_trained_blocks' not in st.session_state or st.session_state.last_trained_blocks != current_block_count:
            st.markdown("### Training Log (Auto-Running)")
            weight_data = np.array(historical_rows, dtype=np.float32)
            
            dataset = PiBlockchainDataset(weight_data, lookback_window)
            dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
            
            model = PiBrainLSTM()
            loss_function = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            epochs = 200
            model.train()
            
            # Simple training loop
            for epoch in range(epochs):
                total_loss = 0
                for seq, target in dataloader:
                    optimizer.zero_grad()
                    predictions = model(seq)
                    loss = loss_function(predictions, target)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    
                # Update UI occasionally
                if epoch % 10 == 0 or epoch == epochs - 1:
                    avg_loss = total_loss / len(dataloader)
                    status_text.text(f"Training Epoch {epoch}/{epochs} | MSE Loss: {avg_loss:.6f}")
                    progress_bar.progress((epoch + 1) / epochs)
            
            status_text.success("Training Complete!")
            progress_bar.progress(1.0)
            
            # Run the prediction
            model.eval()
            recent_blocks = weight_data[-lookback_window:]
            seq_tensor = torch.tensor(recent_blocks, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                next_weights = model(seq_tensor).squeeze().numpy()
            
            st.session_state.predicted_next_weights = next_weights
            st.session_state.current_weights = weight_data[-1]
            st.session_state.last_trained_blocks = current_block_count
            
        else:
            st.info("Meta-model is cached and up-to-date with the latest blockchain ledger.")

        st.markdown("---")
        st.markdown("### Next Epoch Prediction vs. Current Epoch")
        
        labels = ["C1: Originality", "C2: Method Rigor", "C3: Interdisciplinary", "C4: Societal Impact", "C5: Open Science", "C6: Lit Integration", "C7: Empirical Density", "C8: Actionability"]
        
        df_compare = pd.DataFrame({
            "Current Active Weights": st.session_state.current_weights,
            "Predicted Next Epoch": st.session_state.predicted_next_weights
        }, index=labels)
        
        st.bar_chart(df_compare, height=400)
        
        st.markdown(f"**Mathematical Constraint Check:** Predicted Sum = `{sum(st.session_state.predicted_next_weights):.6f}` / `8.0`")

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray; font-size: 0.8em;'>Framework Author: Ali Vafadar Yengejeh | Università degli Studi di Milano-Bicocca</div>", unsafe_allow_html=True)
