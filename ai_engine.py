import json
import hashlib
import os
from datetime import datetime
import fitz
import numpy as np
import streamlit as st
from groq import Groq

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from config import GROQ_API_KEY, PRIMARY_MODEL, FALLBACK_MODEL, MAX_TEXT_TOKENS, SEED_NUMBER, EPOCH_BLOCK_SIZE
from math_engine import compute_formulaic_criteria, compute_logical_integrity, calculate_model_driven_weights, calculate_complex_drift, get_recommendation_spectrum
from blockchain import validate_block_por, init_system

if not GROQ_API_KEY:
    st.error("API Key not found! Please configure your environment variables or Streamlit Secrets.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)
conn = init_system()

# --- AI API Calls ---
def evaluate_scope_alignment(text, scope, model, text_limit):
    if not scope.strip(): return 0.0
    if len(text) > text_limit: text = text[:text_limit]
        
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
            model=model, temperature=0.0, response_format={"type": "json_object"}
        )
        return float(json.loads(response.choices[0].message.content).get("Scope_Alignment", 0.0))
    except Exception: return 0.0

def evaluate_pdf_text(text, model, text_limit):
    if len(text) > text_limit: text = text[:text_limit]
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
        model=model, temperature=0.0, seed=SEED_NUMBER, response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

def process_single_pdf(file_bytes, filename, scope, user_id):
    """Main workflow to process a single PDF file and grade it."""
    file_hash = hashlib.sha256(file_bytes).hexdigest() 
    cursor = conn.cursor()
    
    cursor.execute("SELECT final_score, logic_score, title, fields, subfields, author_name, c1, c2, c3, c4, c5, c6, c7, c8 FROM papers_assessment WHERE eval_hash=? AND user_id=?", (file_hash, user_id))
    cached_result = cursor.fetchone()
    
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pdf_meta_author = doc.metadata.get("author", "").strip()
    full_text = " ".join([page.get_text() for page in doc])
    
    scope_alignment = evaluate_scope_alignment(full_text, scope, FALLBACK_MODEL, MAX_TEXT_TOKENS) if scope.strip() else 0.0

    if cached_result:
        score, logic_score, title, fields_str, subfields_str, author_name, *c_scores = cached_result
        fields = json.loads(fields_str) if fields_str else ["General Science"]
        subfields = json.loads(subfields_str) if subfields_str else ["General"]
        if not author_name or author_name in ["Unknown Author", os.path.splitext(filename)[0]]:
            author_name = pdf_meta_author or "Research Scholar"

        drift = calculate_complex_drift(scope_alignment, c_scores) if scope.strip() else "N/A"
        rec = get_recommendation_spectrum(score, drift) if scope.strip() else "N/A"
        scores_dict = {f"C{i+1}_Metric": c_scores[i] for i in range(8)} # Simplified dict building for cache
        return title, author_name, score, logic_score, drift, rec, fields, subfields, scores_dict, file_hash

    try:
        raw_data = evaluate_pdf_text(full_text, PRIMARY_MODEL, MAX_TEXT_TOKENS)
        model_used = PRIMARY_MODEL
    except Exception as e:
        st.warning(f"Primary model hit a limit. Trying fallback model...")
        try:
            reduced_limit = MAX_TEXT_TOKENS // 2 if 'limit' in str(e).lower() or '413' in str(e) else MAX_TEXT_TOKENS
            raw_data = evaluate_pdf_text(full_text, FALLBACK_MODEL, reduced_limit)
            model_used = FALLBACK_MODEL
        except Exception as e2:
            empty_scores = {k: 0.0 for k in ["C1_Originality", "C2_Methodological_Rigor", "C3_Interdisciplinary", "C4_Societal_Impact", "C5_Open_Science_Potential", "C6_Literature_Integration", "C7_Empirical_Density", "C8_Future_Actionability"]}
            return "Extraction Failed", pdf_meta_author or "Research Scholar", 0.0, 0.0, "N/A", "N/A", ["Unknown"], ["Unknown"], empty_scores, "Failed"
        
    cursor.execute("UPDATE global_eval_counter SET count = count + 1")
    cursor.execute("SELECT count FROM global_eval_counter")
    total_evals = cursor.fetchone()[0]
        
    cursor.execute("SELECT block_height, block_hash, w1, w2, w3, w4, w5, w6, w7, w8 FROM blockchain_por_weights ORDER BY block_height DESC LIMIT 1")
    epoch_data = cursor.fetchone()
    
    block_height, previous_hash, old_weights = epoch_data[0], epoch_data[1], epoch_data[2:]
    
    variables = raw_data.get("variables", {})
    scores_dict = compute_formulaic_criteria(variables)
    scores = [scores_dict[k] for k in ["C1_Originality", "C2_Methodological_Rigor", "C3_Interdisciplinary", "C4_Societal_Impact", "C5_Open_Science_Potential", "C6_Literature_Integration", "C7_Empirical_Density", "C8_Future_Actionability"]]
    
    logic_integrity = compute_logical_integrity(raw_data.get("logic_analysis", {}))

    if total_evals % EPOCH_BLOCK_SIZE == 0:
        active_weights = calculate_model_driven_weights(old_weights, scores, model_used, block_height)
        timestamp = datetime.now().isoformat()
        val_node, block_hash = validate_block_por(block_height + 1, active_weights, timestamp, previous_hash, file_hash, model_used)
        cursor.execute('''INSERT INTO blockchain_por_weights (w1, w2, w3, w4, w5, w6, w7, w8, timestamp, previous_hash, validator_node, block_hash, eval_hash, model_used) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                       (*active_weights, timestamp, previous_hash, val_node, block_hash, file_hash, model_used))
    else:
        active_weights = old_weights

    title = raw_data.get("Extracted_Title", filename)
    extracted_author = raw_data.get("Extracted_Author", "").strip()
    if not extracted_author or extracted_author.lower() in ["unknown", "unknown author", "none", "n/a"] or extracted_author == os.path.splitext(filename)[0]:
        extracted_author = pdf_meta_author or "Research Scholar"

    fields, subfields = raw_data.get("fields", ["General Science"]), raw_data.get("subfields", ["General"])
    
    raw_final_score = float(np.dot(scores, active_weights)) / 8.0
    final_score = float(raw_final_score * (0.7 + (logic_integrity / 333.3)))
    drift = calculate_complex_drift(scope_alignment, scores) if scope.strip() else "N/A"
    rec = get_recommendation_spectrum(final_score, drift) if scope.strip() else "N/A"
    
    cursor.execute('''INSERT INTO papers_assessment (eval_hash, user_id, title, filename, scope, c1, c2, c3, c4, c5, c6, c7, c8, logic_score, scope_alignment, subfields, fields, author_name, final_score, timestamp) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                   (file_hash, user_id, title, filename, scope, *scores, logic_integrity, scope_alignment, json.dumps(subfields), json.dumps(fields), extracted_author, final_score, datetime.now().isoformat()))
    conn.commit()
    
    return title, extracted_author, final_score, logic_integrity, drift, rec, fields, subfields, scores_dict, file_hash

# --- PyTorch Models ---
class PiBlockchainDataset(Dataset):
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
    def __init__(self, input_size=8, hidden_layer_size=32, output_size=8):
        super(PiBrainLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(hidden_layer_size, 16), nn.ReLU(), nn.Linear(16, output_size)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out[:, -1, :])
        return torch.softmax(predictions, dim=-1) * 8.0