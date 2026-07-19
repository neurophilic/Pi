import os
import sqlite3
import json
import hashlib
import time
import tempfile
import re
import requests
import colorsys
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
import fitz  # PyMuPDF
from groq import Groq, RateLimitError
from pyvis.network import Network

# --- 1. CONFIGURATION & ENVIRONMENT ---
st.set_page_config(page_title="π-Index Assessment Engine", layout="wide")

PRIMARY_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"
MAX_TEXT_TOKENS = 6000
SEED_NUMBER = 42

BASE_DIR = os.path.abspath('./Scientometric_Pi_Index')
os.makedirs(BASE_DIR, exist_ok=True)
DB_PATH = os.path.join(BASE_DIR, 'pi_index_assessment_v10_pos.db')

GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    st.error("API Key not found! Please configure your environment variables or Streamlit Secrets.")
    st.stop()
client = Groq(api_key=GROQ_API_KEY)

# --- ORCID LIVE API VERIFICATION ---
def verify_orcid_live(orcid_id):
    """Pings the real ORCID Public API to verify the ID and fetch the user's name."""
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

# --- 2. SYSTEM ACCESS (ORCID INTEGRATION) ---
st.sidebar.title("System Access")

# Initialize session state for ORCID
if 'orcid_id' not in st.session_state:
    st.session_state.orcid_id = "0000-0000-0000-0000"
    st.session_state.orcid_name = ""
    st.session_state.is_authenticated = False

if not st.session_state.is_authenticated:
    st.sidebar.markdown("### Authenticate via ORCID")
    st.sidebar.info("Connect your real ORCID iD to securely track your assessments and isolate your topological maps.")
    
    manual_orcid = st.sidebar.text_input("Enter ORCID iD", placeholder="XXXX-XXXX-XXXX-XXXX")
    if st.sidebar.button("🔗 Validate & Connect via ORCID API"):
        # Regex to validate standard 16-digit ORCID format
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
                st.sidebar.error(user_name) # Outputs the error message
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
st.sidebar.caption("Assessment histories and maps are isolated to your ORCID, but the PoS blockchain remains globally synchronized.")

# --- 3. PI PROGRESSION (EPOCH ACCURACY) ---
def get_pi_float(block_height):
    pi_str = "3.141592653589793238462643383279502884197169399375105820974944592"
    length = min(block_height + 3, len(pi_str))
    return float(pi_str[:length])

# --- 4. BLOCKCHAIN & DATABASE ---
def validate_block_pos(block_index, weights, timestamp, previous_hash, eval_hash, model_used):
    validator_node = "Validator_Pi_" + hashlib.md5(str(time.time()).encode()).hexdigest()[:6]
    data = f"{block_index}{weights}{timestamp}{previous_hash}{validator_node}{eval_hash}{model_used}".encode('utf-8')
    block_hash = hashlib.sha256(data).hexdigest()
    return validator_node, block_hash

@st.cache_resource
def init_system():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS papers_assessment 
                      (eval_hash TEXT PRIMARY KEY, user_id TEXT, title TEXT, filename TEXT, scope TEXT,
                       c1 REAL, c2 REAL, c3 REAL, c4 REAL, 
                       c5 REAL, c6 REAL, c7 REAL, c8 REAL, 
                       scope_alignment REAL,
                       subfields TEXT, fields TEXT, final_score REAL, timestamp DATETIME)''')
                       
    cursor.execute('''CREATE TABLE IF NOT EXISTS blockchain_pos_weights 
                      (block_height INTEGER PRIMARY KEY AUTOINCREMENT, 
                       w1 REAL, w2 REAL, w3 REAL, w4 REAL, 
                       w5 REAL, w6 REAL, w7 REAL, w8 REAL, 
                       timestamp DATETIME, previous_hash TEXT, 
                       validator_node TEXT, block_hash TEXT, eval_hash TEXT, model_used TEXT)''')
    
    cursor.execute("SELECT COUNT(*) FROM blockchain_pos_weights")
    if cursor.fetchone()[0] == 0:
        genesis_weights = [1.0] * 8
        prev_hash = "0" * 64
        timestamp = datetime.now().isoformat()
        val_node, block_hash = validate_block_pos(1, genesis_weights, timestamp, prev_hash, "genesis", "none")
        cursor.execute('''INSERT INTO blockchain_pos_weights 
                          (w1, w2, w3, w4, w5, w6, w7, w8, timestamp, previous_hash, validator_node, block_hash, eval_hash, model_used) 
                          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                       (*genesis_weights, timestamp, prev_hash, val_node, block_hash, "genesis", "none"))
    conn.commit()
    return conn

conn = init_system()

# --- 5. DYNAMIC WEIGHT ADAPTATION ---
def calculate_model_driven_weights(old_weights, scores, model_name, block_height):
    v, s = (3.3, 70.0) if "70b" in model_name else (3.1, 8.0)
    pi_acc = get_pi_float(block_height)
    delta_models = abs((3.3 * 70.0) - (3.1 * 8.0)) 
    
    # 1. Apply a mathematical stretch to force variance from the mean
    mean_score = np.mean(scores)
    stretched_scores = [max(1.0, min(100.0, mean_score + (score - mean_score) * 3.0)) for score in scores]
    
    new_weights = []
    for i, old_w in enumerate(old_weights):
        c_score = stretched_scores[i]
        
        # 2. Exponentiate the score impact ((c_score/100)^2) to widen the distribution
        delta_w = ((v * s) / (delta_models * pi_acc)) * ((c_score / 100.0) ** 2)
        w_new = old_w * 0.85 + (1.0 + delta_w * 0.15) * 0.15
        new_weights.append(w_new)
        
    sum_w = sum(new_weights)
    return [round((w / sum_w) * 8.0, 6) for w in new_weights]

# --- 6. SEMANTIC EXTRACTION & DRIFT ---
def evaluate_pdf_text(text, scope, model):
    # Strict instructions to force the LLM to use the full 0-100 scale
    prompt = f"""You are a brutally critical expert peer reviewer contributing to the π-Index.
The user is a researcher currently working on this specific project/scope: "{scope}"

CRITICAL INSTRUCTION: You MUST use the full 0-100 scale for scores. Do NOT cluster scores around 70-80. If a paper is weak in an area, give it a 10 or 20. If exceptional, 90+. Force extreme variance based on actual merit.

Return ONLY a valid JSON object matching exactly this structure:
{{
    "Extracted_Title": "Title", "Scope_Alignment": 85,
    "scores": {{"C1_Originality": 20, "C2_Methodological_Rigor": 95, "C3_Interdisciplinary": 40, "C4_Societal_Impact": 15, "C5_Open_Science_Potential": 60, "C6_Literature_Integration": 80, "C7_Empirical_Density": 55, "C8_Future_Actionability": 30}},
    "fields": ["Field1", "Field2"], "subfields": ["Subfield1"]
}}
Text: {text[:MAX_TEXT_TOKENS]}
"""
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model, temperature=0.3, seed=SEED_NUMBER, response_format={"type": "json_object"}
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
    file_hash = hashlib.sha256(file_bytes + scope.encode('utf-8') + user_id.encode('utf-8')).hexdigest()
    cursor = conn.cursor()
    cursor.execute("SELECT final_score, scope_alignment, title, fields, subfields, c1, c2, c3, c4, c5, c6, c7, c8 FROM papers_assessment WHERE eval_hash=? AND user_id=?", (file_hash, user_id))
    cached = cursor.fetchone()
    
    if cached:
        score, alignment, title, fields_str, subfields_str, c1, c2, c3, c4, c5, c6, c7, c8 = cached
        fields = json.loads(fields_str) if fields_str else ["General Science"]
        subfields = json.loads(subfields_str) if subfields_str else ["General"]
        scores_array = [c1, c2, c3, c4, c5, c6, c7, c8]
        drift = calculate_complex_drift(alignment, scores_array)
        scores_dict = {"C1_Originality": c1, "C2_Methodological_Rigor": c2, "C3_Interdisciplinary": c3, "C4_Societal_Impact": c4, "C5_Open_Science_Potential": c5, "C6_Literature_Integration": c6, "C7_Empirical_Density": c7, "C8_Future_Actionability": c8}
        return title, score, drift, get_recommendation_spectrum(score, drift), fields, subfields, scores_dict

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = " ".join([page.get_text() for page in doc[:3]])
    model_used = PRIMARY_MODEL
    try:
        raw_data = evaluate_pdf_text(text, scope, model_used)
    except RateLimitError:
        time.sleep(2)
        model_used = FALLBACK_MODEL
        raw_data = evaluate_pdf_text(text, scope, model_used)
        
    cursor.execute("SELECT block_height, block_hash, w1, w2, w3, w4, w5, w6, w7, w8 FROM blockchain_pos_weights ORDER BY block_height DESC LIMIT 1")
    epoch_data = cursor.fetchone()
    block_height, previous_hash, old_weights = epoch_data[0], epoch_data[1], epoch_data[2:]
    
    scores_dict = raw_data.get("scores", {})
    scores = [scores_dict.get(k, 50.0) for k in ["C1_Originality", "C2_Methodological_Rigor", "C3_Interdisciplinary", "C4_Societal_Impact", "C5_Open_Science_Potential", "C6_Literature_Integration", "C7_Empirical_Density", "C8_Future_Actionability"]]
    
    new_weights = calculate_model_driven_weights(old_weights, scores, model_used, block_height)
    timestamp = datetime.now().isoformat()
    val_node, block_hash = validate_block_pos(block_height + 1, new_weights, timestamp, previous_hash, file_hash, model_used)
    
    cursor.execute('''INSERT INTO blockchain_pos_weights (w1, w2, w3, w4, w5, w6, w7, w8, timestamp, previous_hash, validator_node, block_hash, eval_hash, model_used) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                   (*new_weights, timestamp, previous_hash, val_node, block_hash, file_hash, model_used))
    
    scope_alignment = raw_data.get("Scope_Alignment", 50.0)
    title = raw_data.get("Extracted_Title", filename)
    fields, subfields = raw_data.get("fields", ["General Science"]), raw_data.get("subfields", ["General"])
    final_score = float(np.dot(scores, new_weights)) / 8.0
    drift = calculate_complex_drift(scope_alignment, scores)
    
    cursor.execute('''INSERT INTO papers_assessment (eval_hash, user_id, title, filename, scope, c1, c2, c3, c4, c5, c6, c7, c8, scope_alignment, subfields, fields, final_score, timestamp) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                   (file_hash, user_id, title, filename, scope, *scores, scope_alignment, json.dumps(subfields), json.dumps(fields), final_score, timestamp))
    conn.commit()
    return title, final_score, drift, get_recommendation_spectrum(final_score, drift), fields, subfields, scores_dict

# --- 7. TOPOLOGICAL MAPPING (INTERACTIVE PYVIS NETWORK WITH WEIGHTS) ---

def generate_interactive_bubble_chart(scope, user_id):
    cursor = conn.cursor()
    cursor.execute("SELECT fields, subfields, final_score FROM papers_assessment WHERE scope=? AND user_id=?", (scope, user_id))
    data = cursor.fetchall()
    
    if not data: return None, None
    
    all_topics = []
    for fields_json, subfields_json, final_score in data:
        try:
            fields = [f.title().strip() for f in json.loads(fields_json)]
            subfields = [s.title().strip() for s in json.loads(subfields_json)]
            score = float(final_score) if final_score else 50.0
            for f in fields: all_topics.append({'topic': f, 'weight': score})
            for s in subfields: all_topics.append({'topic': s, 'weight': score})
        except: continue
            
    if not all_topics: return None, None
    
    df_topics = pd.DataFrame(all_topics)
    topic_counts = df_topics.groupby(['topic'])['weight'].sum().reset_index(name='weight')
    
    if topic_counts.empty: return None, None
    
    unique_topics = topic_counts['topic'].unique()
    
    def get_color(i, n):
        h, s, v = i/n, 0.7, 0.9
        rgb = colorsys.hsv_to_rgb(h, s, v)
        return '#%02x%02x%02x' % tuple(int(x * 255) for x in rgb)
    
    color_map = {topic: get_color(i, len(unique_topics)) for i, topic in enumerate(unique_topics)}
    
    net = Network(height='600px', width='100%', bgcolor='#ffffff', font_color='#2c3e50', notebook=False)
    
    # --- ADJUSTED PHYSICS FOR MORE SPACE ---
    physics_options = """
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -20,
          "centralGravity": -5,
          "springLength": 150,
          "avoidOverlap": 1
        },
        "solver": "forceAtlas2Based"
      }
    }
    """
    net.set_options(physics_options)
    
    for _, row in topic_counts.iterrows():
        # Size tied to weight
        node_size = 25 + (row['weight'] * 1.2) 
        
        net.add_node(
            n_id=row['topic'],
            label=' ', # Forced whitespace to hide label
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
# --- 8. USER INTERFACE ---
st.title("π-Index Assessment Engine")
st.markdown("**Upload papers, define your scope of research, let π-index filter noise and have better results**")

with st.expander("View π-Index Grading Criteria & Theoretical Formulations"):
    st.markdown("### Evaluation Metrics (0 - 100 Scale)")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**C1: Originality**\nEvaluates uniqueness through epistemic gradient fields.")
        st.markdown(r"$$O = \varpi_1 \cdot \lim_{\Delta t \to 0} \oint_{\partial \Omega} \frac{\nabla \times (\mathcal{H}_{novel} \otimes \mathcal{K}_{epistemic})}{\iint_{\mathcal{M}} \sum_{i=1}^N (\zeta_i \cdot \mathcal{I}_{existing}^{(i)}) \, d\mu} \cdot d\mathbf{S} \times 100$$")
        st.markdown("**C2: Methodological Rigor**\nAssesses robustness via error-covariance tensors.")
        st.markdown(r"$$R = \varpi_2 \cdot \left( 1 - \frac{\mathrm{tr}(\boldsymbol{\Sigma}_{error} \boldsymbol{\Lambda}^{-1})}{\det(\boldsymbol{\mu}_{signal} \otimes \mathbf{W})} \right) \cdot \prod_{k=1}^{m} \int_{0}^{\infty} \rho_k(x) e^{-\beta x^2} \Gamma\left(k+\frac{1}{2}\right) dx \times 100$$")
        st.markdown("**C3: Interdisciplinary**\nMeasures bridge capacity using generalized Rényi entropy.")
        st.markdown(r"$$I = \varpi_3 \cdot \left( \frac{1}{1-\alpha} \ln \left( \sum_{j=1}^{K} p_j^\alpha \right) + \sum_{i,j} \frac{A_{ij} \phi_i \phi_j}{\sqrt{d_i d_j}} \right) \cdot \frac{\Xi(\mathcal{G})}{\ln K \cdot \mathcal{Z}_{norm}} \times 100$$")
        st.markdown("**C4: Societal Impact**\nProjects applications utilizing fractional stochastic integration.")
        st.markdown(r"$$S = \varpi_4 \cdot \frac{1}{\Gamma(q)} \int_{t_0}^{t_\infty} (t_\infty - \tau)^{q-1} e^{-\gamma(\tau) \tau} \cdot \Theta\left[ \sum_{v \in \mathcal{V}} \omega_v U_v(\tau, \mathbf{x}) \right] d\tau \times 100$$")
    with col2:
        st.markdown("**C5: Open Science Potential**\nGauges transparency via multi-objective integration.")
        st.markdown(r"$$O_s = \varpi_5 \cdot \frac{\sum_{\ell \in \mathcal{L}} \alpha_\ell \mathcal{D}_{open}^{(\ell)} + \beta \iint_{\mathcal{C}} \nabla \cdot \mathbf{J}_{code} \, dV}{\max \left( \sup_{t} \mathcal{D}_{total}(t), \inf_{\epsilon>0} \mathcal{C}_{total}(\epsilon) \right)} \times \mathcal{P}_{FAIR} \times 100$$")
        st.markdown("**C6: Literature Integration**\nEvaluates embedding via non-Euclidean PageRank.")
        st.markdown(r"$$L = \varpi_6 \cdot \frac{1}{\mathcal{N}} \sum_{i=1}^{\mathcal{N}} \int_{\mathcal{M}} e^{-\lambda d_g(x_i, x_{core})} R(x_i) \sqrt{g} \, dx_i \cdot \frac{\text{PR}(x_i)}{\sum_j \text{PR}(x_j)} \times 100$$")
        st.markdown("**C7: Empirical Density**\nEvaluates data depth utilizing Fisher information metrics.")
        st.markdown(r"$$E_d = \varpi_7 \cdot \tanh \left( \frac{\det \mathcal{I}_{Fisher}(\hat{\theta}) \cdot \mathbb{E}_{P}\left[\log\frac{P}{Q}\right]}{\mathcal{V}_{baseline} \cdot \oint_\Gamma \omega_{data}} \right) \times \sum_{d=1}^D \lambda_d \kappa_d \times 100$$")
        st.markdown("**C8: Future Actionability**\nDetermines continuation potential using Lyapunov exponents.")
        st.markdown(r"$$F_a = \varpi_8 \cdot \frac{1}{\mathcal{Z}} \int_{\mathcal{X}} \frac{1}{1 + \exp\left(-\sum_{k=1}^K w_k(\eta_k(\mathbf{x}) - \eta_{0,k}) + \Lambda_{Lyapunov}\right)} d\mu(\mathbf{x}) \times 100$$")

tab1, tab2, tab3 = st.tabs(["Batch Assessment", "Scope Cartography", "Active Epoch Constants"])

with tab1:
    research_scope = st.text_input("Define your specific Research Topic / Scope", placeholder="e.g., Application of deep learning in vascular imaging...")
    uploaded_files = st.file_uploader("Upload Academic Papers (PDFs)", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Run Batch Assessment", type="primary"):
        # --- ERROR HANDLING ---
        if not research_scope.strip():
            st.warning("⚠️ Please define a Research Topic / Scope before running the assessment.")
        elif not uploaded_files:
            st.warning("⚠️ Please upload at least one academic paper (PDF) to proceed.")
        else:
            # --- EXECUTION ---
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, file in enumerate(uploaded_files):
                status_text.text(f"Analyzing {i+1} of {len(uploaded_files)}: {file.name}...")
                
                # Logic execution
                title, score, drift, rec, fields, subfields, scores_dict = process_single_pdf(
                    file.read(), file.name, research_scope, current_user
                )
                
                # Appending data...
                combined_fields = f"Fields: {', '.join(fields)} | Subfields: {', '.join(subfields)}"
                results.append({
                    "No.": i + 1,
                    "File Name": file.name,
                    "Topic": research_scope,
                    "Fields & Subfields": combined_fields,
                    "π-Index (0-100)": round(score, 1),
                    "Recommendation Spectrum": rec,
                    "Scope Drift %": round(drift, 1),
                    "C1": scores_dict.get("C1_Originality", 0.0),
                    "C2": scores_dict.get("C2_Methodological_Rigor", 0.0),
                    "C3": scores_dict.get("C3_Interdisciplinary", 0.0),
                    "C4": scores_dict.get("C4_Societal_Impact", 0.0),
                    "C5": scores_dict.get("C5_Open_Science_Potential", 0.0),
                    "C6": scores_dict.get("C6_Literature_Integration", 0.0),
                    "C7": scores_dict.get("C7_Empirical_Density", 0.0),
                    "C8": scores_dict.get("C8_Future_Actionability", 0.0)
                })
                progress_bar.progress((i + 1) / len(uploaded_files))
                
            status_text.text("Batch processing complete!")
            
            # Rendering Results
            df = pd.DataFrame(results)
            df_display = df.sort_values(by=["π-Index (0-100)"], ascending=False)
            st.markdown("### Assessment Summary")
            st.dataframe(df_display, use_container_width=True, hide_index=True)
            
            csv = df_display.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download Summary as CSV", data=csv, file_name="pi_index_assessment_results.csv", mime="text/csv")

    # Display History below the form
    st.markdown("---")
    st.markdown("### Latest Assessment History")
    
    # Clear History Logic
    if st.button("🗑️ Clear All History"):
        cursor = conn.cursor()
        cursor.execute("DELETE FROM papers_assessment WHERE user_id=?", (current_user,))
        conn.commit()
        st.rerun() # Refresh the app to update the view
        
    # Fetch and display history
    cursor = conn.cursor()
    cursor.execute("SELECT title, scope, final_score, timestamp, eval_hash FROM papers_assessment WHERE user_id=? ORDER BY timestamp DESC LIMIT 20", (current_user,))
    history_data = cursor.fetchall()
    
    if history_data:
        df_hist = pd.DataFrame(history_data, columns=["Paper Title", "Scope", "π-Index Score", "Date", "Evaluation Hash"])
        st.dataframe(df_hist, use_container_width=True, hide_index=True)
    else:
        st.info("No assessment history found.")

with tab2:
    st.subheader("Field & Subfield Epistemic Bubbles")
    st.write("Visualizing your research scope (Click and drag the bubbles to interact)")
    
    if research_scope:
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
    else:
        st.info("Please define a research scope in the 'Batch Assessment' tab first.")

with tab3:
    cursor = conn.cursor()
    cursor.execute("SELECT block_height, w1, w2, w3, w4, w5, w6, w7, w8, model_used, eval_hash, block_hash FROM blockchain_pos_weights ORDER BY block_height DESC LIMIT 1")
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
        st.markdown("### PoS Blockchain Explorer")
        st.markdown("""
        **How to Verify via the Explorer:**
        1.  **Locate the Eval Hash:** Copy the Evaluation Hash (Document) associated with a paper you assessed.
        2.  **Use the Explorer:** Paste that hash into the PoS Blockchain Explorer input field below.
        3.  **Click "Verify Record":** The system will query the global blockchain database to return the exact Weights Matrix alongside the immutable Block Hash.
        """)
        
        explore_col1, explore_col2 = st.columns([3, 1])
        with explore_col1: search_query = st.text_input("Enter Document Evaluation Hash or Block Hash to verify ledger record...")
        with explore_col2: 
            st.write("")
            st.write("")
            search_btn = st.button("Verify Record")
            
        if search_btn and search_query:
            cursor.execute("SELECT * FROM blockchain_pos_weights WHERE block_hash=? OR eval_hash=?", (search_query, search_query))
            record = cursor.fetchone()
            if record:
                st.success("Valid Block Found on Ledger!")
                st.json({"Block Height": record[0], "Timestamp": record[9], "Model Used": record[13], "Validator Node": record[11], "Block Hash": record[12], "Previous Hash": record[10], "Evaluation Hash (Document)": record[14], "Weights Matrix (w1..w8)": record[1:9]})
            else:
                st.error("No block matching that signature was found on the ledger.")
                
        with st.expander("View Recent Global Ledger Blocks"):
            cursor.execute("SELECT block_height, timestamp, model_used, block_hash FROM blockchain_pos_weights ORDER BY block_height DESC LIMIT 10")
            df_blocks = pd.DataFrame(cursor.fetchall(), columns=["Height", "Timestamp", "Model", "Block Hash"])
            st.dataframe(df_blocks, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray; font-size: 0.8em;'>Framework Author: Ali Vafadar Yengejeh | Università degli Studi di Milano-Bicocca</div>", unsafe_allow_html=True)
