import os
import sqlite3
import json
import hashlib
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import fitz  # PyMuPDF
from groq import Groq, RateLimitError

# --- 1. CONFIGURATION & ENVIRONMENT ---
st.set_page_config(page_title="π-Index Assessment Engine", layout="wide")

PRIMARY_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"
MAX_TEXT_TOKENS = 6000
SEED_NUMBER = 42

BASE_DIR = os.path.abspath('./Scientometric_Pi_Index')
os.makedirs(BASE_DIR, exist_ok=True)
# Updated DB name for floating point constant migration
DB_PATH = os.path.join(BASE_DIR, 'pi_index_assessment_v7_pos.db')

GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    st.error("API Key not found! Please configure your environment variables or Streamlit Secrets.")
    st.stop()
client = Groq(api_key=GROQ_API_KEY)

# --- 2. PI PROGRESSION (EPOCH ACCURACY) ---
def get_pi_float(block_height):
    """Increases Pi accuracy by revealing more decimals as epochs (blocks) progress."""
    pi_str = "3.141592653589793238462643383279502884197169399375105820974944592"
    # Block 1 starts at 3.14 (4 chars), Block 2 -> 3.141, etc.
    length = min(block_height + 3, len(pi_str))
    return float(pi_str[:length])

# --- 3. BLOCKCHAIN (PROOF OF STAKE) & DATABASE INITIALIZATION ---
def validate_block_pos(block_index, weights, timestamp, previous_hash, eval_hash, model_used):
    """Simulates Proof-of-Stake validation to generate a unique block hash per paper."""
    validator_node = "Validator_Pi_" + hashlib.md5(str(time.time()).encode()).hexdigest()[:6]
    data = f"{block_index}{weights}{timestamp}{previous_hash}{validator_node}{eval_hash}{model_used}".encode('utf-8')
    block_hash = hashlib.sha256(data).hexdigest()
    return validator_node, block_hash

@st.cache_resource
def init_system():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS papers_assessment 
                      (eval_hash TEXT PRIMARY KEY, title TEXT, filename TEXT, scope TEXT,
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
        # Genesis Block: Constants initialized to exactly 1.0
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

# --- 4. DYNAMIC WEIGHT ADAPTATION (LLM DEPENDENT) ---
def calculate_model_driven_weights(old_weights, scores, model_name, block_height):
    """
    Formulates the new constant based on the model's decision, LLM version, size, Pi, and delta.
    """
    if "70b" in model_name:
        v, s = 3.3, 70.0
    else:
        v, s = 3.1, 8.0
        
    pi_acc = get_pi_float(block_height)
    
    # Delta of different models: |(V1 * S1) - (V2 * S2)| = |(3.3 * 70) - (3.1 * 8)|
    delta_models = abs((3.3 * 70.0) - (3.1 * 8.0)) 
    
    new_weights = []
    for i, old_w in enumerate(old_weights):
        c_score = scores[i]
        
        # Change is formulated by LLM version, size, pi, delta of models, and the criteria value decided
        delta_w = ((v * s) / (delta_models * pi_acc)) * (c_score / 100.0)
        
        # We blend the dynamic delta into the old weight. 
        # Using a dampening factor so it oscillates structurally around 1.0 over time.
        w_new = old_w * 0.85 + (1.0 + delta_w * 0.15) * 0.15
        new_weights.append(w_new)
        
    # Normalize so the sum remains equal to the original baseline (8.0)
    sum_w = sum(new_weights)
    normalized_weights = [(w / sum_w) * 8.0 for w in new_weights]
    
    return [round(w, 6) for w in normalized_weights]

# --- 5. SEMANTIC LLM EXTRACTION & MATHEMATICAL DRIFT ---
def evaluate_pdf_text(text, scope, model):
    prompt = f"""You are an expert peer reviewer contributing to the π-Index.
The user is a researcher currently working on this specific project/scope: "{scope}"

Analyze the following excerpt from an academic paper.
1. Extract the Title.
2. Evaluate 'Scope_Alignment' on a scale of 0 to 100 (100 = highly relevant to scope, 0 = completely unrelated).
3. Evaluate the 8 π-Index criteria on a scale of 0 to 100.
4. Identify 3 to 5 overarching scientific "fields".
5. Identify 3 to 5 highly specific "subfields".

Return ONLY a valid JSON object matching exactly this structure:
{{
    "Extracted_Title": "Full title of the paper",
    "Scope_Alignment": 85,
    "scores": {{
        "C1_Originality": 80, "C2_Methodological_Rigor": 70, 
        "C3_Interdisciplinary": 60, "C4_Societal_Impact": 50, 
        "C5_Open_Science_Potential": 60, "C6_Literature_Integration": 70, 
        "C7_Empirical_Density": 80, "C8_Future_Actionability": 70
    }},
    "fields": ["Biomedical Engineering", "Computer Science"],
    "subfields": ["Deep Learning", "Vascular Imaging"]
}}

Text: {text[:MAX_TEXT_TOKENS]}
"""
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model, temperature=0.1, seed=SEED_NUMBER, response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

def calculate_complex_drift(alignment, scores):
    mu = np.mean(scores)
    sigma = np.std(scores)
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
    else: return "Tier VI: Orthogonal / Unrelated Noise"

def process_single_pdf(file_bytes, filename, scope):
    file_hash = hashlib.sha256(file_bytes + scope.encode('utf-8')).hexdigest()
    
    cursor = conn.cursor()
    cursor.execute("SELECT final_score, scope_alignment, title, fields, subfields, c1, c2, c3, c4, c5, c6, c7, c8 FROM papers_assessment WHERE eval_hash=?", (file_hash,))
    cached = cursor.fetchone()
    
    if cached:
        score, alignment, title, fields_str, subfields_str, c1, c2, c3, c4, c5, c6, c7, c8 = cached
        fields = json.loads(fields_str) if fields_str else ["General Science"]
        subfields = json.loads(subfields_str) if subfields_str else ["General"]
        
        scores_array = [c1, c2, c3, c4, c5, c6, c7, c8]
        drift = calculate_complex_drift(alignment, scores_array)
        
        scores_dict = {
            "C1_Originality": c1, "C2_Methodological_Rigor": c2,
            "C3_Interdisciplinary": c3, "C4_Societal_Impact": c4,
            "C5_Open_Science_Potential": c5, "C6_Literature_Integration": c6,
            "C7_Empirical_Density": c7, "C8_Future_Actionability": c8
        }
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
    block_height = epoch_data[0]
    previous_hash = epoch_data[1]
    old_weights = epoch_data[2:]
    
    scores_dict = raw_data.get("scores", {})
    scores = [scores_dict.get(k, 50.0) for k in ["C1_Originality", "C2_Methodological_Rigor", "C3_Interdisciplinary", "C4_Societal_Impact", "C5_Open_Science_Potential", "C6_Literature_Integration", "C7_Empirical_Density", "C8_Future_Actionability"]]
    
    # Calculate new dynamic weights based on this specific paper and model
    new_weights = calculate_model_driven_weights(old_weights, scores, model_used, block_height)
    
    # Mine new block for this specific evaluation
    timestamp = datetime.now().isoformat()
    new_height = block_height + 1
    val_node, block_hash = validate_block_pos(new_height, new_weights, timestamp, previous_hash, file_hash, model_used)
    
    cursor.execute('''INSERT INTO blockchain_pos_weights 
                      (w1, w2, w3, w4, w5, w6, w7, w8, timestamp, previous_hash, validator_node, block_hash, eval_hash, model_used) 
                      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                   (*new_weights, timestamp, previous_hash, val_node, block_hash, file_hash, model_used))
    
    scope_alignment = raw_data.get("Scope_Alignment", 50.0)
    title = raw_data.get("Extracted_Title", filename)
    fields = raw_data.get("fields", ["General Science"])
    subfields = raw_data.get("subfields", ["General"])
    
    # Calculate final score utilizing the newly minted constant weights
    final_score = float(np.dot(scores, new_weights)) / 8.0
    drift = calculate_complex_drift(scope_alignment, scores)
    
    cursor.execute('''INSERT INTO papers_assessment 
                      (eval_hash, title, filename, scope, c1, c2, c3, c4, c5, c6, c7, c8, scope_alignment, subfields, fields, final_score, timestamp) 
                      VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                   (file_hash, title, filename, scope, *scores,
                    scope_alignment,
                    json.dumps(subfields), json.dumps(fields), final_score, timestamp))
    conn.commit()
    
    return title, final_score, drift, get_recommendation_spectrum(final_score, drift), fields, subfields, scores_dict

# --- 6. TOPOLOGICAL MAPPING (3D REALISTIC BUBBLE CHART) ---
def generate_bubble_chart(scope):
    cursor = conn.cursor()
    cursor.execute("SELECT fields, subfields FROM papers_assessment WHERE scope=?", (scope,))
    data = cursor.fetchall()
    
    if not data: return None
    
    all_topics = []
    
    for fields_json, subfields_json in data:
        try:
            fields = [f.title().strip() for f in json.loads(fields_json)]
            subfields = [s.title().strip() for s in json.loads(subfields_json)]
            
            for f in fields:
                all_topics.append({'topic': f, 'category': 'Field'})
            for s in subfields:
                all_topics.append({'topic': s, 'category': 'Subfield'})
        except: continue
            
    if not all_topics: return None
    
    df_topics = pd.DataFrame(all_topics)
    topic_counts = df_topics.groupby(['topic', 'category']).size().reset_index(name='count')
    topic_counts = topic_counts.sort_values(by='count', ascending=False).reset_index(drop=True)
    
    max_count = topic_counts['count'].max()
    min_size = 35
    max_size = 110
    topic_counts['bubble_size'] = min_size + (topic_counts['count'] / max_count) * (max_size - min_size)
    
    np.random.seed(SEED_NUMBER)
    topic_counts['x'] = np.random.normal(0, 1.5, len(topic_counts))
    topic_counts['y'] = np.random.normal(0, 1.5, len(topic_counts))
    
    for _ in range(60):
        for i in range(len(topic_counts)):
            for j in range(len(topic_counts)):
                if i != j:
                    dx = topic_counts.loc[i, 'x'] - topic_counts.loc[j, 'x']
                    dy = topic_counts.loc[i, 'y'] - topic_counts.loc[j, 'y']
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < 0.6:
                        topic_counts.loc[i, 'x'] += dx * 0.15
                        topic_counts.loc[i, 'y'] += dy * 0.15
    
    fig = go.Figure()
    color_palette = px.colors.qualitative.Bold + px.colors.qualitative.Pastel + px.colors.qualitative.Vivid
    
    for i, row in topic_counts.iterrows():
        topic = row['topic']
        size = row['bubble_size']
        count = row['count']
        
        # Labels removed, completely reliant on color palette and hover
        fig.add_trace(go.Scatter(
            x=[row['x']], y=[row['y']],
            mode='markers',
            marker=dict(
                size=size,
                color=color_palette[i % len(color_palette)],
                line=dict(width=1, color='rgba(255, 255, 255, 0.4)'),
                sizemode='diameter',
                gradient=dict(type='radial', color='rgba(255, 255, 255, 0.85)'),
                opacity=0.95
            ),
            name=topic, 
            hovertext=f"<b>{topic}</b><br>Category: {row['category']}<br>Focus Frequency: {count}",
            hoverinfo="text"
        ))
        
    fig.update_layout(
        showlegend=True,
        legend_title_text='Fields & Subfields',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=10, r=10, b=10, t=10),
        hovermode='closest',
        plot_bgcolor='rgba(0,0,0,0)'
    )
                                        
    return fig

# --- 7. USER INTERFACE ---
st.title("π-Index Assessment Engine")
st.markdown("**Upload papers, define your scope of research, let π-index filter noise and have better results**")

with st.expander("View π-Index Grading Criteria & Theoretical Formulations"):
    st.markdown("### Evaluation Metrics (0 - 100 Scale)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**C1: Originality**  \nEvaluates the uniqueness of the hypothesis, approach, or findings through epistemic gradient fields.")
        st.markdown(r"$$O = \varpi_1 \cdot \lim_{\Delta t \to 0} \oint_{\partial \Omega} \frac{\nabla \times (\mathcal{H}_{novel} \otimes \mathcal{K}_{epistemic})}{\iint_{\mathcal{M}} \sum_{i=1}^N (\zeta_i \cdot \mathcal{I}_{existing}^{(i)}) \, d\mu} \cdot d\mathbf{S} \times 100$$")
        
        st.markdown("**C2: Methodological Rigor**  \nAssesses robustness and reproducibility via error-covariance tensors and persistent homology.")
        st.markdown(r"$$R = \varpi_2 \cdot \left( 1 - \frac{\mathrm{tr}(\boldsymbol{\Sigma}_{error} \boldsymbol{\Lambda}^{-1})}{\det(\boldsymbol{\mu}_{signal} \otimes \mathbf{W})} \right) \cdot \prod_{k=1}^{m} \int_{0}^{\infty} \rho_k(x) e^{-\beta x^2} \Gamma\left(k+\frac{1}{2}\right) dx \times 100$$")
        
        st.markdown("**C3: Interdisciplinary**  \nMeasures network bridge capacity using generalized Rényi entropy over disciplinary graphs.")
        st.markdown(r"$$I = \varpi_3 \cdot \left( \frac{1}{1-\alpha} \ln \left( \sum_{j=1}^{K} p_j^\alpha \right) + \sum_{i,j} \frac{A_{ij} \phi_i \phi_j}{\sqrt{d_i d_j}} \right) \cdot \frac{\Xi(\mathcal{G})}{\ln K \cdot \mathcal{Z}_{norm}} \times 100$$")
        
        st.markdown("**C4: Societal Impact**  \nProjects real-world macro applications utilizing fractional stochastic integration.")
        st.markdown(r"$$S = \varpi_4 \cdot \frac{1}{\Gamma(q)} \int_{t_0}^{t_\infty} (t_\infty - \tau)^{q-1} e^{-\gamma(\tau) \tau} \cdot \Theta\left[ \sum_{v \in \mathcal{V}} \omega_v U_v(\tau, \mathbf{x}) \right] d\tau \times 100$$")

    with col2:
        st.markdown("**C5: Open Science Potential**  \nGauges transparent reporting optimization via multi-objective integration over FAIR limits.")
        st.markdown(r"$$O_s = \varpi_5 \cdot \frac{\sum_{\ell \in \mathcal{L}} \alpha_\ell \mathcal{D}_{open}^{(\ell)} + \beta \iint_{\mathcal{C}} \nabla \cdot \mathbf{J}_{code} \, dV}{\max \left( \sup_{t} \mathcal{D}_{total}(t), \inf_{\epsilon>0} \mathcal{C}_{total}(\epsilon) \right)} \times \mathcal{P}_{FAIR} \times 100$$")
        
        st.markdown("**C6: Literature Integration**  \nEvaluates topological foundational embedding via non-Euclidean manifold PageRank distances.")
        st.markdown(r"$$L = \varpi_6 \cdot \frac{1}{\mathcal{N}} \sum_{i=1}^{\mathcal{N}} \int_{\mathcal{M}} e^{-\lambda d_g(x_i, x_{core})} R(x_i) \sqrt{g} \, dx_i \cdot \frac{\text{PR}(x_i)}{\sum_j \text{PR}(x_j)} \times 100$$")
        
        st.markdown("**C7: Empirical Density**  \nEvaluates data depth utilizing Fisher information metrics and Kullback-Leibler divergences.")
        st.markdown(r"$$E_d = \varpi_7 \cdot \tanh \left( \frac{\det \mathcal{I}_{Fisher}(\hat{\theta}) \cdot \mathbb{E}_{P}\left[\log\frac{P}{Q}\right]}{\mathcal{V}_{baseline} \cdot \oint_\Gamma \omega_{data}} \right) \times \sum_{d=1}^D \lambda_d \kappa_d \times 100$$")
        
        st.markdown("**C8: Future Actionability**  \nDetermines theoretical continuation potential using Lyapunov exponents on phase space logistics.")
        st.markdown(r"$$F_a = \varpi_8 \cdot \frac{1}{\mathcal{Z}} \int_{\mathcal{X}} \frac{1}{1 + \exp\left(-\sum_{k=1}^K w_k(\eta_k(\mathbf{x}) - \eta_{0,k}) + \Lambda_{Lyapunov}\right)} d\mu(\mathbf{x}) \times 100$$")

tab1, tab2, tab3 = st.tabs(["Batch Assessment", "Scope Cartography", "Active Epoch Constants"])

with tab1:
    research_scope = st.text_input("Define your specific Research Topic / Scope", placeholder="e.g., Application of deep learning in vascular imaging...")
    
    uploaded_files = st.file_uploader("Upload Academic Papers (PDFs)", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Run Batch Assessment", type="primary") and uploaded_files and research_scope:
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, file in enumerate(uploaded_files):
            status_text.text(f"Analyzing {i+1} of {len(uploaded_files)}: {file.name}...")
            if i > 0: time.sleep(1.5) 
            
            title, score, drift, rec, fields, subfields, scores_dict = process_single_pdf(file.read(), file.name, research_scope)
            
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
        
        df = pd.DataFrame(results)
        df_display = df.sort_values(by=["π-Index (0-100)"], ascending=False)
        
        st.markdown("### Assessment Summary")
        st.dataframe(df_display, use_container_width=True, hide_index=True)
            
        csv = df_display.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Summary as CSV", data=csv, file_name="pi_index_assessment_results.csv", mime="text/csv")

with tab2:
    st.subheader("Field & Subfield Epistemic Bubbles")
    st.write("Visualizing your research scope")
    
    if research_scope:
        fig = generate_bubble_chart(research_scope)
        if fig: 
            st.plotly_chart(fig, use_container_width=True)
        else: 
            st.info("Awaiting sufficient data for this scope.")
    else:
        st.info("Please define a research scope in the 'Batch Assessment' tab first.")

with tab3:
    cursor = conn.cursor()
    cursor.execute("SELECT block_height, w1, w2, w3, w4, w5, w6, w7, w8, model_used, eval_hash FROM blockchain_pos_weights ORDER BY block_height DESC LIMIT 1")
    epoch_data = cursor.fetchone()
    
    if epoch_data:
        block_height = epoch_data[0]
        weights = epoch_data[1:9]
        model_used = epoch_data[9]
        eval_hash = epoch_data[10]
        
        current_pi_base = get_pi_float(block_height)
        
        st.markdown(f"**Last Model Orchestration:** `{model_used}` | **Epoch Block:** `{block_height}`")
        st.markdown(r"*Weights $(\varpi)$ are initialized at $1.0$ and evolve iteratively via LLM version, parameter size, $\pi$ accuracy progression, and the criteria evaluation delta of the selected model.*")
        st.markdown("---")
        
        cols = st.columns(4)
        labels = [
            ("C1 Originality", r"$\varpi_1$"), 
            ("C2 Method Rigor", r"$\varpi_2$"), 
            ("C3 Interdisciplinary", r"$\varpi_3$"), 
            ("C4 Societal Impact", r"$\varpi_4$"), 
            ("C5 Open Science", r"$\varpi_5$"), 
            ("C6 Lit Integration", r"$\varpi_6$"), 
            ("C7 Empirical Density", r"$\varpi_7$"), 
            ("C8 Actionability", r"$\varpi_8$")
        ]
        
        for i, col in enumerate(cols * 2):
            if i < 8: 
                name, symbol = labels[i]
                col.markdown(f"**{name} ({symbol})**")
                col.markdown(f"<h3 style='margin-top:0px; margin-bottom:5px;'>{weights[i]:.6f}</h3>", unsafe_allow_html=True)
                
                with col.expander("Proof of Stake Seed"):
                    seed_hash = hashlib.sha256(f"{weights[i]}_pos_{block_height}_{current_pi_base}_{eval_hash}".encode()).hexdigest()
                    st.code(f"Seed:\n{seed_hash[:24]}...\nPi Acc: {current_pi_base}", language="text")

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray; font-size: 0.8em;'>Framework Author: Ali Vafadar Yengejeh | Università degli Studi di Milano-Bicocca</div>", unsafe_allow_html=True)
