```python
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
from collections import Counter

# --- 1. CONFIGURATION & ENVIRONMENT ---
st.set_page_config(page_title="π-Index Assessment Engine", layout="wide")

PRIMARY_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"
MAX_TEXT_TOKENS = 6000
EPOCH_DAYS = 30
SEED_NUMBER = 42

BASE_DIR = os.path.abspath('./Scientometric_Pi_Index')
os.makedirs(BASE_DIR, exist_ok=True)
DB_PATH = os.path.join(BASE_DIR, 'pi_index_assessment_v3.db')

GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    st.error("API Key not found! Please configure your environment variables or Streamlit Secrets.")
    st.stop()
client = Groq(api_key=GROQ_API_KEY)

# --- 2. DATABASE INITIALIZATION ---
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
                       
    cursor.execute('''CREATE TABLE IF NOT EXISTS epoch_weights 
                      (epoch_id INTEGER PRIMARY KEY AUTOINCREMENT, 
                       w1 REAL, w2 REAL, w3 REAL, w4 REAL, 
                       w5 REAL, w6 REAL, w7 REAL, w8 REAL, timestamp DATETIME)''')
    
    cursor.execute("SELECT COUNT(*) FROM epoch_weights")
    if cursor.fetchone()[0] == 0:
        cursor.execute('''INSERT INTO epoch_weights (w1, w2, w3, w4, w5, w6, w7, w8, timestamp) 
                          VALUES (0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, ?)''', 
                       (datetime.now().isoformat(),))
    conn.commit()
    return conn

conn = init_system()

# --- 3. RECURSIVE ENTROPY WEIGHT METHOD ---
def calculate_ewm_weights(matrix):
    m, n = matrix.shape
    if m <= 1:
        return np.ones(n) / n 
    
    norm_matrix = np.zeros_like(matrix)
    for j in range(n):
        col = matrix[:, j]
        c_min, c_max = col.min(), col.max()
        if c_max - c_min > 1e-9:
            norm_matrix[:, j] = (col - c_min) / (c_max - c_min)
        else:
            norm_matrix[:, j] = 0.5 

    col_sums = norm_matrix.sum(axis=0)
    col_sums[col_sums == 0] = 1e-9 
    p_matrix = norm_matrix / col_sums
    
    p_matrix_eps = np.where(p_matrix == 0, 1e-12, p_matrix)
    entropy = - (1.0 / np.log(m)) * np.sum(p_matrix * np.log(p_matrix_eps), axis=0)
    
    d = 1.0 - entropy
    d_sum = d.sum()
    if d_sum == 0:
        return np.ones(n) / n
    return d / d_sum

def trigger_epoch_recalculation():
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp FROM epoch_weights ORDER BY epoch_id DESC LIMIT 1")
    last_epoch_date = datetime.fromisoformat(cursor.fetchone()[0])
    
    if datetime.now() - last_epoch_date >= timedelta(days=EPOCH_DAYS):
        target_date = (datetime.now() - timedelta(days=EPOCH_DAYS)).isoformat()
        cursor.execute("SELECT c1, c2, c3, c4, c5, c6, c7, c8 FROM papers_assessment WHERE timestamp >= ?", (target_date,))
        rows = cursor.fetchall()
        
        if len(rows) > 5:
            new_weights = calculate_ewm_weights(np.array(rows))
            cursor.execute('''INSERT INTO epoch_weights (w1, w2, w3, w4, w5, w6, w7, w8, timestamp) 
                              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                           (*new_weights, datetime.now().isoformat()))
            conn.commit()

# --- 4. SEMANTIC LLM EXTRACTION & MATHEMATICAL DRIFT ---
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
    """Calculates non-linear epistemic drift mapped to criteria variance and alignment offset."""
    mu = np.mean(scores)
    sigma = np.std(scores)
    delta = (100.0 - alignment) / 100.0
    
    # Utilizing logistic decay modified by standard deviation of quality to amplify structural divergence
    drift_metric = 100.0 * (1.0 - np.exp(-3.0 * (delta ** 1.5) * (1.0 + (sigma / 100.0)) / (0.1 + (mu / 100.0))))
    return float(max(0.0, min(100.0, drift_metric)))

def get_recommendation_spectrum(score, drift):
    """Generates a multi-tier continuous spectrum rather than static categories."""
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
    
    try:
        raw_data = evaluate_pdf_text(text, scope, PRIMARY_MODEL)
    except RateLimitError:
        time.sleep(2)
        raw_data = evaluate_pdf_text(text, scope, FALLBACK_MODEL)
        
    cursor.execute("SELECT w1, w2, w3, w4, w5, w6, w7, w8 FROM epoch_weights ORDER BY epoch_id DESC LIMIT 1")
    weights = cursor.fetchone()
    
    scores_dict = raw_data.get("scores", {})
    scores = [scores_dict.get(k, 50.0) for k in ["C1_Originality", "C2_Methodological_Rigor", "C3_Interdisciplinary", "C4_Societal_Impact", "C5_Open_Science_Potential", "C6_Literature_Integration", "C7_Empirical_Density", "C8_Future_Actionability"]]
    
    scope_alignment = raw_data.get("Scope_Alignment", 50.0)
    title = raw_data.get("Extracted_Title", filename)
    fields = raw_data.get("fields", ["General Science"])
    subfields = raw_data.get("subfields", ["General"])
    
    final_score = float(np.dot(scores, weights))
    drift = calculate_complex_drift(scope_alignment, scores)
    
    cursor.execute('''INSERT INTO papers_assessment 
                      (eval_hash, title, filename, scope, c1, c2, c3, c4, c5, c6, c7, c8, scope_alignment, subfields, fields, final_score, timestamp) 
                      VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                   (file_hash, title, filename, scope, *scores,
                    scope_alignment,
                    json.dumps(subfields), json.dumps(fields), final_score, datetime.now().isoformat()))
    conn.commit()
    trigger_epoch_recalculation()
    
    return title, final_score, drift, get_recommendation_spectrum(final_score, drift), fields, subfields, scores_dict

# --- 5. TOPOLOGICAL MAPPING (3D REALISTIC BUBBLE CHART) ---
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
    
    # Calculate frequencies
    df_topics = pd.DataFrame(all_topics)
    topic_counts = df_topics.groupby(['topic', 'category']).size().reset_index(name='count')
    topic_counts = topic_counts.sort_values(by='count', ascending=False).reset_index(drop=True)
    
    # Normalize sizes for the bubbles
    max_count = topic_counts['count'].max()
    min_size = 35
    max_size = 110
    topic_counts['bubble_size'] = min_size + (topic_counts['count'] / max_count) * (max_size - min_size)
    
    # Create organic scatter positions
    np.random.seed(SEED_NUMBER)
    topic_counts['x'] = np.random.normal(0, 1.5, len(topic_counts))
    topic_counts['y'] = np.random.normal(0, 1.5, len(topic_counts))
    
    # Add repulsion to avoid overlap
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
    
    # Use a broad qualitative color palette
    color_palette = px.colors.qualitative.Bold + px.colors.qualitative.Pastel + px.colors.qualitative.Vivid
    
    # Map topics independently to legend and colors
    for i, row in topic_counts.iterrows():
        topic = row['topic']
        size = row['bubble_size']
        count = row['count']
        
        # 3D Gradient effect styling
        fig.add_trace(go.Scatter(
            x=[row['x']], y=[row['y']],
            mode='markers+text',
            marker=dict(
                size=size,
                color=color_palette[i % len(color_palette)],
                line=dict(width=1, color='rgba(255, 255, 255, 0.4)'),
                sizemode='diameter',
                gradient=dict(type='radial', color='rgba(255, 255, 255, 0.85)'), # Realistic 3D glossy reflection
                opacity=0.95
            ),
            # Only display inline text if bubble is large enough to fit it
            text=topic if size > 45 else "",
            textposition="middle center",
            textfont=dict(color='#2c3e50', size=11, family="Arial Black"),
            name=topic, # Legend will display the specific Field/Subfield Name
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

# --- 6. USER INTERFACE ---
st.title("π-Index Assessment Engine")
st.markdown("**Upload papers, define your scope of research, let π-index filter noise and have better results**")

with st.expander("View π-Index Grading Criteria & Theoretical Formulations"):
    st.markdown("### Evaluation Metrics (0 - 100 Scale)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**C1: Originality**  \nEvaluates the uniqueness of the hypothesis, approach, or findings through epistemic gradient fields.")
        st.markdown(r"$$O = \lim_{\Delta t \to 0} \oint_{\partial \Omega} \frac{\nabla \times (\mathcal{H}_{novel} \otimes \mathcal{K}_{epistemic})}{\iint_{\mathcal{M}} \sum_{i=1}^N (\zeta_i \cdot \mathcal{I}_{existing}^{(i)}) \, d\mu} \cdot d\mathbf{S} \times 100$$")
        
        st.markdown("**C2: Methodological Rigor**  \nAssesses robustness and reproducibility via error-covariance tensors and persistent homology.")
        st.markdown(r"$$R = \left( 1 - \frac{\mathrm{tr}(\boldsymbol{\Sigma}_{error} \boldsymbol{\Lambda}^{-1})}{\det(\boldsymbol{\mu}_{signal} \otimes \mathbf{W})} \right) \cdot \prod_{k=1}^{m} \int_{0}^{\infty} \rho_k(x) e^{-\beta x^2} \Gamma\left(k+\frac{1}{2}\right) dx \times 100$$")
        
        st.markdown("**C3: Interdisciplinary**  \nMeasures network bridge capacity using generalized Rényi entropy over disciplinary graphs.")
        st.markdown(r"$$I = \left( \frac{1}{1-\alpha} \ln \left( \sum_{j=1}^{K} p_j^\alpha \right) + \sum_{i,j} \frac{A_{ij} \phi_i \phi_j}{\sqrt{d_i d_j}} \right) \cdot \frac{\Xi(\mathcal{G})}{\ln K \cdot \mathcal{Z}_{norm}} \times 100$$")
        
        st.markdown("**C4: Societal Impact**  \nProjects real-world macro applications utilizing fractional stochastic integration.")
        st.markdown(r"$$S = \frac{1}{\Gamma(q)} \int_{t_0}^{t_\infty} (t_\infty - \tau)^{q-1} e^{-\gamma(\tau) \tau} \cdot \Theta\left[ \sum_{v \in \mathcal{V}} \omega_v U_v(\tau, \mathbf{x}) \right] d\tau \times 100$$")

    with col2:
        st.markdown("**C5: Open Science Potential**  \nGauges transparent reporting optimization via multi-objective integration over FAIR limits.")
        st.markdown(r"$$O_s = \frac{\sum_{\ell \in \mathcal{L}} \alpha_\ell \mathcal{D}_{open}^{(\ell)} + \beta \iint_{\mathcal{C}} \nabla \cdot \mathbf{J}_{code} \, dV}{\max \left( \sup_{t} \mathcal{D}_{total}(t), \inf_{\epsilon>0} \mathcal{C}_{total}(\epsilon) \right)} \times \mathcal{P}_{FAIR} \times 100$$")
        
        st.markdown("**C6: Literature Integration**  \nEvaluates topological foundational embedding via non-Euclidean manifold PageRank distances.")
        st.markdown(r"$$L = \frac{1}{\mathcal{N}} \sum_{i=1}^{\mathcal{N}} \int_{\mathcal{M}} e^{-\lambda d_g(x_i, x_{core})} R(x_i) \sqrt{g} \, dx_i \cdot \frac{\text{PR}(x_i)}{\sum_j \text{PR}(x_j)} \times 100$$")
        
        st.markdown("**C7: Empirical Density**  \nEvaluates data depth utilizing Fisher information metrics and Kullback-Leibler divergences.")
        st.markdown(r"$$E_d = \tanh \left( \frac{\det \mathcal{I}_{Fisher}(\hat{\theta}) \cdot \mathbb{E}_{P}\left[\log\frac{P}{Q}\right]}{\mathcal{V}_{baseline} \cdot \oint_\Gamma \omega_{data}} \right) \times \sum_{d=1}^D \lambda_d \kappa_d \times 100$$")
        
        st.markdown("**C8: Future Actionability**  \nDetermines theoretical continuation potential using Lyapunov exponents on phase space logistics.")
        st.markdown(r"$$F_a = \frac{1}{\mathcal{Z}} \int_{\mathcal{X}} \frac{1}{1 + \exp\left(-\sum_{k=1}^K w_k(\eta_k(\mathbf{x}) - \eta_{0,k}) + \Lambda_{Lyapunov}\right)} d\mu(\mathbf{x}) \times 100$$")

tab1, tab2, tab3 = st.tabs(["Batch Assessment", "Scope Cartography", "Weight Matrix"])

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
            
            # Merging Fields and Subfields into a single display string for Column 4
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
        
        # DataFrame Processing
        df = pd.DataFrame(results)
        df_display = df.sort_values(by=["π-Index (0-100)"], ascending=False)
        
        st.markdown("### Assessment Summary")
        st.dataframe(df_display, use_container_width=True, hide_index=True)
            
        csv = df_display.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Summary as CSV", data=csv, file_name="pi_index_assessment_results.csv", mime="text/csv")

with tab2:
    st.subheader("Field & Subfield Epistemic Bubbles")
    st.write("Visualizing the disciplines and specializations involved in your uploaded literature. Bubble size correlates with topic frequency.")
    
    if research_scope:
        fig = generate_bubble_chart(research_scope)
        if fig: 
            st.plotly_chart(fig, use_container_width=True)
        else: 
            st.info("Awaiting sufficient data for this scope.")
    else:
        st.info("Please define a research scope in the 'Batch Assessment' tab first.")

with tab3:
    st.subheader("Recursive Weight Adaptations (EWM)")
    cursor = conn.cursor()
    cursor.execute("SELECT w1, w2, w3, w4, w5, w6, w7, w8, timestamp FROM epoch_weights ORDER BY epoch_id DESC LIMIT 1")
    weights = cursor.fetchone()
    
    if weights:
        st.caption(f"Last Matrix Update: {weights[8]}")
        cols = st.columns(4)
        labels = ["Originality", "Method Rigor", "Interdisciplinary", "Societal Impact", "Open Science", "Lit Integration", "Empirical Density", "Actionability"]
        for i, col in enumerate(cols * 2):
            if i < 8: col.metric(labels[i], f"{(weights[i]*100):.2f}%")

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray; font-size: 0.8em;'>Framework Author: Ali Vafadar Yengejeh | Università degli Studi di Milano-Bicocca</div>", unsafe_allow_html=True)

```
