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
import streamlit as st
import fitz  # PyMuPDF
from groq import Groq, RateLimitError

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

# --- 4. SEMANTIC LLM EXTRACTION ---
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

def get_recommendation(score, drift):
    if drift > 50.0: 
        return "Unrelated"
    elif score >= 65.0: 
        return "Recommended"
    else: 
        return "Borderline"

def process_single_pdf(file_bytes, filename, scope):
    file_hash = hashlib.sha256(file_bytes + scope.encode('utf-8')).hexdigest()
    
    cursor = conn.cursor()
    cursor.execute("SELECT final_score, scope_alignment, title, fields, subfields, c1, c2, c3, c4, c5, c6, c7, c8 FROM papers_assessment WHERE eval_hash=?", (file_hash,))
    cached = cursor.fetchone()
    
    if cached:
        score, alignment, title, fields_str, subfields_str, c1, c2, c3, c4, c5, c6, c7, c8 = cached
        fields = json.loads(fields_str) if fields_str else ["General Science"]
        subfields = json.loads(subfields_str) if subfields_str else ["General"]
        drift = max(0.0, min(100.0, 100.0 - alignment))
        scores_dict = {
            "C1_Originality": c1, "C2_Methodological_Rigor": c2,
            "C3_Interdisciplinary": c3, "C4_Societal_Impact": c4,
            "C5_Open_Science_Potential": c5, "C6_Literature_Integration": c6,
            "C7_Empirical_Density": c7, "C8_Future_Actionability": c8
        }
        return title, score, drift, get_recommendation(score, drift), fields, subfields, scores_dict

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
    drift = max(0.0, min(100.0, 100.0 - scope_alignment))
    
    cursor.execute('''INSERT INTO papers_assessment 
                      (eval_hash, title, filename, scope, c1, c2, c3, c4, c5, c6, c7, c8, scope_alignment, subfields, fields, final_score, timestamp) 
                      VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                   (file_hash, title, filename, scope, *scores,
                    scope_alignment,
                    json.dumps(subfields), json.dumps(fields), final_score, datetime.now().isoformat()))
    conn.commit()
    trigger_epoch_recalculation()
    
    return title, final_score, drift, get_recommendation(final_score, drift), fields, subfields, scores_dict

# --- 5. TOPOLOGICAL MAPPING (2-CIRCLE VENN DIAGRAM) ---
def generate_venn_network(scope):
    cursor = conn.cursor()
    cursor.execute("SELECT fields, subfields FROM papers_assessment WHERE scope=?", (scope,))
    data = cursor.fetchall()
    
    if not data: return None
    
    G = nx.Graph()
    field_nodes, subfield_nodes = set(), set()
    
    for fields_json, subfields_json in data:
        try:
            fields = [f.title().strip() for f in json.loads(fields_json)]
            subfields = [s.title().strip() for s in json.loads(subfields_json)]
            
            for f in fields:
                G.add_node(f, type='field')
                field_nodes.add(f)
            for s in subfields:
                G.add_node(s, type='subfield')
                subfield_nodes.add(s)
                
            for f in fields:
                for s in subfields:
                    G.add_edge(f, s)
        except: continue
            
    if not field_nodes and not subfield_nodes: return None

    initial_pos = {}
    
    for f in field_nodes: 
        initial_pos[f] = [-0.5 + np.random.uniform(-0.3, 0.3), np.random.uniform(-0.5, 0.5)]
    for s in subfield_nodes: 
        initial_pos[s] = [0.5 + np.random.uniform(-0.3, 0.3), np.random.uniform(-0.5, 0.5)]

    pos = nx.spring_layout(G, pos=initial_pos, k=0.05, iterations=15, seed=SEED_NUMBER)
        
    node_traces = []
    types = {'field': ('#2ecc71', 16, 'square', 'Fields'), 
             'subfield': ('#3498db', 12, 'circle', 'Subfields')}
             
    for n_type, (color, size, symbol, name) in types.items():
        nodes = [n for n, d in G.nodes(data=True) if d.get('type') == n_type]
        if not nodes: continue
        trace = go.Scatter(
            x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes],
            mode='markers+text',
            text=[f"<b>{n}</b>" for n in nodes],
            textposition="top center",
            hovertext=nodes, hoverinfo="text",
            marker=dict(symbol=symbol, size=size, color=color, line=dict(width=1, color='white')),
            name=name
        )
        node_traces.append(trace)
        
    fig = go.Figure(data=node_traces)
    
    fig.update_layout(
        shapes=[
            dict(type="circle", xref="x", yref="y", x0=-1.5, y0=-1.0, x1=0.5, y1=1.0, fillcolor="rgba(46, 204, 113, 0.15)", line_color="rgba(46, 204, 113, 0.4)", layer="below"),
            dict(type="circle", xref="x", yref="y", x0=-0.5, y0=-1.0, x1=1.5, y1=1.0, fillcolor="rgba(52, 152, 219, 0.15)", line_color="rgba(52, 152, 219, 0.4)", layer="below"),
        ],
        showlegend=True, hovermode='closest', margin=dict(b=0,l=0,r=0,t=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.8, 1.8]), 
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.2, 1.2])
    )
    
    fig.add_annotation(x=-1.0, y=1.1, text="Fields", showarrow=False, font=dict(size=16, color="#2ecc71"))
    fig.add_annotation(x=1.0, y=1.1, text="Subfields", showarrow=False, font=dict(size=16, color="#3498db"))
                                        
    return fig

# --- 6. USER INTERFACE ---
st.title("π-Index Assessment Engine")
st.markdown("**Upload papers, define your scope of research, let π-index filter noise and have better results**")

with st.expander("View π-Index Grading Criteria & Theoretical Formulations"):
    st.markdown("### Evaluation Metrics (0 - 100 Scale)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**C1: Originality**")
        st.markdown("Evaluates the uniqueness of the hypothesis, approach, or findings through epistemic gradient fields.")
        st.latex(r"$$O = \lim_{\Delta t \to 0} \oint_{\partial \Omega} \frac{\nabla \times (\mathcal{H}_{novel} \otimes \mathcal{K}_{epistemic})}{\iint_{\mathcal{M}} \sum_{i=1}^N (\zeta_i \cdot \mathcal{I}_{existing}^{(i)}) \, d\mu} \cdot d\mathbf{S} \times 100$$")
        
        st.markdown("**C2: Methodological Rigor**")
        st.markdown("Assesses robustness and reproducibility via error-covariance tensors and persistent homology.")
        st.latex(r"$$R = \left( 1 - \frac{\mathrm{tr}(\boldsymbol{\Sigma}_{error} \boldsymbol{\Lambda}^{-1})}{\det(\boldsymbol{\mu}_{signal} \otimes \mathbf{W})} \right) \cdot \prod_{k=1}^{m} \int_{0}^{\infty} \rho_k(x) e^{-\beta x^2} \Gamma\left(k+\frac{1}{2}\right) dx \times 100$$")
        
        st.markdown("**C3: Interdisciplinary**")
        st.markdown("Measures network bridge capacity using generalized Rényi entropy over disciplinary graphs.")
        st.latex(r"$$I = \left( \frac{1}{1-\alpha} \ln \left( \sum_{j=1}^{K} p_j^\alpha \right) + \sum_{i,j} \frac{A_{ij} \phi_i \phi_j}{\sqrt{d_i d_j}} \right) \cdot \frac{\Xi(\mathcal{G})}{\ln K \cdot \mathcal{Z}_{norm}} \times 100$$")
        
        st.markdown("**C4: Societal Impact**")
        st.markdown("Projects real-world macro applications utilizing fractional stochastic integration.")
        st.latex(r"$$S = \frac{1}{\Gamma(q)} \int_{t_0}^{t_\infty} (t_\infty - \tau)^{q-1} e^{-\gamma(\tau) \tau} \cdot \Theta\left[ \sum_{v \in \mathcal{V}} \omega_v U_v(\tau, \mathbf{x}) \right] d\tau \times 100$$")

    with col2:
        st.markdown("**C5: Open Science Potential**")
        st.markdown("Gauges transparent reporting optimization via multi-objective integration over FAIR limits.")
        st.latex(r"$$O_s = \frac{\sum_{\ell \in \mathcal{L}} \alpha_\ell \mathcal{D}_{open}^{(\ell)} + \beta \iint_{\mathcal{C}} \nabla \cdot \mathbf{J}_{code} \, dV}{\max \left( \sup_{t} \mathcal{D}_{total}(t), \inf_{\epsilon>0} \mathcal{C}_{total}(\epsilon) \right)} \times \mathcal{P}_{FAIR} \times 100$$")
        
        st.markdown("**C6: Literature Integration**")
        st.markdown("Evaluates topological foundational embedding via non-Euclidean manifold PageRank distances.")
        st.latex(r"$$L = \frac{1}{\mathcal{N}} \sum_{i=1}^{\mathcal{N}} \int_{\mathcal{M}} e^{-\lambda d_g(x_i, x_{core})} R(x_i) \sqrt{g} \, dx_i \cdot \frac{\text{PR}(x_i)}{\sum_j \text{PR}(x_j)} \times 100$$")
        
        st.markdown("**C7: Empirical Density**")
        st.markdown("Evaluates data depth utilizing Fisher information metrics and Kullback-Leibler divergences.")
        st.latex(r"$$E_d = \tanh \left( \frac{\det \mathcal{I}_{Fisher}(\hat{\theta}) \cdot \mathbb{E}_{P}\left[\log\frac{P}{Q}\right]}{\mathcal{V}_{baseline} \cdot \oint_\Gamma \omega_{data}} \right) \times \sum_{d=1}^D \lambda_d \kappa_d \times 100$$")
        
        st.markdown("**C8: Future Actionability**")
        st.markdown("Determines theoretical continuation potential using Lyapunov exponents on phase space logistics.")
        st.latex(r"$$F_a = \frac{1}{\mathcal{Z}} \int_{\mathcal{X}} \frac{1}{1 + \exp\left(-\sum_{k=1}^K w_k(\eta_k(\mathbf{x}) - \eta_{0,k}) + \Lambda_{Lyapunov}\right)} d\mu(\mathbf{x}) \times 100$$")

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
            
            # Merging Fields and Subfields into a single display string
            combined_fields = f"Fields: {', '.join(fields)} | Subfields: {', '.join(subfields)}"
            
            results.append({
                "No.": i + 1,
                "File Name": file.name,
                "Topic": research_scope,
                "Fields & Subfields": combined_fields,
                "π-Index (0-100)": round(score, 1),
                "Recommendation": rec,
                "Scope Drift %": round(drift, 1),
                "C1": scores_dict.get("C1_Originality", 0.0),
                "C2": scores_dict.get("C2_Methodological_Rigor", 0.0),
                "C3": scores_dict.get("C3_Interdisciplinary", 0.0),
                "C4": scores_dict.get("C4_Societal Impact", 0.0),
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
    st.subheader("Field & Subfield Epistemic Network")
    st.write("Visualizing the disciplines and specializations involved in your uploaded literature.")
    
    if research_scope:
        fig = generate_venn_network(research_scope)
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
