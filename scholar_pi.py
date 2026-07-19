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
st.set_page_config(page_title="Pi-Index XAI Batch Triage", layout="wide")

PRIMARY_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"
MAX_TEXT_TOKENS = 6000
EPOCH_DAYS = 30
SEED_NUMBER = 42

BASE_DIR = os.path.abspath('./Scientometric_Pi_Index')
os.makedirs(BASE_DIR, exist_ok=True)
DB_PATH = os.path.join(BASE_DIR, 'merged_pi_index_xai.db')

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
    
    # Merged papers_triage table with XAI 'rationale' and 'scope'
    cursor.execute('''CREATE TABLE IF NOT EXISTS papers_triage 
                      (eval_hash TEXT PRIMARY KEY, title TEXT, filename TEXT, scope TEXT,
                       c1 REAL, c2 REAL, c3 REAL, c4 REAL, 
                       c5 REAL, c6 REAL, c7 REAL, c8 REAL, 
                       rationale TEXT, scope_alignment REAL,
                       keywords TEXT, departments TEXT, final_score REAL, timestamp DATETIME)''')
                       
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
        cursor.execute("SELECT c1, c2, c3, c4, c5, c6, c7, c8 FROM papers_triage WHERE timestamp >= ?", (target_date,))
        rows = cursor.fetchall()
        
        if len(rows) > 5:
            new_weights = calculate_ewm_weights(np.array(rows))
            cursor.execute('''INSERT INTO epoch_weights (w1, w2, w3, w4, w5, w6, w7, w8, timestamp) 
                              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                           (*new_weights, datetime.now().isoformat()))
            conn.commit()

# --- 4. SEMANTIC LLM EXTRACTION & XAI ---
def evaluate_pdf_text(text, scope, model):
    prompt = f"""You are an expert peer reviewer contributing to the Pi-Index.
The user is a researcher currently working on this specific project/scope: "{scope}"

Analyze the following excerpt from an academic paper (usually Title, Abstract, Intro).
1. Extract the Title.
2. Evaluate 'Scope_Alignment' from 0.0 to 10.0 (10.0 = highly relevant to scope, 0.0 = completely unrelated).
3. Evaluate the 8 Pi-Index criteria (0.0 to 10.0).
4. Provide a short 'justification' for each of the 8 scores.
5. Identify 5 research keywords.
6. Map to up to 3 standard Science Departments.

Return ONLY a valid JSON object matching exactly this structure:
{{
    "Extracted_Title": "Full title of the paper",
    "Scope_Alignment": 8.5,
    "scores": {{
        "C1_Originality": 8.0, "C2_Methodological_Rigor": 7.0, 
        "C3_Interdisciplinary": 6.0, "C4_Societal_Impact": 5.0, 
        "C5_Open_Science_Potential": 6.0, "C6_Literature_Integration": 7.0, 
        "C7_Empirical_Density": 8.0, "C8_Future_Actionability": 7.0
    }},
    "justification": {{
        "C1_Originality": "Reasoning here...", 
        "C2_Methodological_Rigor": "Reasoning here...",
        "C3_Interdisciplinary": "Reasoning here...",
        "C4_Societal_Impact": "Reasoning here...",
        "C5_Open_Science_Potential": "Reasoning here...",
        "C6_Literature_Integration": "Reasoning here...",
        "C7_Empirical_Density": "Reasoning here...",
        "C8_Future_Actionability": "Reasoning here..."
    }},
    "keywords": ["kw1", "kw2", "kw3", "kw4", "kw5"],
    "departments": ["Dept1", "Dept2"]
}}

Text: {text[:MAX_TEXT_TOKENS]}
"""
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model, temperature=0.1, seed=SEED_NUMBER, response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

def get_recommendation(score, drift):
    if score >= 6.5 and drift <= 30.0: return "Highly Recommended"
    elif score >= 6.5 and drift > 30.0: return "Read with Caution (Scope Drift)"
    elif score < 6.5 and drift <= 30.0: return "Borderline (In Scope, Low Quality)"
    else: return "Skip / Discard"

def process_single_pdf(file_bytes, filename, scope):
    file_hash = hashlib.sha256(file_bytes + scope.encode('utf-8')).hexdigest()
    
    cursor = conn.cursor()
    cursor.execute("SELECT final_score, scope_alignment, title, rationale, departments, c1, c2, c3, c4, c5, c6, c7, c8 FROM papers_triage WHERE eval_hash=?", (file_hash,))
    cached = cursor.fetchone()
    
    if cached:
        score, alignment, title, rationale_str, depts_str, c1, c2, c3, c4, c5, c6, c7, c8 = cached
        depts = json.loads(depts_str) if depts_str else ["General Science"]
        drift = max(0.0, min(100.0, (10.0 - alignment) * 10))
        rationale = json.loads(rationale_str) if rationale_str else {}
        scores_dict = {
            "C1_Originality": c1, "C2_Methodological_Rigor": c2,
            "C3_Interdisciplinary": c3, "C4_Societal_Impact": c4,
            "C5_Open_Science_Potential": c5, "C6_Literature_Integration": c6,
            "C7_Empirical_Density": c7, "C8_Future_Actionability": c8
        }
        return title, score, drift, get_recommendation(score, drift), rationale, depts, scores_dict

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
    scores = [scores_dict.get(k, 5.0) for k in ["C1_Originality", "C2_Methodological_Rigor", "C3_Interdisciplinary", "C4_Societal_Impact", "C5_Open_Science_Potential", "C6_Literature_Integration", "C7_Empirical_Density", "C8_Future_Actionability"]]
    
    scope_alignment = raw_data.get("Scope_Alignment", 5.0)
    title = raw_data.get("Extracted_Title", filename)
    depts = raw_data.get("departments", ["General Science"])
    rationale = raw_data.get("justification", {})
    
    final_score = float(np.dot(scores, weights))
    drift = max(0.0, min(100.0, (10.0 - scope_alignment) * 10))
    
    cursor.execute('''INSERT INTO papers_triage 
                      (eval_hash, title, filename, scope, c1, c2, c3, c4, c5, c6, c7, c8, rationale, scope_alignment, keywords, departments, final_score, timestamp) 
                      VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                   (file_hash, title, filename, scope, *scores,
                    json.dumps(rationale), scope_alignment,
                    json.dumps(raw_data.get("keywords", [])), json.dumps(depts), final_score, datetime.now().isoformat()))
    conn.commit()
    trigger_epoch_recalculation()
    
    return title, final_score, drift, get_recommendation(final_score, drift), rationale, depts, scores_dict

# --- 5. TOPOLOGICAL MAPPING (CENTERED NETWORK) ---
def generate_centered_network(scope):
    cursor = conn.cursor()
    cursor.execute("SELECT title, keywords, departments FROM papers_triage WHERE scope=?", (scope,))
    data = cursor.fetchall()
    
    if not data: return None
    
    G = nx.Graph()
    topic_node = f"Topic: {scope}"
    G.add_node(topic_node, type='topic')
    
    for title, kw_json, dept_json in data:
        try:
            keywords = [k.title().strip() for k in json.loads(kw_json)]
            depts = [d.title().strip() for d in json.loads(dept_json)]
            
            G.add_node(title, type='paper')
            G.add_edge(topic_node, title)
            
            for dept in depts:
                G.add_node(dept, type='department')
                G.add_edge(title, dept)
            for kw in keywords:
                G.add_node(kw, type='keyword')
                G.add_edge(title, kw)
        except: continue
            
    pos = nx.spring_layout(G, k=0.7, seed=SEED_NUMBER)
    pos[topic_node] = np.array([0, 0])  # Lock scope to center
    
    edge_x, edge_y = [], []
    for edge in G.edges():
        edge_x.extend([pos[edge[0]][0], pos[edge[1]][0], None])
        edge_y.extend([pos[edge[0]][1], pos[edge[1]][1], None])
        
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#ccc'), mode='lines', hoverinfo='none')
    
    node_traces = []
    types = {'topic': ('#f39c12', 25, 'star'), 'paper': ('#2ecc71', 12, 'circle'), 
             'department': ('#e74c3c', 16, 'square'), 'keyword': ('#3498db', 8, 'circle')}
             
    for n_type, (color, size, symbol) in types.items():
        nodes = [n for n, d in G.nodes(data=True) if d.get('type') == n_type]
        if not nodes: continue
        trace = go.Scatter(
            x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes],
            mode='markers+text' if n_type in ['topic', 'department'] else 'markers',
            text=[f"<b>{n}</b>" for n in nodes] if n_type in ['topic', 'department'] else "",
            textposition="bottom center",
            hovertext=nodes, hoverinfo="text",
            marker=dict(symbol=symbol, size=size, color=color, line=dict(width=1, color='white')),
            name=n_type.capitalize()
        )
        node_traces.append(trace)
                                        
    return go.Figure(data=[edge_trace] + node_traces, layout=go.Layout(
        showlegend=True, hovermode='closest', margin=dict(b=0,l=0,r=0,t=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), 
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    ))

# --- 6. USER INTERFACE ---
st.title("Pi-Index XAI Batch Triage Engine")
st.markdown("**Upload PDFs, define your scope, let the Pi-Index filter the noise, and explain its reasoning.**")

tab1, tab2, tab3 = st.tabs(["Batch Triage & XAI", "Scope Cartography", "Weight Matrix"])

with tab1:
    research_scope = st.text_input("Define your specific Research Topic / Scope", placeholder="e.g., Use of transformer models for predicting protein folding...")
    group_by_dept = st.checkbox("Group summary table by Primary Scientific Department")
    
    uploaded_files = st.file_uploader("Upload Academic Papers (PDFs)", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Run Batch Triage", type="primary") and uploaded_files and research_scope:
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, file in enumerate(uploaded_files):
            status_text.text(f"Analyzing {i+1} of {len(uploaded_files)}: {file.name}...")
            if i > 0: time.sleep(1.5) 
            
            title, score, drift, rec, rationale, depts, scores_dict = process_single_pdf(file.read(), file.name, research_scope)
            primary_dept = depts[0] if depts else "Uncategorized"
            
            results.append({
                "Filename": file.name,
                "Extracted Title": title,
                "Primary Department": primary_dept,
                "All Departments": ", ".join(depts),
                "Pi-Index": round(score, 3),
                "C1: Originality": scores_dict.get("C1_Originality", 0.0),
                "C2: Rigor": scores_dict.get("C2_Methodological_Rigor", 0.0),
                "C3: Interdisciplinary": scores_dict.get("C3_Interdisciplinary", 0.0),
                "C4: Societal Impact": scores_dict.get("C4_Societal_Impact", 0.0),
                "C5: Open Science": scores_dict.get("C5_Open_Science_Potential", 0.0),
                "C6: Lit Integration": scores_dict.get("C6_Literature_Integration", 0.0),
                "C7: Empirical Density": scores_dict.get("C7_Empirical_Density", 0.0),
                "C8: Actionability": scores_dict.get("C8_Future_Actionability", 0.0),
                "Scope Drift %": round(drift, 1),
                "Recommendation": rec,
                "Rationale": rationale
            })
            progress_bar.progress((i + 1) / len(uploaded_files))
            
        status_text.text("Batch processing complete!")
        
        # DataFrame Processing
        df = pd.DataFrame(results)
        df_display = df.drop(columns=["Rationale"])  # Hide rationale from the main table
        df_display = df_display.sort_values(by=["Recommendation", "Pi-Index"], ascending=[False, False])
        
        st.markdown("### Triage Summary")
        if group_by_dept:
            grouped = df_display.groupby("Primary Department")
            for dept, group in grouped:
                st.markdown(f"#### {dept}")
                st.dataframe(group.drop(columns=["Primary Department"]), use_container_width=True)
        else:
            st.dataframe(df_display, use_container_width=True)
            
        csv = df_display.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Summary as CSV", data=csv, file_name="pi_index_triage_results.csv", mime="text/csv")

        # Explainability (XAI) Expanders
        st.markdown("### Evaluation Rationale (XAI)")
        for i, row in df.iterrows():
            with st.expander(f"{row['Extracted Title']} (Score: {row['Pi-Index']})"):
                st.markdown(f"**Recommendation:** {row['Recommendation']} | **Drift:** {row['Scope Drift %']}%")
                for crit, reason in row['Rationale'].items():
                    st.markdown(f"**{crit}**: {reason}")

with tab2:
    st.subheader("Scope-Centered Epistemic Network")
    st.write("Visualizes how your uploaded papers branch out into specific departments and keywords around your topic.")
    
    # Text input syncs with Tab 1, or can be changed directly. Re-renders automatically on input.
    search_scope = st.text_input("Enter scope to map:", value=research_scope)
    
    if search_scope:
        fig = generate_centered_network(search_scope)
        if fig: 
            st.plotly_chart(fig, use_container_width=True)
        else: 
            st.info("Awaiting sufficient data for this scope.")

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
