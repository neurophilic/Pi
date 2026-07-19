import os
import sqlite3
import json
import hashlib
import time
import io
from datetime import datetime, timedeltaimport os
import sqlite3
import json
import hashlib
import time
import io
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import streamlit as st
import fitz  # PyMuPDF
from groq import Groq, RateLimitError

# --- 1. CONFIGURATION & ENVIRONMENT ---
st.set_page_config(page_title="Pi-Index Batch Triage", page_icon="🏛️", layout="wide")

PRIMARY_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"
MAX_TEXT_TOKENS = 6000
EPOCH_DAYS = 30
SEED_NUMBER = 42

BASE_DIR = os.path.abspath('./Scientometric_Pi_Index')
os.makedirs(BASE_DIR, exist_ok=True)
DB_PATH = os.path.join(BASE_DIR, 'batch_triage_pi_index.db')

GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    st.error("⚠️ API Key not found! Please configure your environment variables or Streamlit Secrets.")
    st.stop()
client = Groq(api_key=GROQ_API_KEY)

# --- 2. DATABASE INITIALIZATION ---
@st.cache_resource
def init_system():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS papers_triage 
                      (eval_hash TEXT PRIMARY KEY, title TEXT, filename TEXT,
                       c1 REAL, c2 REAL, c3 REAL, c4 REAL, 
                       c5 REAL, c6 REAL, c7 REAL, c8 REAL, scope_alignment REAL,
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

# --- 4. SEMANTIC LLM EXTRACTION ---
def evaluate_pdf_text(text, scope, model):
    prompt = f"""You are an expert peer reviewer contributing to the Pi-Index.
The user is a researcher currently working on this specific project/scope: "{scope}"

Analyze the following excerpt from an academic paper (usually Title, Abstract, Intro).
1. Extract the Title.
2. Evaluate 'Scope_Alignment' from 0.0 to 10.0 (10.0 = highly relevant to scope, 0.0 = completely unrelated).
3. Evaluate the 8 Pi-Index criteria (0.0 to 10.0).
4. Identify 5 research keywords.
5. Map to up to 3 standard Science Departments.

Return ONLY a valid JSON object:
{{
    "Extracted_Title": "Full title of the paper",
    "Scope_Alignment": <float>,
    "C1_Originality": <float>,
    "C2_Methodological_Rigor": <float>,
    "C3_Interdisciplinary": <float>,
    "C4_Societal_Impact": <float>,
    "C5_Open_Science_Potential": <float>,
    "C6_Literature_Integration": <float>,
    "C7_Empirical_Density": <float>,
    "C8_Future_Actionability": <float>,
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
    if score >= 6.5 and drift <= 30.0:
        return "🌟 Highly Recommended"
    elif score >= 6.5 and drift > 30.0:
        return "⚠️ Read with Caution (Scope Drift)"
    elif score < 6.5 and drift <= 30.0:
        return "🔍 Borderline (In Scope, Low Quality)"
    else:
        return "🚫 Skip / Discard"

def process_single_pdf(file_bytes, filename, scope):
    # Hash the file contents PLUS the user's scope. 
    # If they change their scope, it re-evaluates the drift.
    file_hash = hashlib.sha256(file_bytes + scope.encode('utf-8')).hexdigest()
    
    cursor = conn.cursor()
    cursor.execute("SELECT final_score, scope_alignment, title FROM papers_triage WHERE eval_hash=?", (file_hash,))
    cached = cursor.fetchone()
    
    if cached:
        score, alignment, title = cached
        drift = max(0.0, min(100.0, (10.0 - alignment) * 10))
        return title, score, drift, get_recommendation(score, drift)

    # Extract text from the first 3 pages (covers Title + Abstract without blowing up tokens)
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = " ".join([page.get_text() for page in doc[:3]])
    
    try:
        raw_data = evaluate_pdf_text(text, scope, PRIMARY_MODEL)
    except RateLimitError:
        time.sleep(2)
        raw_data = evaluate_pdf_text(text, scope, FALLBACK_MODEL)
        
    cursor.execute("SELECT w1, w2, w3, w4, w5, w6, w7, w8 FROM epoch_weights ORDER BY epoch_id DESC LIMIT 1")
    weights = cursor.fetchone()
    
    scores = [raw_data.get(f"C{i}_...", 5.0) if f"C{i}_..." in raw_data else raw_data.get(list(raw_data.keys())[i+2], 5.0) for i in range(1,9)]
    scores = [raw_data.get(k, 5.0) for k in ["C1_Originality", "C2_Methodological_Rigor", "C3_Interdisciplinary", "C4_Societal_Impact", "C5_Open_Science_Potential", "C6_Literature_Integration", "C7_Empirical_Density", "C8_Future_Actionability"]]
    
    scope_alignment = raw_data.get("Scope_Alignment", 5.0)
    title = raw_data.get("Extracted_Title", filename)
    depts = raw_data.get("departments", ["General Science"])
    
    final_score = float(np.dot(scores, weights))
    drift = max(0.0, min(100.0, (10.0 - scope_alignment) * 10))
    
    cursor.execute('''INSERT INTO papers_triage 
                      (eval_hash, title, filename, c1, c2, c3, c4, c5, c6, c7, c8, scope_alignment, keywords, departments, final_score, timestamp) 
                      VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                   (file_hash, title, filename, *scores, scope_alignment,
                    json.dumps(raw_data.get("keywords", [])), json.dumps(depts), final_score, datetime.now().isoformat()))
    conn.commit()
    trigger_epoch_recalculation()
    
    return title, final_score, drift, get_recommendation(final_score, drift)

# --- 5. TOPOLOGICAL MAPPING ---
def generate_trend_network():
    cursor = conn.cursor()
    target_date = (datetime.now() - timedelta(days=EPOCH_DAYS)).isoformat()
    cursor.execute("SELECT keywords, departments FROM papers_triage WHERE timestamp >= ?", (target_date,))
    
    G = nx.Graph()
    for row in cursor.fetchall():
        try:
            keywords = [k.title().strip() for k in json.loads(row[0])]
            depts = [d.title().strip() for d in (json.loads(row[1]) if row[1] else ["General Science"])]
            for dept in depts:
                G.add_node(dept, type='department')
                for kw in keywords:
                    G.add_node(kw, type='keyword')
                    if G.has_edge(dept, kw): G[dept][kw]['weight'] += 1
                    else: G.add_edge(dept, kw, weight=1)
            for i in range(len(keywords)):
                for j in range(i+1, len(keywords)):
                    if G.has_edge(keywords[i], keywords[j]): G[keywords[i]][keywords[j]]['weight'] += 0.5
                    else: G.add_edge(keywords[i], keywords[j], weight=0.5)
        except: continue
            
    if len(G.nodes) == 0: return None

    pos = nx.spring_layout(G, k=0.6, seed=SEED_NUMBER)
    edge_x, edge_y = [], []
    for edge in G.edges():
        edge_x.extend([pos[edge[0]][0], pos[edge[1]][0], None])
        edge_y.extend([pos[edge[0]][1], pos[edge[1]][1], None])
        
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.3, color='#AAA'), mode='lines', hoverinfo='none')
    
    dept_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'department']
    kw_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'keyword']
            
    kw_trace = go.Scatter(x=[pos[n][0] for n in kw_nodes], y=[pos[n][1] for n in kw_nodes], mode='markers+text',
                          text=kw_nodes, textposition="top center", hovertext=[f"{n} (Freq: {int(G.degree(n, weight='weight'))})" for n in kw_nodes],
                          marker=dict(color='#3498db', size=8), textfont=dict(size=9))

    dept_trace = go.Scatter(x=[pos[n][0] for n in dept_nodes], y=[pos[n][1] for n in dept_nodes], mode='markers+text',
                            text=[f"<b>{n}</b>" for n in dept_nodes], textposition="bottom center",
                            hovertext=[f"🏛️ {n}" for n in dept_nodes], marker=dict(symbol='square', size=18, color='#e74c3c'), textfont=dict(size=12, color='black'))
                                        
    return go.Figure(data=[edge_trace, kw_trace, dept_trace], layout=go.Layout(showlegend=False, hovermode='closest', margin=dict(b=0,l=0,r=0,t=0), xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

# --- 6. USER INTERFACE ---
st.title("📚 Batch PDF Triage Engine")
st.markdown("**Upload dozens of PDFs, define your scope, and let the $\pi$-Index filter out the noise.**")

tab1, tab2, tab3 = st.tabs(["📥 Batch Triage", "🌌 Cartography", "⚙️ Weight Matrix"])

with tab1:
    research_scope = st.text_input("🎯 Define your specific Research Topic / Scope", placeholder="e.g., Use of transformer models for predicting protein folding...")
    uploaded_files = st.file_uploader("Upload Academic Papers (PDFs)", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Run Batch Triage", type="primary") and uploaded_files and research_scope:
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, file in enumerate(uploaded_files):
            status_text.text(f"Analyzing {i+1} of {len(uploaded_files)}: {file.name}...")
            
            # Rate limit protection between files
            if i > 0: time.sleep(1.5) 
            
            title, score, drift, rec = process_single_pdf(file.read(), file.name, research_scope)
            results.append({
                "Filename": file.name,
                "Extracted Title": title,
                "π-Index (0-10)": round(score, 3),
                "Scope Drift %": round(drift, 1),
                "Recommendation": rec
            })
            progress_bar.progress((i + 1) / len(uploaded_files))
            
        status_text.text("✅ Batch processing complete!")
        
        # Display Results
        df = pd.DataFrame(results)
        df = df.sort_values(by=["Recommendation", "π-Index (0-10)"], ascending=[False, False])
        
        st.markdown("### 📊 Triage Results")
        st.dataframe(df, use_container_width=True)
        
        # CSV Export
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Download Results as CSV", data=csv, file_name="pi_index_triage_results.csv", mime="text/csv")

with tab2:
    st.subheader("Global Epistemic Network")
    st.write("Maps granular research topics (blue nodes) to overarching Scientific Departments (red squares).")
    fig = generate_trend_network()
    if fig: st.plotly_chart(fig, use_container_width=True)
    else: st.warning("Awaiting sufficient data.")

with tab3:
    st.subheader("Recursive Weight Adaptations")
    cursor = conn.cursor()
    cursor.execute("SELECT w1, w2, w3, w4, w5, w6, w7, w8, timestamp FROM epoch_weights ORDER BY epoch_id DESC LIMIT 1")
    weights = cursor.fetchone()
    if weights:
        st.caption(f"Last Matrix Update: {weights[8]}")
        cols = st.columns(4)
        labels = ["Originality", "Method Rigor", "Interdisciplinary", "Societal Impact", "Open Science", "Lit Integration", "Empirical Density", "Actionability"]
        for i, col in enumerate(cols * 2):
            if i < 8: col.metric(labels[i], f"{(weights[i]*100):.2f}%")

import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import streamlit as st
import fitz  # PyMuPDF
from groq import Groq, RateLimitError

# --- 1. CONFIGURATION & ENVIRONMENT ---
st.set_page_config(page_title="Pi-Index Batch Triage", page_icon="🏛️", layout="wide")

PRIMARY_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"
MAX_TEXT_TOKENS = 6000
EPOCH_DAYS = 30
SEED_NUMBER = 42

BASE_DIR = os.path.abspath('./Scientometric_Pi_Index')
os.makedirs(BASE_DIR, exist_ok=True)
DB_PATH = os.path.join(BASE_DIR, 'batch_triage_pi_index.db')

GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    st.error("⚠️ API Key not found! Please configure your environment variables or Streamlit Secrets.")
    st.stop()
client = Groq(api_key=GROQ_API_KEY)

# --- 2. DATABASE INITIALIZATION ---
@st.cache_resource
def init_system():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS papers_triage 
                      (eval_hash TEXT PRIMARY KEY, title TEXT, filename TEXT,
                       c1 REAL, c2 REAL, c3 REAL, c4 REAL, 
                       c5 REAL, c6 REAL, c7 REAL, c8 REAL, scope_alignment REAL,
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

# --- 4. SEMANTIC LLM EXTRACTION ---
def evaluate_pdf_text(text, scope, model):
    prompt = f"""You are an expert peer reviewer contributing to the Pi-Index.
The user is a researcher currently working on this specific project/scope: "{scope}"

Analyze the following excerpt from an academic paper (usually Title, Abstract, Intro).
1. Extract the Title.
2. Evaluate 'Scope_Alignment' from 0.0 to 10.0 (10.0 = highly relevant to scope, 0.0 = completely unrelated).
3. Evaluate the 8 Pi-Index criteria (0.0 to 10.0).
4. Identify 5 research keywords.
5. Map to up to 3 standard Science Departments.

Return ONLY a valid JSON object:
{{
    "Extracted_Title": "Full title of the paper",
    "Scope_Alignment": <float>,
    "C1_Originality": <float>,
    "C2_Methodological_Rigor": <float>,
    "C3_Interdisciplinary": <float>,
    "C4_Societal_Impact": <float>,
    "C5_Open_Science_Potential": <float>,
    "C6_Literature_Integration": <float>,
    "C7_Empirical_Density": <float>,
    "C8_Future_Actionability": <float>,
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
    if score >= 6.5 and drift <= 30.0:
        return "🌟 Highly Recommended"
    elif score >= 6.5 and drift > 30.0:
        return "⚠️ Read with Caution (Scope Drift)"
    elif score < 6.5 and drift <= 30.0:
        return "🔍 Borderline (In Scope, Low Quality)"
    else:
        return "🚫 Skip / Discard"

def process_single_pdf(file_bytes, filename, scope):
    # Hash the file contents PLUS the user's scope. 
    # If they change their scope, it re-evaluates the drift.
    file_hash = hashlib.sha256(file_bytes + scope.encode('utf-8')).hexdigest()
    
    cursor = conn.cursor()
    cursor.execute("SELECT final_score, scope_alignment, title FROM papers_triage WHERE eval_hash=?", (file_hash,))
    cached = cursor.fetchone()
    
    if cached:
        score, alignment, title = cached
        drift = max(0.0, min(100.0, (10.0 - alignment) * 10))
        return title, score, drift, get_recommendation(score, drift)

    # Extract text from the first 3 pages (covers Title + Abstract without blowing up tokens)
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = " ".join([page.get_text() for page in doc[:3]])
    
    try:
        raw_data = evaluate_pdf_text(text, scope, PRIMARY_MODEL)
    except RateLimitError:
        time.sleep(2)
        raw_data = evaluate_pdf_text(text, scope, FALLBACK_MODEL)
        
    cursor.execute("SELECT w1, w2, w3, w4, w5, w6, w7, w8 FROM epoch_weights ORDER BY epoch_id DESC LIMIT 1")
    weights = cursor.fetchone()
    
    scores = [raw_data.get(f"C{i}_...", 5.0) if f"C{i}_..." in raw_data else raw_data.get(list(raw_data.keys())[i+2], 5.0) for i in range(1,9)]
    scores = [raw_data.get(k, 5.0) for k in ["C1_Originality", "C2_Methodological_Rigor", "C3_Interdisciplinary", "C4_Societal_Impact", "C5_Open_Science_Potential", "C6_Literature_Integration", "C7_Empirical_Density", "C8_Future_Actionability"]]
    
    scope_alignment = raw_data.get("Scope_Alignment", 5.0)
    title = raw_data.get("Extracted_Title", filename)
    depts = raw_data.get("departments", ["General Science"])
    
    final_score = float(np.dot(scores, weights))
    drift = max(0.0, min(100.0, (10.0 - scope_alignment) * 10))
    
    cursor.execute('''INSERT INTO papers_triage 
                      (eval_hash, title, filename, c1, c2, c3, c4, c5, c6, c7, c8, scope_alignment, keywords, departments, final_score, timestamp) 
                      VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                   (file_hash, title, filename, *scores, scope_alignment,
                    json.dumps(raw_data.get("keywords", [])), json.dumps(depts), final_score, datetime.now().isoformat()))
    conn.commit()
    trigger_epoch_recalculation()
    
    return title, final_score, drift, get_recommendation(final_score, drift)

# --- 5. TOPOLOGICAL MAPPING ---
def generate_trend_network():
    cursor = conn.cursor()
    target_date = (datetime.now() - timedelta(days=EPOCH_DAYS)).isoformat()
    cursor.execute("SELECT keywords, departments FROM papers_triage WHERE timestamp >= ?", (target_date,))
    
    G = nx.Graph()
    for row in cursor.fetchall():
        try:
            keywords = [k.title().strip() for k in json.loads(row[0])]
            depts = [d.title().strip() for d in (json.loads(row[1]) if row[1] else ["General Science"])]
            for dept in depts:
                G.add_node(dept, type='department')
                for kw in keywords:
                    G.add_node(kw, type='keyword')
                    if G.has_edge(dept, kw): G[dept][kw]['weight'] += 1
                    else: G.add_edge(dept, kw, weight=1)
            for i in range(len(keywords)):
                for j in range(i+1, len(keywords)):
                    if G.has_edge(keywords[i], keywords[j]): G[keywords[i]][keywords[j]]['weight'] += 0.5
                    else: G.add_edge(keywords[i], keywords[j], weight=0.5)
        except: continue
            
    if len(G.nodes) == 0: return None

    pos = nx.spring_layout(G, k=0.6, seed=SEED_NUMBER)
    edge_x, edge_y = [], []
    for edge in G.edges():
        edge_x.extend([pos[edge[0]][0], pos[edge[1]][0], None])
        edge_y.extend([pos[edge[0]][1], pos[edge[1]][1], None])
        
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.3, color='#AAA'), mode='lines', hoverinfo='none')
    
    dept_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'department']
    kw_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'keyword']
            
    kw_trace = go.Scatter(x=[pos[n][0] for n in kw_nodes], y=[pos[n][1] for n in kw_nodes], mode='markers+text',
                          text=kw_nodes, textposition="top center", hovertext=[f"{n} (Freq: {int(G.degree(n, weight='weight'))})" for n in kw_nodes],
                          marker=dict(color='#3498db', size=8), textfont=dict(size=9))

    dept_trace = go.Scatter(x=[pos[n][0] for n in dept_nodes], y=[pos[n][1] for n in dept_nodes], mode='markers+text',
                            text=[f"<b>{n}</b>" for n in dept_nodes], textposition="bottom center",
                            hovertext=[f"🏛️ {n}" for n in dept_nodes], marker=dict(symbol='square', size=18, color='#e74c3c'), textfont=dict(size=12, color='black'))
                                        
    return go.Figure(data=[edge_trace, kw_trace, dept_trace], layout=go.Layout(showlegend=False, hovermode='closest', margin=dict(b=0,l=0,r=0,t=0), xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

# --- 6. USER INTERFACE ---
st.title("📚 Batch PDF Triage Engine")
st.markdown("**Upload dozens of PDFs, define your scope, and let the $\pi$-Index filter out the noise.**")

tab1, tab2, tab3 = st.tabs(["📥 Batch Triage", "🌌 Cartography", "⚙️ Weight Matrix"])

with tab1:
    research_scope = st.text_input("🎯 Define your specific Research Topic / Scope", placeholder="e.g., Use of transformer models for predicting protein folding...")
    uploaded_files = st.file_uploader("Upload Academic Papers (PDFs)", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Run Batch Triage", type="primary") and uploaded_files and research_scope:
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, file in enumerate(uploaded_files):
            status_text.text(f"Analyzing {i+1} of {len(uploaded_files)}: {file.name}...")
            
            # Rate limit protection between files
            if i > 0: time.sleep(1.5) 
            
            title, score, drift, rec = process_single_pdf(file.read(), file.name, research_scope)
            results.append({
                "Filename": file.name,
                "Extracted Title": title,
                "π-Index (0-10)": round(score, 3),
                "Scope Drift %": round(drift, 1),
                "Recommendation": rec
            })
            progress_bar.progress((i + 1) / len(uploaded_files))
            
        status_text.text("✅ Batch processing complete!")
        
        # Display Results
        df = pd.DataFrame(results)
        df = df.sort_values(by=["Recommendation", "π-Index (0-10)"], ascending=[False, False])
        
        st.markdown("### 📊 Triage Results")
        st.dataframe(df, use_container_width=True)
        
        # CSV Export
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Download Results as CSV", data=csv, file_name="pi_index_triage_results.csv", mime="text/csv")

with tab2:
    st.subheader("Global Epistemic Network")
    st.write("Maps granular research topics (blue nodes) to overarching Scientific Departments (red squares).")
    fig = generate_trend_network()
    if fig: st.plotly_chart(fig, use_container_width=True)
    else: st.warning("Awaiting sufficient data.")

with tab3:
    st.subheader("Recursive Weight Adaptations")
    cursor = conn.cursor()
    cursor.execute("SELECT w1, w2, w3, w4, w5, w6, w7, w8, timestamp FROM epoch_weights ORDER BY epoch_id DESC LIMIT 1")
    weights = cursor.fetchone()
    if weights:
        st.caption(f"Last Matrix Update: {weights[8]}")
        cols = st.columns(4)
        labels = ["Originality", "Method Rigor", "Interdisciplinary", "Societal Impact", "Open Science", "Lit Integration", "Empirical Density", "Actionability"]
        for i, col in enumerate(cols * 2):
            if i < 8: col.metric(labels[i], f"{(weights[i]*100):.2f}%")
