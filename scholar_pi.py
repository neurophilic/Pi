import os
import sqlite3
import json
import hashlib
import time
from datetime import datetime, timedelta
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import streamlit as st
from groq import Groq, RateLimitError

# --- 1. CONFIGURATION & ENVIRONMENT ---
st.set_page_config(page_title="Pi-Index Triage (π-Index)", layout="wide")

PRIMARY_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"
EPOCH_DAYS = 30
SEED_NUMBER = 42

BASE_DIR = os.path.abspath('./Scientometric_Pi_Index')
os.makedirs(BASE_DIR, exist_ok=True)
DB_PATH = os.path.join(BASE_DIR, 'triage_pi_index.db')

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
    
    # Updated Table for abstract-based assessments + Scope Alignment
    cursor.execute('''CREATE TABLE IF NOT EXISTS papers_triage 
                      (eval_hash TEXT PRIMARY KEY, title TEXT, 
                       c1 REAL, c2 REAL, c3 REAL, c4 REAL, 
                       c5 REAL, c6 REAL, c7 REAL, c8 REAL, scope_alignment REAL,
                       keywords TEXT, departments TEXT, final_score REAL, timestamp DATETIME)''')
                       
    # Table for historical 30-day epoch EWM weights
    cursor.execute('''CREATE TABLE IF NOT EXISTS epoch_weights 
                      (epoch_id INTEGER PRIMARY KEY AUTOINCREMENT, 
                       w1 REAL, w2 REAL, w3 REAL, w4 REAL, 
                       w5 REAL, w6 REAL, w7 REAL, w8 REAL, timestamp DATETIME)''')
    
    # Initialize default uniform weights if database is empty (12.5% each)
    cursor.execute("SELECT COUNT(*) FROM epoch_weights")
    if cursor.fetchone()[0] == 0:
        cursor.execute('''INSERT INTO epoch_weights (w1, w2, w3, w4, w5, w6, w7, w8, timestamp) 
                          VALUES (0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, ?)''', 
                       (datetime.now().isoformat(),))
    conn.commit()
    return conn

conn = init_system()

# --- 3. RECURSIVE ENTROPY WEIGHT METHOD (EWM) ALGORITHM ---
def calculate_ewm_weights(matrix):
    """Calculates objective criteria weights using Shannon Entropy."""
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
    """Evaluates temporal delta and recursively adjusts the Pi Index weights."""
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp FROM epoch_weights ORDER BY epoch_id DESC LIMIT 1")
    last_epoch_str = cursor.fetchone()[0]
    last_epoch_date = datetime.fromisoformat(last_epoch_str)
    
    if datetime.now() - last_epoch_date >= timedelta(days=EPOCH_DAYS):
        st.toast(f"⏳ {EPOCH_DAYS}-Day Recursion reached. Executing EWM recalibration...", icon="🔄")
        
        target_date = (datetime.now() - timedelta(days=EPOCH_DAYS)).isoformat()
        cursor.execute("SELECT c1, c2, c3, c4, c5, c6, c7, c8 FROM papers_triage WHERE timestamp >= ?", (target_date,))
        rows = cursor.fetchall()
        
        if len(rows) > 5:
            matrix = np.array(rows)
            new_weights = calculate_ewm_weights(matrix)
            
            cursor.execute('''INSERT INTO epoch_weights (w1, w2, w3, w4, w5, w6, w7, w8, timestamp) 
                              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                           (*new_weights, datetime.now().isoformat()))
            conn.commit()
            st.toast("✅ Pi Index successfully adapted to current scientific variance.", icon="📈")

# --- 4. SEMANTIC LLM EXTRACTION & CLASSIFICATION ---
def evaluate_paper(title, abstract, scope, model):
    """Leverages LLM to extract metrics, keywords, departments, and scope drift."""
    prompt = f"""You are an expert peer reviewer contributing to the rigorous Pi-Index database.
The user is a researcher currently working on the following specific project/scope:
"{scope}"

Analyze the following academic paper Title and Abstract.
Evaluate the paper across the 8 Pi-Index criteria, assigning a strict score from 0.0 to 10.0 for each.
CRITICAL TASK: Evaluate 'Scope_Alignment' from 0.0 to 10.0. (10.0 = highly relevant to the researcher's scope, 0.0 = completely unrelated).
Identify 5 specific research keywords.
Map this paper to up to 3 standard Science Departments (e.g., "Quantum Physics", "Computational Biology").

Return ONLY a valid JSON object matching this exact structure:
{{
    "C1_Originality": <float>,
    "C2_Methodological_Rigor": <float>,
    "C3_Interdisciplinary": <float>,
    "C4_Societal_Impact": <float>,
    "C5_Open_Science_Potential": <float>,
    "C6_Literature_Integration": <float>,
    "C7_Empirical_Density": <float>,
    "C8_Future_Actionability": <float>,
    "Scope_Alignment": <float>,
    "keywords": ["kw1", "kw2", "kw3", "kw4", "kw5"],
    "departments": ["Dept1", "Dept2"]
}}

Title: {title}
Abstract: {abstract}
"""
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model, temperature=0.1, seed=SEED_NUMBER, response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

def process_triage(title, abstract, scope):
    # Hash includes the scope! Same paper + different scope = new evaluation
    hash_input = f"{title}|{abstract}|{scope}".encode('utf-8')
    eval_hash = hashlib.sha256(hash_input).hexdigest()
    
    cursor = conn.cursor()
    cursor.execute("SELECT final_score, c1, c2, c3, c4, c5, c6, c7, c8, departments, scope_alignment FROM papers_triage WHERE eval_hash=?", (eval_hash,))
    cached = cursor.fetchone()
    
    if cached:
        depts = json.loads(cached[9]) if cached[9] else []
        return cached[0], list(cached[1:9]), depts, cached[10], True, False
        
    used_fallback = False
    try:
        raw_scores = evaluate_paper(title, abstract, scope, PRIMARY_MODEL)
    except RateLimitError:
        used_fallback = True
        st.toast(f"⚠️ Rate limit exceeded. Failing over to {FALLBACK_MODEL}.", icon="🔄")
        time.sleep(2)
        raw_scores = evaluate_paper(title, abstract, scope, FALLBACK_MODEL)
        
    cursor.execute("SELECT w1, w2, w3, w4, w5, w6, w7, w8 FROM epoch_weights ORDER BY epoch_id DESC LIMIT 1")
    weights = cursor.fetchone()
    
    scores = [
        raw_scores.get("C1_Originality", 5.0), 
        raw_scores.get("C2_Methodological_Rigor", 5.0), 
        raw_scores.get("C3_Interdisciplinary", 5.0), 
        raw_scores.get("C4_Societal_Impact", 5.0),
        raw_scores.get("C5_Open_Science_Potential", 5.0), 
        raw_scores.get("C6_Literature_Integration", 5.0),
        raw_scores.get("C7_Empirical_Density", 5.0), 
        raw_scores.get("C8_Future_Actionability", 5.0)
    ]
    
    scope_alignment = raw_scores.get("Scope_Alignment", 5.0)
    departments = raw_scores.get("departments", ["General Science"])
    
    # Calculate final dynamic Pi-Index score (Score Vector · Weight Vector)
    final_score = float(np.dot(scores, weights))
    
    cursor.execute('''INSERT INTO papers_triage 
                      (eval_hash, title, c1, c2, c3, c4, c5, c6, c7, c8, scope_alignment, keywords, departments, final_score, timestamp) 
                      VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                   (eval_hash, title, *scores, scope_alignment,
                    json.dumps(raw_scores.get("keywords", [])), 
                    json.dumps(departments),
                    final_score, datetime.now().isoformat()))
    conn.commit()
    
    trigger_epoch_recalculation()
    return final_score, scores, departments, scope_alignment, False, used_fallback

# --- 5. TOPOLOGICAL SCIENCE MAPPING (BIPARTITE) ---
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
                    if G.has_edge(dept, kw):
                        G[dept][kw]['weight'] += 1
                    else:
                        G.add_edge(dept, kw, weight=1)
                        
            for i in range(len(keywords)):
                for j in range(i+1, len(keywords)):
                    if G.has_edge(keywords[i], keywords[j]):
                        G[keywords[i]][keywords[j]]['weight'] += 0.5
                    else:
                        G.add_edge(keywords[i], keywords[j], weight=0.5)
        except Exception:
            continue
            
    if len(G.nodes) == 0:
        return None

    pos = nx.spring_layout(G, k=0.6, seed=SEED_NUMBER)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.3, color='#AAAAAA'), hoverinfo='none', mode='lines')
    
    dept_nodes, kw_nodes = [], []
    for node, data in G.nodes(data=True):
        if data.get('type') == 'department':
            dept_nodes.append(node)
        else:
            kw_nodes.append(node)
            
    kw_x, kw_y = [pos[n][0] for n in kw_nodes], [pos[n][1] for n in kw_nodes]
    kw_text = [f"{n} (Freq: {int(G.degree(n, weight='weight'))})" for n in kw_nodes]
    kw_trace = go.Scatter(x=kw_x, y=kw_y, mode='markers+text', text=kw_nodes,
                          textposition="top center", hoverinfo='text', hovertext=kw_text,
                          marker=dict(showscale=False, color='#3498db', size=8, line_width=1, line_color='white'),
                          textfont=dict(size=9, color='#555'))

    dept_x, dept_y = [pos[n][0] for n in dept_nodes], [pos[n][1] for n in dept_nodes]
    dept_text = [f"🏛️ {n} (Centrality: {G.degree(n)})" for n in dept_nodes]
    dept_trace = go.Scatter(x=dept_x, y=dept_y, mode='markers+text', text=[f"<b>{n}</b>" for n in dept_nodes],
                            textposition="bottom center", hoverinfo='text', hovertext=dept_text,
                            marker=dict(symbol='square', size=18, color='#e74c3c', line_width=2, line_color='black'),
                            textfont=dict(size=12, color='black'))
                                        
    fig = go.Figure(data=[edge_trace, kw_trace, dept_trace],
                    layout=go.Layout(
                        title='The Pi-Index Epistemic Cartography: Departmental Intersections',
                        showlegend=False, hovermode='closest',
                        plot_bgcolor='rgba(250,250,250,1)',
                        margin=dict(b=0,l=0,r=0,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    return fig

# --- 6. USER INTERFACE ---
st.title("🏛️ The Pi Index ($\pi$-Index) - Triage Engine")
st.markdown("**Quantify Paper Quality & Prevent Scope Drift in Your Literature Review**")

tab1, tab2, tab3 = st.tabs(["📄 Triage a Paper", "🌌 Departmental Cartography", "⚙️ The Recursive Loop"])

with tab1:
    st.markdown("Paste the Title and Abstract of a paper, and define your current research scope. The engine will evaluate the paper's intrinsic quality ($\pi$-Index) and tell you how far off-topic it is (Scope Drift).")
    
    research_scope = st.text_input("🎯 Your Current Research Topic / Scope", placeholder="e.g., Use of transformer models for predicting protein folding...")
    paper_title = st.text_input("📑 Paper Title")
    paper_abstract = st.text_area("📝 Paper Abstract", height=200)
    
    if st.button("Evaluate Paper", type="primary"):
        if not paper_title or not paper_abstract or not research_scope:
            st.warning("⚠️ Please provide your research scope, paper title, and abstract.")
        else:
            with st.spinner("Decoding semantic density and evaluating scope drift..."):
                score, raw_scores, depts, scope_alignment, cached, used_fallback = process_triage(paper_title, paper_abstract, research_scope)
                
                # Math for Scope Drift: 10.0 alignment = 0% drift, 0.0 alignment = 100% drift
                drift_percentage = max(0.0, min(100.0, (10.0 - scope_alignment) * 10))
                
                # --- RECOMMENDATION LOGIC ---
                st.markdown("### 📊 Triage Recommendation")
                
                if score >= 6.5 and drift_percentage <= 30.0:
                    st.success(f"**🌟 Highly Recommended**\n\nGreat $\pi$-Index ({score:.2f}) and minimal scope drift ({drift_percentage:.0f}%). This paper is high-quality and directly relevant to your research.")
                elif score >= 6.5 and drift_percentage > 30.0:
                    st.warning(f"**⚠️ Warning: Scope Drift**\n\nWhile the intrinsic quality is good ($\pi$-Index: {score:.2f}), this paper drifts heavily ({drift_percentage:.0f}%) from your stated scope. Read with caution to avoid getting distracted.")
                elif score < 6.5 and drift_percentage <= 30.0:
                    st.info(f"**🔍 Borderline (In Scope, Low Quality)**\n\nIt matches your topic well (Drift: {drift_percentage:.0f}%), but the $\pi$-Index is relatively low ({score:.2f}). Might be worth skimming, but don't rely heavily on it.")
                else:
                    st.error(f"**🚫 Do Not Read (Discard)**\n\nLow quality ($\pi$-Index: {score:.2f}) AND highly irrelevant to your current scope (Drift: {drift_percentage:.0f}%). Skip this paper.")
                
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                col1.metric("Aggregate $\pi$-Index Score", f"{score:.3f} / 10.000")
                col2.metric("Scope Drift", f"{drift_percentage:.1f}%", delta=f"{scope_alignment}/10 Alignment", delta_color="off")
                
                if depts:
                    st.markdown(f"**Affiliated Departments:** `{'` • `'.join(depts)}`")
                
                st.markdown("### Epistemic Vector Breakdown")
                labels = ["C1: Originality", "C2: Methodological Rigor", "C3: Interdisciplinary Synthesis", 
                          "C4: Societal Impact", "C5: Open Science", "C6: Literature Integration", 
                          "C7: Empirical Density", "C8: Future Actionability"]
                cols = st.columns(4)
                for i, col in enumerate(cols * 2):
                    if i < 8:
                        col.metric(labels[i], f"{raw_scores[i]:.2f}")

with tab2:
    st.subheader("Global Epistemic Network")
    st.write("This topological projection maps granular research topics (blue nodes) to their overarching Scientific Departments (red squares).")
    fig = generate_trend_network()
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Awaiting sufficient abstract inputs to generate the network projection.")

with tab3:
    st.subheader("The $\pi$-Index Recursive Algorithm")
    st.markdown("""
    The $\pi$-Index employs Shannon Information Entropy to adaptively weigh scoring criteria based on current scientific trends.
    
    * **High Entropy (Uniformity):** When the scientific community masters a criterion, its variance drops, and the $\pi$-Index reduces its weighting.
    * **Low Entropy (Disruption):** Criteria where manuscript scores are highly volatile represent the current frontier. The $\pi$-Index assigns them heavier weights to reward true breakthroughs.
    """)
    
    cursor = conn.cursor()
    cursor.execute("SELECT w1, w2, w3, w4, w5, w6, w7, w8, timestamp FROM epoch_weights ORDER BY epoch_id DESC LIMIT 1")
    weights = cursor.fetchone()
    
    if weights:
        st.caption(f"Last Recursive Matrix Update: {weights[8]}")
        cols = st.columns(4)
        labels = ["C1: Originality", "C2: Method Rigor", "C3: Interdisciplinary", "C4: Societal Impact", 
                  "C5: Open Science", "C6: Lit Integration", "C7: Empirical Density", "C8: Actionability"]
        for i, col in enumerate(cols * 2):
            if i < 8:
                col.metric(labels[i], f"{(weights[i]*100):.2f}%")
