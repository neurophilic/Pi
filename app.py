import os
import re
import json
import time
import requests
import colorsys
import tempfile
import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import EPOCH_BLOCK_SIZE
from blockchain import init_system
from math_engine import get_pi_float
from ai_engine import process_single_pdf, PiBlockchainDataset, PiBrainLSTM

st.set_page_config(page_title="π-Index Assessment Engine", layout="wide")
conn = init_system()

# --- UI Utilities ---
def verify_orcid_live(orcid_id):
    try:
        url = f"https://pub.orcid.org/v3.0/{orcid_id}/person"
        headers = {"Accept": "application/json"}
        response = requests.get(url, headers=headers, timeout=5)
        
        if response.status_code == 200:
            name_data = response.json().get('name', {})
            if name_data:
                given = name_data.get('given-names', {}).get('value', '') if name_data.get('given-names') else ''
                family = name_data.get('family-name', {}).get('value', '') if name_data.get('family-name') else ''
                return True, f"{given} {family}".strip() or "Verified Researcher (Name Private)"
            return True, "Verified Researcher (Name Private)"
        return False, "ORCID ID not found on public registry."
    except Exception as e:
        return False, f"API Error: {str(e)}"

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
    if topic_counts.empty: return html_string, table_html
        
    unique_topics = topic_counts['topic'].unique()
    
    def get_color(i, n):
        h, s, v = i/n if n > 0 else 0, 0.7, 0.9
        rgb = colorsys.hsv_to_rgb(h, s, v)
        return '#%02x%02x%02x' % tuple(int(x * 255) for x in rgb)
    
    color_map = {topic: get_color(i, len(unique_topics)) for i, topic in enumerate(unique_topics)}
    
    net = Network(height='600px', width='100%', bgcolor='#ffffff', font_color='#2c3e50', notebook=False)
    physics_options = """{ "physics": { "barnesHut": { "gravitationalConstant": -1000, "centralGravity": 1, "springLength": 100, "avoidOverlap": 1.0 }, "stabilization": { "enabled": true, "iterations": 500, "fit": true }, "solver": "barnesHut" } }"""
    net.set_options(physics_options)
    
    for _, row in topic_counts.iterrows():
        node_size = 30 + (row['weight'] * 2.5) 
        net.add_node(n_id=row['topic'], label=' ', title=f"Topic: {row['topic']} | Weight: {row['weight']}", size=node_size, physics=True, color=color_map[row['topic']])
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp_file:
        net.save_graph(tmp_file.name)
        with open(tmp_file.name, 'r', encoding='utf-8') as f: html_string = f.read()
    os.remove(tmp_file.name)

    unique_network_id = f"pi_network_{int(time.time() * 1000)}"
    html_string = html_string.replace('mynetwork', unique_network_id)

    table_html = "<style>.table-big { width: 100%; font-size: 14px; border-collapse: collapse; margin-top: 10px; font-family: sans-serif; } .table-big th { background-color: #2c3e50; color: white; padding: 10px; text-align: left; } .table-big td { border-bottom: 1px solid #ddd; padding: 8px; vertical-align: middle; } .color-box { width: 18px; height: 18px; display: inline-block; border-radius: 3px; border: 1px solid #ccc; margin: 0 auto;} .legend-container { max-height: 550px; overflow-y: auto; border: 1px solid #eee; }</style>"
    table_html += "<div class='legend-container'><table class='table-big'><thead><tr><th style='width: 25%; text-align: center;'>Color</th><th>Topic</th></tr></thead><tbody>"
    for _, row in topic_counts.sort_values(by="weight", ascending=False).iterrows():
        table_html += f"<tr><td style='text-align: center;'><div class='color-box' style='background-color:{color_map[row['topic']]};'></div></td><td>{row['topic']}</td></tr>"
    table_html += "</tbody></table></div>"
    
    return html_string, table_html

# --- UI LAYOUT ---
st.sidebar.title("System Access")

if 'assessment_update_token' not in st.session_state: st.session_state['assessment_update_token'] = time.time()
if 'orcid_id' not in st.session_state:
    st.session_state.orcid_id = "0000-0000-0000-0000"
    st.session_state.orcid_name = ""
    st.session_state.is_authenticated = False

if not st.session_state.is_authenticated:
    st.sidebar.markdown("### Authenticate via ORCID")
    manual_orcid = st.sidebar.text_input("Enter ORCID iD", placeholder="XXXX-XXXX-XXXX-XXXX")
    if st.sidebar.button("🔗 Validate & Connect"):
        clean_orcid = manual_orcid.strip()
        if re.match(r'^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$', clean_orcid):
            with st.sidebar.status("Connecting to ORCID Registry..."):
                is_valid, user_name = verify_orcid_live(clean_orcid)
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
st.markdown("**Upload papers, define your scope of research, let π-index filter noise and have better results**")

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

tab1, tab2, tab3, tab4 = st.tabs(["Batch Assessment", "Scope Cartography", "Active Epoch Constants", "π-Brain Neural Network"])

with tab1:
    research_scope = st.text_input("Define your specific Research Topic / Scope (Optional)", placeholder="e.g., Application of deep learning in vascular imaging...")
    uploaded_files = st.file_uploader("Upload Academic Papers (PDFs)", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Run Batch Assessment", type="primary"):
        if not uploaded_files: st.warning("Please upload at least one academic paper (PDF) to proceed.")
        else:
            results_list = []
            progress_bar, status_text = st.progress(0), st.empty()
            for i, file in enumerate(uploaded_files):
                status_text.text(f"Analyzing {i+1} of {len(uploaded_files)}: {file.name}...")
                title, author_name, score, logic_integrity, drift, rec, fields, subfields, scores_dict, eval_hash = process_single_pdf(file.read(), file.name, research_scope, current_user)
                
                record = {
                    "No.": i + 1, "File Name": file.name, "Primary Author": author_name, 
                    "Fields & Subfields": f"Fields: {', '.join(fields)} | Subfields: {', '.join(subfields)}",
                    "Logic Integrity (%)": round(logic_integrity, 1), "π-Index (0-100)": round(score, 1),
                }
                if research_scope.strip():
                    record.update({"Topic": research_scope, "Recommendation Spectrum": rec, "Scope Drift %": round(drift, 1) if drift != "N/A" else "N/A"})
                
                record.update({f"C{j+1}": round(scores_dict.get(list(scores_dict.keys())[j], 0.0), 1) for j in range(8)})
                record["Eval Hash"] = eval_hash
                results_list.append(record)
                progress_bar.progress((i + 1) / len(uploaded_files))
                
            status_text.success("Batch processing complete!")
            st.session_state['latest_assessment_results'] = pd.DataFrame(results_list)
            st.session_state['assessment_update_token'] = time.time()
            st.session_state['last_trained_blocks'] = -1
            
    if 'latest_assessment_results' in st.session_state:
        st.dataframe(st.session_state['latest_assessment_results'], use_container_width=True, hide_index=True)

    st.markdown("### Latest Assessment History")
    if st.session_state.is_authenticated:
        cursor = conn.cursor()
        cursor.execute("SELECT title, author_name, scope, final_score, timestamp, eval_hash FROM papers_assessment WHERE user_id=? ORDER BY timestamp DESC LIMIT 20", (current_user,))
        history_data = cursor.fetchall()
        if history_data: st.dataframe(pd.DataFrame(history_data, columns=["Paper Title", "Primary Author", "Scope", "π-Index Score", "Date", "Evaluation Hash"]), use_container_width=True, hide_index=True)
        else: st.info("No assessment history found.")
    else: st.warning("Please connect your ORCID iD in the sidebar.")

with tab2:
    st.subheader("Epistemic Bubbles (Author & Portfolio Cartography)")
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
        with col2: st.markdown("### Legend"); st.markdown(table_html, unsafe_allow_html=True)
    else: st.info("Awaiting sufficient data for this selection.")

with tab3:
    cursor = conn.cursor()
    cursor.execute("SELECT block_height, w1, w2, w3, w4, w5, w6, w7, w8, model_used, eval_hash, block_hash FROM blockchain_por_weights ORDER BY block_height DESC LIMIT 1")
    epoch_data = cursor.fetchone()
    
    if epoch_data:
        block_height, weights, model_used, eval_hash, block_hash = epoch_data[0], epoch_data[1:9], epoch_data[9], epoch_data[10], epoch_data[11]
        cursor.execute("SELECT COUNT(DISTINCT eval_hash) FROM blockchain_por_weights WHERE eval_hash != 'genesis'")
        total_papers_processed = cursor.fetchone()[0]

        st.markdown(f"**Processed:** `{total_papers_processed}` | **Block Size:** `{EPOCH_BLOCK_SIZE}` | **Model:** `{model_used}` | **Block:** `{block_height}` | **Pi Acc:** `{get_pi_float(block_height)}`")
        
        cols = st.columns(4)
        labels = [("C1", r"$\varpi_1$"), ("C2", r"$\varpi_2$"), ("C3", r"$\varpi_3$"), ("C4", r"$\varpi_4$"), ("C5", r"$\varpi_5$"), ("C6", r"$\varpi_6$"), ("C7", r"$\varpi_7$"), ("C8", r"$\varpi_8$")]
        for i, col in enumerate(cols * 2):
            if i < 8:
                col.markdown(f"**{labels[i][0]} ({labels[i][1]})**")
                col.markdown(f"<h3 style='margin-top:0px; margin-bottom:5px;'>{weights[i]:.6f}</h3>", unsafe_allow_html=True)
                
        st.markdown("### PoR Blockchain Explorer")
        explore_col1, explore_col2 = st.columns([3, 1])
        with explore_col1: search_query = st.text_input("Enter Document Evaluation Hash or Block Hash to verify ledger record...")
        with explore_col2: st.write(""); st.write(""); search_btn = st.button("Verify Record")
            
        if search_btn and search_query:
            cursor.execute("SELECT * FROM blockchain_por_weights WHERE block_hash=? OR eval_hash=?", (search_query, search_query))
            record = cursor.fetchone()
            if record:
                st.success("Valid Block Found on Ledger!")
                st.json({"Block Height": record[0], "Timestamp": record[9], "Model Used": record[13], "Validator Node": record[11], "Block Hash": record[12], "Evaluation Hash": record[14], "Weights Matrix": record[1:9]})
            else: st.error("No block matching that signature was found on the ledger.")

with tab4:
    st.subheader("π-Brain: Meta-Learning on the PoR Blockchain")
    cursor = conn.cursor()
    cursor.execute("SELECT w1, w2, w3, w4, w5, w6, w7, w8 FROM blockchain_por_weights ORDER BY block_height ASC")
    historical_rows = cursor.fetchall()
    
    lookback_window = 5
    if len(historical_rows) < lookback_window + 2:
        st.warning(f"Not enough blockchain data to train the meta-model. You need at least {lookback_window + 2} blocks.")
    else:
        current_block_count = len(historical_rows)
        if 'last_trained_blocks' not in st.session_state or st.session_state.last_trained_blocks != current_block_count:
            weight_data = np.array(historical_rows, dtype=np.float32)
            dataset = PiBlockchainDataset(weight_data, lookback_window)
            dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
            
            model, loss_function, optimizer = PiBrainLSTM(), nn.MSELoss(), optim.Adam(PiBrainLSTM().parameters(), lr=0.001)
            progress_bar, status_text = st.progress(0), st.empty()
            epochs = 200
            
            model.train()
            for epoch in range(epochs):
                total_loss = 0
                for seq, target in dataloader:
                    optimizer.zero_grad()
                    loss = loss_function(model(seq), target)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                if epoch % 10 == 0 or epoch == epochs - 1:
                    status_text.text(f"Training Epoch {epoch}/{epochs} | MSE Loss: {total_loss / len(dataloader):.6f}")
                    progress_bar.progress((epoch + 1) / epochs)
            
            model.eval()
            with torch.no_grad():
                st.session_state.predicted_next_weights = model(torch.tensor(weight_data[-lookback_window:], dtype=torch.float32).unsqueeze(0)).squeeze().numpy()
                st.session_state.current_weights = weight_data[-1]
                st.session_state.last_trained_blocks = current_block_count
        else:
            st.info("Meta-model is cached and up-to-date with the latest blockchain ledger.")

        df_compare = pd.DataFrame({"Current Active Weights": st.session_state.current_weights, "Predicted Next Epoch": st.session_state.predicted_next_weights}, index=["C1: Originality", "C2: Method Rigor", "C3: Interdisciplinary", "C4: Societal Impact", "C5: Open Science", "C6: Lit Integration", "C7: Empirical Density", "C8: Actionability"])
        st.bar_chart(df_compare, height=400)
        st.markdown(f"**Mathematical Constraint Check:** Predicted Sum = `{sum(st.session_state.predicted_next_weights):.6f}` / `8.0`")

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray; font-size: 0.8em;'>Framework Author: Ali Vafadar Yengejeh | Università degli Studi di Milano-Bicocca</div>", unsafe_allow_html=True)
