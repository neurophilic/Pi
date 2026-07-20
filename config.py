import os
import streamlit as st

# --- CONFIGURATION & ENVIRONMENT SETUP ---
PRIMARY_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"
MAX_TEXT_TOKENS = 12000
SEED_NUMBER = 42

# Epoch & Blockchain Settings
EPOCH_BLOCK_SIZE = 5

# File Paths
BASE_DIR = os.path.abspath('./Scientometric_Pi_Index')
os.makedirs(BASE_DIR, exist_ok=True)
DB_PATH = os.path.join(BASE_DIR, 'pi_index_main.db')

# Securely grab the API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")