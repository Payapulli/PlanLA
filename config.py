# PlanLA API Configuration
# Securely loads API keys from Streamlit Cloud secrets or local .env file

import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables if running locally
load_dotenv()

def get_secret(key: str, default: str = "") -> str:
    """Fetch a secret from Streamlit Cloud or .env."""
    if key in st.secrets:
        return st.secrets[key]
    return os.getenv(key, default)

# LA Open Data Portal API Keys
LA_DATA_API_KEY = get_secret("LA_DATA_API_KEY")
LA_DATA_API_SECRET = get_secret("LA_DATA_API_SECRET")

# LA GeoHub API Key
LA_GEOHUB_API_KEY = get_secret("LA_GEOHUB_API_KEY")

# Hugging Face API Key (for free LLM access)
HF_API_KEY = get_secret("HF_API_KEY") or get_secret("HUGGINGFACE_API_KEY")

# Optional: sanity check in logs (don’t print actual values)
if not HF_API_KEY:
    print("⚠️ Warning: No Hugging Face API key found.")
if not LA_DATA_API_KEY:
    print("⚠️ Warning: No LA Open Data key found.")