# PlanLA API Configuration
# Loads API keys from .env file for security

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# LA Open Data Portal API Key
LA_DATA_API_KEY = os.getenv("LA_DATA_API_KEY", "")
LA_DATA_API_SECRET = os.getenv("LA_DATA_API_SECRET", "")

# LA GeoHub API Key
LA_GEOHUB_API_KEY = os.getenv("LA_GEOHUB_API_KEY", "")

# Hugging Face API Key (for free LLM)
# Get one free at: https://huggingface.co/settings/tokens
HF_API_KEY = os.getenv("HF_API_KEY", os.getenv("HUGGINGFACE_API_KEY", ""))
