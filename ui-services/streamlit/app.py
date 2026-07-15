"""Ops/internal-facing UI stub. Calls the api-services HTTP API — doesn't import
ml/ directly, so the UI and the model can be deployed and scaled independently.
"""
import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv("config/.env")
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.title("CV Project — Ops Console")

if st.button("Check API health"):
    resp = requests.get(f"{API_BASE_URL}/health", timeout=5)
    st.json(resp.json())
