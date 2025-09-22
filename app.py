import os
import streamlit as st
from dotenv import load_dotenv

# --- Environment Setup ---
if st.secrets:
    os.environ["LANGSMITH_TRACING"] = st.secrets.get("LANGSMITH_TRACING", "")
    os.environ["LANGSMITH_ENDPOINT"] = st.secrets.get("LANGSMITH_ENDPOINT", "")
    os.environ["LANGSMITH_API_KEY"] = st.secrets.get("LANGSMITH_API_KEY", "")
    os.environ["LANGSMITH_PROJECT"] = st.secrets.get("LANGSMITH_PROJECT", "")
else:
    load_dotenv()

from functions import app

# --- Page Configuration ---
st.set_page_config(
    page_title="Market Researcher AI",
    page_icon="📊",
    layout="wide"
)

# --- Custom CSS for styling ---
st.markdown(
    """
    <style>
        .stApp {
            background-color: #f8f9fc;
        }
        .stButton>button {
            color: #fff;
            background-color: #2E86DE;
            border-radius: 8px;
            padding: 10px 20px;
            border: none;
            font-weight: bold;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #1B4F72;
        }
        .stTextInput>div>div>input {
            border-radius: 8px;
            border: 1px solid #ccc;
            padding: 8px;
        }
        .report-container {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 25px;
            border: 1px solid #e1e4e8;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #2c3e50;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- UI Header ---
st.title("📊 AI Market Researcher")
st.write(
    "Get a **comprehensive competitor and market analysis** powered by AI. "
    "Simply enter a company, product, or market topic below, and the AI will gather and summarize relevant insights."
)

# --- Research Input Section ---
st.header("📝 Start Your Market Research")

user_prompt = st.text_input(
    "Enter a competitor, company, or market topic:",
    placeholder="e.g., Market analysis of Slack competitors",
)

# --- Generate Report Button ---
if st.button("Generate Report"):
    if not user_prompt:
        st.warning("⚠️ Please enter a research topic before generating the report.")
    else:
        with st.spinner("🤖 Gathering insights and preparing your report..."):
            try:
                final_report = app.invoke({"query": user_prompt})
                st.session_state.report = final_report.get("synthesized_answer", "")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                st.session_state.report = None

# --- Display Final Report ---
if "report" in st.session_state and st.session_state.report:
    st.header("📈 Your Market Research Report")
    with st.container():
        st.markdown('<div class="report-container">', unsafe_allow_html=True)
        st.markdown(st.session_state.report, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
