import os
import streamlit as st
from dotenv import load_dotenv

if st.secrets:
    os.environ["LANGSMITH_TRACING"] = st.secrets["LANGSMITH_TRACING"]
    os.environ["LANGSMITH_ENDPOINT"] = st.secrets["LANGSMITH_ENDPOINT"]
    os.environ["LANGSMITH_API_KEY"] = st.secrets["LANGSMITH_API_KEY"]
    os.environ["LANGSMITH_PROJECT"] = st.secrets["LANGSMITH_PROJECT"]
else:
    load_dotenv()
from functions import app

# Page configuration
st.set_page_config(page_title="AI Research Agent", layout="wide")

# Custom CSS for styling
st.markdown(
    """
<style>
    .stApp {
        background-color: #f0f2f6;
    }
    .stButton>button {
        color: #fff;
        background-color: #4B4BFF;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #3A3AD5;
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
    }
    .report-container {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 25px;
        border: 1px solid #e1e4e8;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #2c3e50;
    }
</style>
""",
    unsafe_allow_html=True,
)

# --- UI Layout ---

st.title("📄 AI Research Agent")
st.write(
    "Enter a research topic, and the AI agent will generate a comprehensive report by planning, searching the web, and synthesizing the information."
)

# API Key inputs in the sidebar


# Use session state to store keys

# Main content area
st.header("📝 Start Your Research")

# User input for the research query
user_prompt = st.text_input(
    "Enter your research query:",
    placeholder="e.g., Market analysis of the main competitors for Slack",
)

# Button to start the research
if st.button("Generate Report"):
    # Validate inputs

    if not user_prompt:
        st.warning("Please enter a research query.")
    else:
        # Run the research process
        with st.spinner(
            "🤖 The AI agent is at work... Planning, searching, and writing. This may take a minute or two."
        ):
            try:
                final_report = app.invoke(
                    {
                        "query": "Give me a market analysis of the main competitors for Slack"
                    }
                )
                st.session_state.report = final_report["synthesized_answer"]
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                st.session_state.report = None

# Display the generated report if it exists in the session state
if "report" in st.session_state and st.session_state.report:
    st.header("📊 Your Generated Report")
    with st.container():
        st.markdown('<div class="report-container">', unsafe_allow_html=True)
        st.markdown(st.session_state.report, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

