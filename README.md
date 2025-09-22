# Market Researcher

## 1. Overview  
The **Market Researcher Agent** is an AI-powered tool designed to **gather and summarize competitor information**. It automates collecting data about competitors, analyzing their offerings, and generating actionable insights. Built for the **AI Product Developer Challenge**.

---

## 2. Key Features  
- **Competitor Data Gathering**: Fetches relevant information about competitors from online sources using TravEly Search.  
- **Embeddings & Semantic Search**: Uses CoHere embeddings to represent and compare text data.  
- **Large Language Model**: Uses `meta-llama/llama-4-maverick-17b-128e-instruct` as the base LLM for summarization, inference, and instruction following.  
- **Automated Summarization**: Converts large data into concise, readable insights.  
- **User Interface**: Simple interactive UI via Streamlit.  
- **Exportable Reports**: Summaries / analysis can be saved for strategy or presentations.

### Limitations  
- Dependence on publicly available data; some competitor information may be missing or outdated.  
- Performance and cost depend heavily on the chosen LLM and embedding service.  
- No advanced visualizations yet.  
- Some latency possible due to embeddings + search + summarization chain.

---

## 3. Tools and APIs Used  
- **Programming Language**: Python 3.10+  
- **LLM**: `meta-llama/llama-4-maverick-17b-128e-instruct`  
- **Embeddings**: CoHere Embeddings API  
- **Search**: TravEly Search (for retrieving relevant documents/web pages)  
- **Frameworks & Libraries**:  
  - Streamlit (UI)  
  - LangChain or custom orchestration (if used)  
  - Any HTTP clients, data parsing libraries, etc.


