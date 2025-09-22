# Market Researcher

## 1. Overview
The **Market Researcher Agent** is an AI-powered tool designed to **gather and summarize competitor information**. It automates collecting data about competitors, analyzing their offerings, and generating actionable insights. This project was built for the **AI Product Developer Challenge**.

---

## 2. Key Features
- **Competitor Data Gathering**: Fetches relevant information about competitors from online sources using Tavily Search.
- **Embeddings & Semantic Search**: Uses Cohere embeddings to represent and compare text data.
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
- **Embeddings**: Cohere Embeddings API
- **Search**: Tavily Search (for retrieving relevant documents/web pages)
- **Frameworks & Libraries**:
  - Streamlit (UI)
  - LangChain & LangGraph (Orchestration)
  - beautifulsoup4 (HTML parsing)
  - FAISS (Vector Store)

---

## 4. Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/raghavmour/market-researcher.git](https://github.com/raghavmour/market-researcher.git)
    cd market-researcher
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up environment variables:**
    Create a `.env` file in the root directory and add the following keys:
    ```
    LANGSMITH_TRACING="true"
    LANGSMITH_ENDPOINT="[https://api.smith.langchain.com](https://api.smith.langchain.com)"
    LANGSMITH_API_KEY="..."
    LANGSMITH_PROJECT="..."
    GROQ_API_KEY="..."
    TAVILY_API_KEY="..."
    COHERE_API_KEY="..."
    ```

4.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

---

## 5. Architecture Diagram

```mermaid
graph TD
    A[User Input: Research Query] --> B{Main Graph};

    subgraph B [Main Graph]
        C[Planner Node] --> D{For each sub-query};
        D --> E[Sub_graph];
        E --> F[Collect Summaries];
        F --> G[Report Generator Node];
    end

    subgraph E [Sub-Graph Execution]
        direction LR
        H[WebSearchTool] --> I[Web_Retriver];
        I --> J[Retriever];
        J --> K[Summarizer Node];
    end

    G --> L[Final Report];
    L --> M[Display to User];

    style B fill:#f9f9f9,stroke:#333,stroke-width:2px
    style E fill:#e6f7ff,stroke:#333,stroke-width:2px
