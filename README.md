# Market Researcher

## What the Agent Does

The Market Researcher is an AI-powered agent designed to automate the process of gathering and summarizing competitor and market information. Users can input a company, product, or market topic, and the agent will generate a comprehensive analysis report. This tool is built for the AI Product Developer Challenge.

## Key Features and Limitations

### Key Features
* **Competitor Data Gathering**: The agent fetches relevant information about competitors from online sources using Tavily Search.
* **Embeddings & Semantic Search**: It uses Cohere embeddings for representing and comparing text data.
* **Large Language Model**: The agent leverages `meta-llama/llama-4-maverick-17b-128e-instruct` as its base Large Language Model for summarization and inference.
* **Automated Summarization**: It condenses large amounts of data into concise, readable insights.
* **User Interface**: The agent provides a simple and interactive user interface built with Streamlit.
* **Exportable Reports**: The generated summaries and analyses can be saved for strategic planning or presentations.

### Limitations
* The agent's knowledge is dependent on publicly available data, which means some competitor information might be missing or not up-to-date.
* The performance and cost are heavily dependent on the chosen LLM and embedding service.
* The agent does not yet feature advanced data visualizations.
* There can be some latency due to the multi-step process of embeddings, search, and summarization.

## Future Improvements
* **Advanced Data Visualizations**: Integrate libraries like Matplotlib, Seaborn, or Plotly to create charts and graphs for a more intuitive understanding of market data.
* **Wider Range of Data Sources**: Expand data gathering to include sources like social media, news articles, and financial reports for a more comprehensive analysis.
* **Interactive Reports**: Allow users to interact with the generated reports by clicking on data points to get more detailed information.
* **Caching**: Implement a caching mechanism to store the results of previous searches and reduce latency for recurring queries.
* **User Feedback**: Add a feature for users to provide feedback on the generated reports to help fine-tune the AI model and improve the quality of the generated insights.

## Tools and APIs Used

* **Programming Language**: Python
* **AI Models**: `meta-llama/llama-4-maverick-17b-128e-instruct`
* **Multi-Agent Frameworks**: LangChain, LangGraph
* **UI**: Streamlit
* **Embeddings**: Cohere Embeddings API
* **Search**: Tavily Search

## Setup Instructions

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
