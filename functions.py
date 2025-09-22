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
from langchain_groq import ChatGroq
import os

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
llm = ChatGroq(
    # model="llama-3.1-8b-instant"
    model="meta-llama/llama-4-maverick-17b-128e-instruct"
)
from typing import TypedDict, List, Optional, Dict, Annotated
from langchain.schema import Document
from operator import add


def merge_dicts_overwrite_reducer(
    a: Dict[str, str], b: Dict[str, str]
) -> Dict[str, str]:
    merged = a.copy()
    merged.update(b)
    return merged


class AgentState(TypedDict):
    query: str  # The user's research question
    subqueries: list[str]
    sub_summaries: Annotated[Dict[str, str], merge_dicts_overwrite_reducer]
    synthesized_answer: str  # Final summarized answer


from typing import Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field


class GenerateNodeOutput(BaseModel):
    subqueries: List[str] = Field(
        description="List of sub-queries of the orginal query"
    )


model = llm.with_structured_output(GenerateNodeOutput)
# Prompt to generate sub-queries
# subquery_prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "Break the following research question into 4 focused sub-questions to retrieve accurate information.",
#         ),
#         ("user", "{query}"),
#     ]
# )
from langchain.prompts import ChatPromptTemplate

subquery_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert in competitor analysis and market research. 
Your task is to break down the user's topic into 3 to 5 detailed, answerable sub-questions focused on understanding the competitive landscape.

Based on the topic: "{query}", generate key questions that will help analyze:
1. The main competitors in this space,
2. Their strengths and weaknesses,
3. Market positioning and strategies,
4. Customer perception and differentiators.

Ensure the questions are specific and actionable, leading to insights that will support a comprehensive competitor analysis report.
""",
        ),
        ("user", "{query}"),
    ]
)

subquery_chain: Runnable = subquery_prompt | model


# --- Planner Node ---
def planner_node(state: AgentState) -> AgentState:
    query = state["query"]
    # print(f"[Planner] Received query: {query}")
    if not query:
        return {"error": "Empty query received."}

    # print(f"[Planner] Received query: {query}")

    # Generate sub-queries using the LLM
    subqueries = subquery_chain.invoke({"query": query})
    print(f"[Planner] Generated sub-queries: {subqueries}")

    state["subqueries"] = subqueries.subqueries
    return state


def send_sub_query(state: AgentState) -> AgentState:
    return [Send("Sub_graph", {"m_query": s}) for s in state["subqueries"]]


from typing import TypedDict, List, Optional, Dict, Annotated
from langchain.schema import Document
from operator import add


class SubState(TypedDict):
    m_query: str  # The user's research question
    urls: Annotated[list[str], add]
    raw_html: Annotated[list[str], add]  # Raw HTML of fetched web pages
    parsed_content: Annotated[list[str], add]
    retrieved_docs: Dict[str, List[Document]]
    sub_summaries: Annotated[Dict[str, str], merge_dicts_overwrite_reducer]


from langchain_tavily import TavilySearch
from langgraph.constants import Send

os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]
tavily_search = TavilySearch(max_results=3)

# def web_search_tool(state:AgentState)->AgentState:
#   subqueries = state["subqueries"]
#   print(f"[WebSearchTool] Running {len(subqueries)} Tavily searches...")

#   all_url=[]
#   all_results = []
#   for q in subqueries:
#       result_chunks = tavily_search(q)
#       print(f" - Results for '{q}': {result_chunks['results']}")
#       for result in result_chunks["results"]:
#         all_results.append(result["content"])
#         all_url.append(result["url"])
#   state["urls"] = all_url
#   state["raw_html"] = all_results
#   return state


def web_search_tool(state: SubState):
    all_url = []
    all_results = []
    query = state["m_query"]
    result_chunks = tavily_search(query)
    # print(f" - Results for '{query}': {result_chunks['results']}")
    for result in result_chunks["results"]:
        all_results.append(result["content"])
        all_url.append(result["url"])

    return {"urls": all_url, "raw_html": all_results}


import trafilatura
import requests
from bs4 import BeautifulSoup


# def Document_Retriever(state:AgentState)->AgentState:
#   urls = state["urls"]
#   print(f"[DocumentRetriever] Running {len(urls)} URL fetches...")

#   full_text = []
#   for url in urls:
#     try:
#       html = requests.get(url, timeout=10).text
#       soup = BeautifulSoup(html, 'html.parser')

#             # Attempt to find the main content (adjust for specific sites if needed)
#       content_div = soup.find('div', {'class': 'mw-parser-output'}) or soup.body
#       if content_div:
#           paragraphs = content_div.find_all('p')
#           page_text = "\n".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
#           full_text.append(page_text)

#       else:
#           print(f"[Warning] Could not find content for URL: {url}")
#     except Exception as e:
#       print(f"[Error] Failed to retrieve or parse {url}: {e}")


#   state["parsed_content"] = full_text
#   return state
import requests
from bs4 import BeautifulSoup

def Web_Retriver(url: str | dict):
    """
    Fetches and parses the main textual content from a webpage.
    If a URL is unresponsive or fails, it is skipped and returns an empty parsed_content list.
    """

    # --- Normalize input ---
    if isinstance(url, dict):
        url = url.get("url")

    full_text = []

    if not url or not isinstance(url, str):
        # Invalid URL, skip gracefully
        return {"parsed_content": full_text}

    try:
        # Try to fetch the page with a timeout
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raises for HTTP errors like 404, 500

        # Parse the HTML content
        soup = BeautifulSoup(response.text, "html.parser")

        # Try to locate the main content
        content_div = soup.find("div", {"class": "mw-parser-output"}) or soup.body
        if content_div:
            paragraphs = content_div.find_all("p")
            page_text = "\n".join(
                [p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)]
            )
            if page_text.strip():
                full_text.append(page_text)

        return {"parsed_content": full_text}

    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
        # Skip unresponsive URLs silently
        return {"parsed_content": full_text}

    except requests.exceptions.HTTPError:
        # Skip URLs with bad HTTP status codes
        return {"parsed_content": full_text}

    except Exception:
        # Catch-all for unexpected issues, skip silently
        return {"parsed_content": full_text}


def parallel_web_search(state: SubState) -> SubState:
    return [Send("Web_Retriver", {"url": s}) for s in state["urls"]]


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_cohere import CohereEmbeddings


def Retriver(state: dict) -> dict:
    chunks = state["parsed_content"]  # should be a list of text chunks

    # Create embedding
    embedding = CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=st.secrets["COHERE_API_KEY"],
        user_agent="langchain",
    )

    # Combine and split content
    full_text = "\n".join(chunk for chunk in chunks)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    splits = splitter.create_documents([full_text])

    # Create retrievers
    vectordb = FAISS.from_documents(splits, embedding=embedding)
    faiss_retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 3

    ensemble_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever], weights=[0.5, 0.5]
    )

    # Map subqueries to retrieved documents
    query_doc_map = {}
    query = state["m_query"]
    retrieved_docs = ensemble_retriever.invoke(query)
    query_doc_map[query] = retrieved_docs
    # print(f"[Summarizer] Retrieved {len(retrieved_docs)} docs for query: '{query}'")

    # Save to state
    state["retrieved_docs"] = query_doc_map
    return state


from typing import Dict, List
from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from langchain_core.messages import AIMessage


# Use OpenAI or any compatible chat model


def summarize_documents(docs: List[Document], sub_question: str) -> str:
    """Summarizes a list of documents for a given sub-question."""
    joined_content = "\n\n".join(doc.page_content for doc in docs[:5])  # top-k
    prompt = f"""
    You are an expert market research analyst. Your task is to provide a concise and factual summary of the provided text for a specific research question.
    Focus on extracting the key data points, statistics, and insights relevant to the question."
    
    Research Question: {sub_question}"

    Content from Web Search:
    ---
    {joined_content}
    ---
    
    Provide a detailed summary based *only* on the provided content.

"""
    response = llm.invoke(prompt)
    return response.content.strip()


def multi_hop_summarizer(state: SubState) -> SubState:
    retrieved_docs: Dict[str, List[Document]] = state["retrieved_docs"]

    sub_summaries = {}
    sub_q = state["m_query"]
    docs = retrieved_docs.get(sub_q, [])
    if docs:
        summary = summarize_documents(docs, sub_q)
        sub_summaries[sub_q] = summary

    # final_answer = combine_sub_summaries(sub_summaries, original_query)
    # state["synthesized_answer"] = final_answer
    state["sub_summaries"] = sub_summaries
    return state


def report_generator_node(state: dict) -> dict:
    summaries = state["sub_summaries"]
    query = state["query"]

    section_text = "\n\n".join(
        f"## {sub_q}\n{answer}" for sub_q, answer in summaries.items()
    )

    prompt = f"""
    You are a senior market research analyst tasked with compiling a final, comprehensive report.
    You have been provided with a research topic and a series of summaries for different sub-questions.
    Your job is to synthesize this information into a single, well-structured, and insightful market research report in Markdown format.

    Main Research Topic: {query}

    Summaries for sub-questions:
    ---
    {section_text}
    ---

    Please generate the final report with the following structure:
    - A clear and concise title.
    - A brief introduction/executive summary.
    - A section for each sub-question, using the provided summaries to build a coherent narrative.
    - A concluding paragraph that synthesizes the key findings and provides a forward-looking statement.
    - **Do not** make up any information. Base the entire report strictly on the summaries provided.
    
"""

    report_md = llm.invoke(prompt).content
    return {
        **state,
        "synthesized_answer": report_md,
    }


from langgraph.graph import StateGraph, END, START

graph = StateGraph(SubState)
graph.add_node("WebSearchTool", web_search_tool)
graph.add_node("Web_Retriver", Web_Retriver)
graph.add_node("Retriever", Retriver)
graph.add_node("summarizer", multi_hop_summarizer)
# graph.add_node("validator",validator_node)

# graph.add_edge("WebSearchTool", "Web_Retriver")
graph.add_edge("Web_Retriver", "Retriever")
graph.add_edge("Retriever", "summarizer")
graph.add_edge("summarizer", END)

graph.add_edge(START, "WebSearchTool")
graph.add_conditional_edges(
    "WebSearchTool", parallel_web_search, {"Web_Retriver": "Web_Retriver"}
)
# graph.add_conditional_edges(
#     "validator",
#     should_continue,
#     {
#         "rerun": "Planner",
#         "done": "generator"
#     }
# )

sub_graph = graph.compile()


graph = StateGraph(AgentState)

graph.add_node("Planner", planner_node)
graph.add_node("Sub_graph", sub_graph)
graph.add_node("generator", report_generator_node)

graph.add_conditional_edges("Planner", send_sub_query, {"Sub_graph": "Sub_graph"})
graph.add_edge("Sub_graph", "generator")
graph.add_edge(START, "Planner")
graph.add_edge("generator", END)

app = graph.compile()



