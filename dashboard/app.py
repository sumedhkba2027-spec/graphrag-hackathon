import os
import sys
import time

import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath("."))

from pipeline1_llm_only.pipeline1 import run_pipeline1
from pipeline2_basic_rag.pipeline2 import run_pipeline2
from pipeline3_graphrag.pipeline3 import run_pipeline3


DEFAULT_QUESTION = "What did NEXON discuss about revenue and growth?"


st.set_page_config(
    page_title="GraphRAG Benchmark Dashboard",
    layout="wide",
)


def run_safely(label, runner, question):
    start = time.time()
    try:
        result = runner(question)
        result["error"] = None
        return result
    except Exception as exc:
        return {
            "pipeline": label,
            "question": question,
            "answer": "",
            "tokens_used": 0,
            "latency_seconds": round(time.time() - start, 2),
            "cost_usd": 0.0,
            "error": str(exc),
        }


def token_delta(base_tokens, graph_tokens):
    if not base_tokens or not graph_tokens:
        return None
    return round((base_tokens - graph_tokens) / base_tokens * 100, 1)


def render_result(title, subtitle, result, extra_metrics=None):
    st.header(title)
    st.caption(subtitle)

    if result.get("error"):
        st.error(result["error"])
    else:
        st.info(result.get("answer") or "No answer returned.")

    st.subheader("Metrics")
    st.metric("Tokens Used", result.get("tokens_used", 0))
    st.metric("Latency (s)", result.get("latency_seconds", 0))
    st.metric("Cost ($)", result.get("cost_usd", 0))

    for metric_label, metric_value in (extra_metrics or {}).items():
        st.metric(metric_label, metric_value)


def render_token_summary(result1, result2, result3):
    reduction_vs_p2 = token_delta(result2.get("tokens_used"), result3.get("tokens_used"))
    reduction_vs_p1 = token_delta(result1.get("tokens_used"), result3.get("tokens_used"))

    st.divider()
    st.subheader("Token Reduction Summary")

    col1, col2 = st.columns(2)
    with col1:
        if reduction_vs_p2 is None:
            st.metric("GraphRAG vs Basic RAG", "N/A")
        else:
            label = "fewer tokens" if reduction_vs_p2 >= 0 else "more tokens"
            st.metric(
                "GraphRAG vs Basic RAG",
                f"{abs(reduction_vs_p2)}% {label}",
                delta=result2.get("tokens_used", 0) - result3.get("tokens_used", 0),
            )

    with col2:
        if reduction_vs_p1 is None:
            st.metric("GraphRAG vs LLM Only", "N/A")
        else:
            label = "fewer tokens" if reduction_vs_p1 >= 0 else "more tokens"
            st.metric(
                "GraphRAG vs LLM Only",
                f"{abs(reduction_vs_p1)}% {label}",
                delta=result1.get("tokens_used", 0) - result3.get("tokens_used", 0),
            )

    if reduction_vs_p2 is not None:
        if reduction_vs_p2 >= 0:
            st.success(f"GraphRAG used {reduction_vs_p2}% fewer tokens than Basic RAG.")
        else:
            st.warning(f"GraphRAG used {abs(reduction_vs_p2)}% more tokens than Basic RAG for this query.")


st.title("GraphRAG Benchmark Dashboard")
st.write("Compare LLM-Only, Basic RAG, and TigerGraph GraphRAG on the same finance query.")

question = st.text_input(
    "Enter your question:",
    DEFAULT_QUESTION,
)

if st.button("Run Benchmark", type="primary"):
    if not question.strip():
        st.warning("Enter a question before running the benchmark.")
        st.stop()

    progress = st.progress(0, text="Running Pipeline 1: LLM Only")
    result1 = run_safely("Pipeline 1 - LLM Only", run_pipeline1, question)

    progress.progress(33, text="Running Pipeline 2: Basic RAG")
    result2 = run_safely("Pipeline 2 - Basic RAG", run_pipeline2, question)

    progress.progress(66, text="Running Pipeline 3: GraphRAG")
    result3 = run_safely("Pipeline 3 - GraphRAG", run_pipeline3, question)
    progress.progress(100, text="Benchmark complete")

    col1, col2, col3 = st.columns(3)
    with col1:
        render_result("Pipeline 1", "LLM Only", result1)
    with col2:
        render_result(
            "Pipeline 2",
            "Basic RAG",
            result2,
            {"Chunks Retrieved": result2.get("retrieved_chunks", 0)},
        )
    with col3:
        render_result(
            "Pipeline 3",
            "TigerGraph GraphRAG",
            result3,
            {
                "Entities Found": len(result3.get("entities_found", [])),
                "Context Chunks": result3.get("context_chunks", 0),
            },
        )

    st.divider()
    st.subheader("Side-by-Side Comparison")
    comparison = pd.DataFrame(
        [
            {
                "Pipeline": "LLM Only",
                "Tokens Used": result1.get("tokens_used", 0),
                "Latency (s)": result1.get("latency_seconds", 0),
                "Cost ($)": result1.get("cost_usd", 0),
                "Status": "Error" if result1.get("error") else "OK",
            },
            {
                "Pipeline": "Basic RAG",
                "Tokens Used": result2.get("tokens_used", 0),
                "Latency (s)": result2.get("latency_seconds", 0),
                "Cost ($)": result2.get("cost_usd", 0),
                "Status": "Error" if result2.get("error") else "OK",
            },
            {
                "Pipeline": "GraphRAG",
                "Tokens Used": result3.get("tokens_used", 0),
                "Latency (s)": result3.get("latency_seconds", 0),
                "Cost ($)": result3.get("cost_usd", 0),
                "Status": "Error" if result3.get("error") else "OK",
            },
        ]
    )
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    render_token_summary(result1, result2, result3)
