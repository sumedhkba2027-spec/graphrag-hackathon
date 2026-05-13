import sys
import os

sys.path.append(os.path.abspath("."))
import streamlit as st

from pipeline1_llm_only.pipeline1 import run_pipeline1
from pipeline2_basic_rag.pipeline2 import run_pipeline2

st.set_page_config(
    page_title="GraphRAG Benchmark Dashboard",
    layout="wide"
)

st.title("🐯 GraphRAG Benchmark Dashboard")

st.write(
    "Compare LLM-Only vs Basic RAG vs GraphRAG"
)

# User input
question = st.text_input(
    "Enter your question:",
    "What caused World War II?"
)

if st.button("Run Benchmark"):

    with st.spinner("Running pipelines..."):

        # Run pipelines
        result1 = run_pipeline1(question)

        result2 = run_pipeline2(question)

    # Create columns
    col1, col2 = st.columns(2)

    # Pipeline 1
    with col1:

        st.header("Pipeline 1 — LLM Only")

        st.write(result1["answer"])

        st.subheader("Metrics")

        st.metric(
            "Tokens Used",
            result1["tokens_used"]
        )

        st.metric(
            "Latency (s)",
            result1["latency_seconds"]
        )

        st.metric(
            "Cost ($)",
            result1["cost_usd"]
        )

    # Pipeline 2
    with col2:

        st.header("Pipeline 2 — Basic RAG")

        st.write(result2["answer"])

        st.subheader("Metrics")

        st.metric(
            "Retrieved Chunks",
            result2["retrieved_chunks"]
        )

        st.metric(
            "Tokens Used",
            result2["tokens_used"]
        )

        st.metric(
            "Latency (s)",
            result2["latency_seconds"]
        )

        st.metric(
            "Cost ($)",
            result2["cost_usd"]
        )