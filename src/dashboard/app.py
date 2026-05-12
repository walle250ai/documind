#!/usr/bin/env python3

import os
import time
import json
import requests
import pandas as pd
import streamlit as st
from pathlib import Path
from typing import Dict, Any, List

# API configuration
@st.cache_resource
def get_api_url() -> str:
    return os.environ.get("API_URL", "http://localhost:8000")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar
with st.sidebar:
    st.title("📄 DocuMind")
    
    st.divider()
    st.subheader("Ingestion")
    
    uploaded_file = st.file_uploader("Upload Document", type=["pdf", "md", "txt"])
    collection_name = st.text_input("Collection Name", value="my_docs")
    chunking_strategy = st.selectbox(
        "Chunking Strategy",
        ["fixed", "semantic", "hierarchical"],
        index=0
    )
    if st.button("Ingest Document"):
        if uploaded_file and collection_name:
            with st.spinner("Ingesting document..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                data = {
                    "collection_name": collection_name,
                    "strategy": chunking_strategy
                }
                api_url = get_api_url()
                response = requests.post(f"{api_url}/ingest", files=files, data=data)
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Success! Ingested {result['total']} chunks", icon="✅")
                else:
                    st.error(f"Error: {response.text}")
        else:
            st.error("Please upload a file and enter a collection name")
    
    st.divider()
    st.subheader("Retrieval")
    
    retrieval_strategy = st.selectbox(
        "Retrieval Strategy",
        ["naive", "hybrid", "hyde", "hybrid_rerank"],
        index=0,
        format_func=lambda x: {
            "naive": "Naive",
            "hybrid": "Hybrid",
            "hyde": "HyDE",
            "hybrid_rerank": "Hybrid+Rerank"
        }[x]
    )
    top_k = st.slider("Top-K Results", 1, 10, 5)

# Main area tabs
tab1, tab2, tab3 = st.tabs(["Chat", "Strategy Comparison", "Cost Dashboard"])

with tab1:
    st.header("💬 Chat")
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(message["content"])
                if "sources" in message:
                    with st.expander("Sources"):
                        for i, source in enumerate(message["sources"]):
                            st.markdown(f"**Source {i+1} (Score: {source['score']:.3f})**")
                            st.markdown(source["text"])
                if "badges" in message:
                    st.write(message["badges"])
    
    # Question input
    if question := st.chat_input("Ask a question..."):
        # Display user question
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.chat_history.append({"role": "user", "content": question})
        
        # Query API
        api_url = get_api_url()
        with st.spinner("Thinking..."):
            payload = {
                "question": question,
                "collection_name": collection_name,
                "retrieval_strategy": retrieval_strategy,
                "top_k": top_k
            }
            response = requests.post(f"{api_url}/query", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                
                # Display answer
                with st.chat_message("assistant"):
                    st.markdown(result["answer"])
                    
                    # Display sources
                    with st.expander("Sources"):
                        for i, source in enumerate(result["retrieved_chunks"]):
                            st.markdown(f"**Source {i+1} (Score: {source['score']:.3f})**")
                            st.markdown(source["text"])
                    
                    # Display badges
                    latency = result["latency_ms"] / 1000
                    cost = result["estimated_cost_usd"]
                    badges = f"⚡ {latency:.1f}s | 💰 ${cost:.4f}"
                    st.write(badges)
                
                # Save to chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["retrieved_chunks"],
                    "badges": badges
                })
            else:
                st.error(f"Error: {response.text}")

with tab2:
    st.header("⚖️ Strategy Comparison")
    
    comp_question = st.text_input("Enter question to compare strategies:")
    comp_collection_name = st.text_input("Collection Name for Comparison", value="my_docs")
    comp_top_k = st.slider("Top-K for Comparison", 1, 10, 5, key="comp_top_k")
    
    if st.button("Run Comparison") and comp_question:
        strategies = ["naive", "hybrid", "hyde", "hybrid_rerank"]
        strategy_display = {
            "naive": "Naive",
            "hybrid": "Hybrid",
            "hyde": "HyDE",
            "hybrid_rerank": "Hybrid+Rerank"
        }
        results = {}
        
        with st.spinner("Running all strategies..."):
            api_url = get_api_url()
            for strategy in strategies:
                payload = {
                    "question": comp_question,
                    "collection_name": comp_collection_name,
                    "retrieval_strategy": strategy,
                    "top_k": comp_top_k
                }
                response = requests.post(f"{api_url}/query", json=payload)
                if response.status_code == 200:
                    results[strategy] = response.json()
        
        # Display side-by-side answers
        cols = st.columns(4)
        scores_data = []
        
        for idx, strategy in enumerate(strategies):
            with cols[idx]:
                st.subheader(strategy_display[strategy])
                if strategy in results:
                    result = results[strategy]
                    st.markdown(result["answer"])
                    
                    # Display sources
                    with st.expander(f"Sources ({len(result['retrieved_chunks'])})"):
                        for i, source in enumerate(result["retrieved_chunks"]):
                            st.markdown(f"**Source {i+1} (Score: {source['score']:.3f})**")
                            st.markdown(source["text"])
                    
                    # Collect score data
                    avg_score = sum(s["score"] for s in result["retrieved_chunks"]) / len(result["retrieved_chunks"]) if result["retrieved_chunks"] else 0
                    scores_data.append({
                        "Strategy": strategy_display[strategy],
                        "Average Score": avg_score
                    })
                else:
                    st.error("Failed")
        
        # Display bar chart
        if scores_data:
            st.subheader("Average Retrieval Scores")
            df = pd.DataFrame(scores_data)
            st.bar_chart(df.set_index("Strategy"))

with tab3:
    st.header("💰 Cost Dashboard")
    
    # Add slider for since_days
    since_days = st.slider("Show data for last (days)", 1, 365, 30)
    
    api_url = get_api_url()
    with st.spinner("Loading cost summary..."):
        response = requests.get(f"{api_url}/cost-summary", params={"since_days": since_days})
        if response.status_code == 200:
            summary = response.json()
            
            # Display metric cards
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Total Queries", summary["total_queries"])
            with col2:
                st.metric("Total Cost", f"${summary['total_cost_usd']:.4f}")
            with col3:
                st.metric("Average Cost per Query", f"${summary['avg_cost_per_query']:.4f}")
            with col4:
                st.metric("Most Expensive Strategy", summary["most_expensive_strategy"])
            with col5:
                st.metric("Cheapest Strategy", summary["cheapest_strategy"])
            
            # Pie chart for cost by strategy
            if summary["cost_by_strategy"]:
                st.subheader("Cost Breakdown by Strategy")
                pie_data = pd.DataFrame(
                    list(summary["cost_by_strategy"].items()),
                    columns=["Strategy", "Cost"]
                )
                st.pie_chart(pie_data.set_index("Strategy"))
            
            # Line chart for daily costs and queries
            if summary["daily_costs"]:
                st.subheader("Daily Costs and Queries")
                daily_df = pd.DataFrame(summary["daily_costs"])
                daily_df["date"] = pd.to_datetime(daily_df["date"])
                daily_df = daily_df.sort_values("date")
                
                # Calculate cumulative cost
                daily_df["cumulative_cost"] = daily_df["cost"].cumsum()
                
                # Display line chart for cumulative cost
                st.line_chart(daily_df.set_index("date")["cumulative_cost"])
        else:
            st.error(f"Error: {response.text}")
