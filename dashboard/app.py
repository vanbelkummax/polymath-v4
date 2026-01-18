#!/usr/bin/env python3
"""
Polymath v4 - Streamlit Dashboard

Visual interface for:
- Unified search (papers + repos)
- Paper discovery (CORE API)
- System status & ingestion monitor
- Literature review generator

Run: streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import psycopg2
import pandas as pd
from datetime import datetime

from lib.config import config

# Page config
st.set_page_config(
    page_title="Polymath v4",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# Database Connection
# ============================================================================

@st.cache_resource
def get_connection():
    return psycopg2.connect(config.POSTGRES_DSN)


def run_query(query: str, params=None):
    """Run a SQL query and return DataFrame."""
    conn = get_connection()
    return pd.read_sql_query(query, conn, params=params)


# ============================================================================
# Sidebar Navigation
# ============================================================================

st.sidebar.title("📚 Polymath v4")
page = st.sidebar.radio(
    "Navigation",
    ["🔍 Search", "📊 Dashboard", "📥 Discovery", "📝 Literature Review"],
    index=0
)


# ============================================================================
# Search Page
# ============================================================================

def search_page():
    st.title("🔍 Unified Search")
    st.markdown("Search across papers and repositories")

    # Search input
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input("Search query", placeholder="spatial transcriptomics cell segmentation")
    with col2:
        search_type = st.selectbox("Type", ["All", "Papers", "Repos"])

    # Options
    col1, col2, col3 = st.columns(3)
    with col1:
        use_graphrag = st.checkbox("GraphRAG expansion", value=False)
    with col2:
        use_rerank = st.checkbox("Rerank results", value=True)
    with col3:
        top_k = st.slider("Results", 5, 50, 20)

    if query:
        with st.spinner("Searching..."):
            try:
                from lib.search.hybrid_search import HybridSearcher
                searcher = HybridSearcher(rerank=use_rerank)
                results = searcher.hybrid_search(
                    query, n=top_k, graph_expand=use_graphrag
                )

                if results:
                    st.success(f"Found {len(results)} results")

                    for r in results:
                        with st.expander(f"📄 {r.title[:80]}... (score: {r.score:.4f})"):
                            st.markdown(f"**Passage:**")
                            st.text(r.passage_text[:1000])
                            st.markdown(f"*Doc ID: {r.doc_id}*")
                else:
                    st.warning("No results found")

            except Exception as e:
                st.error(f"Search error: {e}")

    # Recent searches hint
    with st.expander("Search tips"):
        st.markdown("""
        - Use quotes for exact phrases: `"cell segmentation"`
        - GraphRAG expands queries with related concepts
        - Reranking improves relevance but is slower
        """)


# ============================================================================
# Dashboard Page
# ============================================================================

def dashboard_page():
    st.title("📊 System Dashboard")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        df = run_query("SELECT COUNT(*) as count FROM documents")
        st.metric("📄 Documents", f"{df['count'][0]:,}")

    with col2:
        df = run_query("SELECT COUNT(*) as count FROM passages WHERE embedding IS NOT NULL")
        st.metric("📝 Passages", f"{df['count'][0]:,}")

    with col3:
        df = run_query("SELECT COUNT(*) as count FROM repositories")
        st.metric("💻 Repositories", f"{df['count'][0]:,}")

    with col4:
        df = run_query("SELECT COUNT(*) as count FROM passage_concepts")
        st.metric("🧠 Concepts", f"{df['count'][0]:,}")

    st.divider()

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📅 Documents by Year")
        df = run_query("""
            SELECT year, COUNT(*) as count
            FROM documents
            WHERE year IS NOT NULL AND year > 2015
            GROUP BY year ORDER BY year
        """)
        if not df.empty:
            st.bar_chart(df.set_index('year'))

    with col2:
        st.subheader("🏷️ Top Concepts")
        df = run_query("""
            SELECT concept_name, COUNT(*) as count
            FROM passage_concepts
            WHERE concept_type IN ('domain', 'method')
            AND confidence > 0.6
            GROUP BY concept_name
            ORDER BY count DESC
            LIMIT 15
        """)
        if not df.empty:
            st.dataframe(df, use_container_width=True)

    st.divider()

    # Repository stats
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💻 Top Repositories")
        df = run_query("""
            SELECT owner || '/' || name as repo, stars, language
            FROM repositories
            WHERE stars IS NOT NULL
            ORDER BY stars DESC
            LIMIT 10
        """)
        if not df.empty:
            st.dataframe(df, use_container_width=True)

    with col2:
        st.subheader("🔧 Software Mentions")
        df = run_query("""
            SELECT canonical_name, COUNT(*) as mentions
            FROM software_mentions
            GROUP BY canonical_name
            ORDER BY mentions DESC
            LIMIT 10
        """)
        if not df.empty:
            st.dataframe(df, use_container_width=True)

    # Recent ingestion
    st.subheader("📥 Recent Ingestions")
    df = run_query("""
        SELECT title, source_method, created_at
        FROM documents
        ORDER BY created_at DESC
        LIMIT 10
    """)
    if not df.empty:
        st.dataframe(df, use_container_width=True)


# ============================================================================
# Discovery Page
# ============================================================================

def discovery_page():
    st.title("📥 Paper Discovery")
    st.markdown("Discover new papers from CORE API (130M+ open access papers)")

    # Search
    query = st.text_input("Search CORE API", placeholder="spatial transcriptomics methods")

    col1, col2, col3 = st.columns(3)
    with col1:
        year_min = st.number_input("Year from", min_value=2000, max_value=2026, value=2022)
    with col2:
        limit = st.slider("Max results", 10, 100, 30)
    with col3:
        auto_ingest = st.checkbox("Auto-ingest", value=False)

    if st.button("🔍 Search CORE") and query:
        with st.spinner("Searching CORE API..."):
            try:
                from scripts.discover_papers import discover_papers

                results = discover_papers(
                    query=query,
                    max_results=limit,
                    year_min=year_min,
                    auto_ingest=auto_ingest,
                    compute_embeddings=auto_ingest,
                    dry_run=not auto_ingest
                )

                st.success(f"Found {results['total_found']} papers")

                col1, col2, col3 = st.columns(3)
                col1.metric("Total Found", results['total_found'])
                col2.metric("New Papers", results['new_papers'])
                col3.metric("Duplicates", results['duplicates'])

                if results.get('papers'):
                    st.subheader("Results")
                    for p in results['papers'][:20]:
                        with st.expander(f"📄 {p.get('title', 'Unknown')[:60]}..."):
                            st.write(f"**Year:** {p.get('year')}")
                            st.write(f"**DOI:** {p.get('doi') or 'N/A'}")
                            if p.get('status'):
                                st.write(f"**Status:** {p['status']}")

                if results.get('manual_retrieval_needed'):
                    st.warning(f"{len(results['manual_retrieval_needed'])} papers need manual retrieval")

            except Exception as e:
                st.error(f"Error: {e}")

    # Gap analysis
    st.divider()
    st.subheader("📊 Corpus Gap Analysis")

    if st.button("Analyze Gaps"):
        with st.spinner("Analyzing..."):
            try:
                from scripts.active_librarian import analyze_corpus_gaps
                conn = get_connection()
                gaps = analyze_corpus_gaps(conn, min_mentions=3)

                col1, col2 = st.columns(2)
                col1.metric("Corpus Size", f"{gaps['corpus_size']:,}")
                col2.metric("Missing DOIs", gaps['total_missing'])

                st.write("**Top Research Areas:**")
                st.write(", ".join(gaps['top_concepts'][:10]))

                if gaps['missing_dois']:
                    st.write("**Top Missing DOIs:**")
                    df = pd.DataFrame([
                        {'DOI': doi, 'Mentions': count}
                        for doi, count in list(gaps['missing_dois'].items())[:10]
                    ])
                    st.dataframe(df, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")


# ============================================================================
# Literature Review Page
# ============================================================================

def literature_review_page():
    st.title("📝 Literature Review Generator")
    st.markdown("Generate literature reviews from your corpus")

    # Query input
    query = st.text_input("Topic/Query", placeholder="deep learning for spatial transcriptomics")

    col1, col2, col3 = st.columns(3)
    with col1:
        top_k = st.slider("Number of papers", 3, 20, 8)
    with col2:
        llm = st.selectbox("LLM", ["gemini", "anthropic"])
    with col3:
        mode = st.selectbox("Mode", ["Literature Review", "Comparison"])

    if st.button("📝 Generate") and query:
        with st.spinner("Finding relevant papers..."):
            try:
                conn = get_connection()

                from scripts.summarize_papers import (
                    get_passages_by_query,
                    generate_literature_review,
                    generate_comparison
                )

                papers = get_passages_by_query(conn, query, top_k=top_k)

                if not papers:
                    st.warning("No papers found")
                    return

                st.success(f"Found {len(papers)} relevant papers")

                # Show papers
                with st.expander("📚 Papers included"):
                    for p in papers:
                        st.write(f"- {p['citation']} {p['title'][:60]}...")

                # Generate
                with st.spinner(f"Generating {mode.lower()} with {llm}..."):
                    if mode == "Literature Review":
                        output = generate_literature_review(query, papers, llm)
                    else:
                        output = generate_comparison(query, papers, llm)

                    st.subheader("Generated Review")
                    st.markdown(output)

                    # Download button
                    st.download_button(
                        "📥 Download Markdown",
                        output,
                        file_name=f"review_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )

            except Exception as e:
                st.error(f"Error: {e}")


# ============================================================================
# Main Router
# ============================================================================

if page == "🔍 Search":
    search_page()
elif page == "📊 Dashboard":
    dashboard_page()
elif page == "📥 Discovery":
    discovery_page()
elif page == "📝 Literature Review":
    literature_review_page()


# Footer
st.sidebar.divider()
st.sidebar.caption("Polymath v4 - Knowledge Discovery System")
st.sidebar.caption(f"v4.0.0 | {datetime.now().strftime('%Y-%m-%d')}")
