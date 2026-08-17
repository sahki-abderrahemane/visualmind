"""
VisualMind Streamlit Analytics Dashboard.

Provides:
- Search metrics
- Search type analytics
- Top queries
- Zero-result searches
- Search trends
- FAISS index health
- Embedding statistics
- CLIP embedding visualization with UMAP
"""

from __future__ import annotations

import pickle
import sqlite3
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATABASE_PATH = PROJECT_ROOT / "data" / "products.db"
FAISS_INDEX_PATH = PROJECT_ROOT / "data" / "faiss_index.bin"
METADATA_PATH = PROJECT_ROOT / "data" / "embeddings_metadata.pkl"


# ============================================================
# Streamlit configuration
# ============================================================

st.set_page_config(
    page_title="VisualMind Analytics",
    page_icon="🔎",
    layout="wide",
)


# ============================================================
# Data loading
# ============================================================

@st.cache_data(ttl=30)
def load_search_events() -> pd.DataFrame:
    """Load search telemetry from SQLite."""

    connection = sqlite3.connect(DATABASE_PATH)

    try:
        events = pd.read_sql_query(
            """
            SELECT
                id,
                query_type,
                query_text,
                result_count,
                top_result_id,
                response_time_ms,
                timestamp
            FROM search_events
            ORDER BY timestamp DESC
            """,
            connection,
        )
    finally:
        connection.close()

    if not events.empty:
        events["timestamp"] = pd.to_datetime(
            events["timestamp"],
            utc=True,
            errors="coerce",
        )

    return events


@st.cache_resource
def load_faiss_index():
    """Load the persistent FAISS index once."""

    if not FAISS_INDEX_PATH.exists():
        return None

    return faiss.read_index(
        str(FAISS_INDEX_PATH)
    )


@st.cache_resource
def load_embedding_metadata():
    """Load the FAISS position → product mapping."""

    if not METADATA_PATH.exists():
        return None

    with METADATA_PATH.open("rb") as file:
        return pickle.load(file)


# ============================================================
# Load data
# ============================================================

events = load_search_events()
faiss_index = load_faiss_index()
embedding_metadata = load_embedding_metadata()


# ============================================================
# Header
# ============================================================

st.title("VisualMind Analytics")

st.caption(
    "Multimodal product search monitoring and embedding analytics"
)


# ============================================================
# Search Metrics
# ============================================================

st.header("Search Metrics")

if events.empty:

    searches_today = 0
    searches_this_week = 0
    image_searches = 0
    text_searches = 0
    multimodal_searches = 0
    average_response = 0.0

    st.warning(
        "No search events have been recorded yet."
    )

else:

    now = pd.Timestamp.now(tz="UTC")

    event_times = events["timestamp"]

    today_start = now.normalize()

    searches_today = int(
        (event_times >= today_start).sum()
    )

    week_start = (
        today_start
        - pd.Timedelta(
            days=today_start.weekday()
        )
    )

    searches_this_week = int(
        (event_times >= week_start).sum()
    )

    image_searches = int(
        (events["query_type"] == "image").sum()
    )

    text_searches = int(
        (events["query_type"] == "text").sum()
    )

    multimodal_searches = int(
        (events["query_type"] == "multimodal").sum()
    )

    average_response = float(
        events["response_time_ms"].mean()
    )


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Searches",
        f"{len(events):,}",
    )

with col2:
    st.metric(
        "Searches Today",
        f"{searches_today:,}",
    )

with col3:
    st.metric(
        "Searches This Week",
        f"{searches_this_week:,}",
    )

with col4:
    st.metric(
        "Avg Response",
        f"{average_response:.1f} ms",
    )


# ============================================================
# Search Type Metrics
# ============================================================

st.subheader("Search Types")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Image",
        f"{image_searches:,}",
    )

with col2:
    st.metric(
        "Text",
        f"{text_searches:,}",
    )

with col3:
    st.metric(
        "Multimodal",
        f"{multimodal_searches:,}",
    )


# ============================================================
# Search Analytics
# ============================================================

st.header("Search Analytics")

if not events.empty:

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Searches by Type")

        search_counts = (
            events["query_type"]
            .value_counts()
            .rename_axis("query_type")
            .reset_index(name="searches")
        )

        st.bar_chart(
            search_counts.set_index(
                "query_type"
            )
        )

    with col2:

        st.subheader("Average Response Time")

        response_by_type = (
            events.groupby("query_type")[
                "response_time_ms"
            ]
            .mean()
            .round(2)
        )

        st.bar_chart(
            response_by_type
        )


# ============================================================
# Search Trends
# ============================================================

st.subheader("Search Trends")

if not events.empty:

    daily_searches = (
        events
        .set_index("timestamp")
        .resample("D")
        .size()
        .rename("searches")
    )

    st.line_chart(
        daily_searches
    )

else:

    st.info(
        "No search trend data available."
    )


# ============================================================
# Top Queries
# ============================================================

st.header("Top Queries")

text_events = events[
    (events["query_type"] == "text")
    | (events["query_type"] == "multimodal")
].copy()

if not text_events.empty:

    text_events["query_text"] = (
        text_events["query_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    top_queries = (
        text_events[
            text_events["query_text"] != ""
        ]["query_text"]
        .value_counts()
        .head(10)
        .rename_axis("query")
        .reset_index(name="searches")
    )

    if not top_queries.empty:

        st.dataframe(
            top_queries,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "No textual queries available."
        )

else:

    st.info(
        "No textual search events available."
    )


# ============================================================
# Zero-result Searches
# ============================================================

st.subheader("Zero-Result Searches")

if not events.empty:

    zero_results = events[
        events["result_count"] == 0
    ]

    st.metric(
        "Zero-result searches",
        f"{len(zero_results):,}",
    )

    if not zero_results.empty:

        zero_result_queries = (
            zero_results[
                [
                    "timestamp",
                    "query_type",
                    "query_text",
                ]
            ]
            .head(20)
        )

        st.dataframe(
            zero_result_queries,
            use_container_width=True,
            hide_index=True,
        )

else:

    st.info(
        "No search data available."
    )


# ============================================================
# Recent Search Events
# ============================================================

st.header("Recent Searches")

if not events.empty:

    display_columns = [
        "timestamp",
        "query_type",
        "query_text",
        "result_count",
        "top_result_id",
        "response_time_ms",
    ]

    st.dataframe(
        events[display_columns].head(25),
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "No search events recorded."
    )


# ============================================================
# FAISS Index Health
# ============================================================

st.header("Index Health")

if faiss_index is None:

    st.error(
        "FAISS index not found."
    )

else:

    index_type = type(faiss_index).__name__

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "FAISS Vectors",
            f"{faiss_index.ntotal:,}",
        )

    with col2:
        st.metric(
            "Embedding Dimension",
            f"{faiss_index.d}",
        )

    with col3:
        st.metric(
            "Index Type",
            index_type,
        )


# ============================================================
# Embedding Statistics
# ============================================================

st.header("Embedding Statistics")

if faiss_index is not None:

    total_vectors = faiss_index.ntotal
    dimension = faiss_index.d

    st.write(
        f"**Total embeddings:** {total_vectors:,}"
    )

    st.write(
        f"**Embedding dimension:** {dimension}"
    )

    if embedding_metadata is not None:

        product_ids = embedding_metadata.get(
            "product_ids",
            [],
        )

        st.write(
            f"**Mapped products:** {len(product_ids):,}"
        )

        if len(product_ids) == total_vectors:

            st.success(
                "FAISS index and product mapping are aligned."
            )

        else:

            st.error(
                "FAISS index and product mapping are NOT aligned."
            )


# ============================================================
# Embedding Visualization
# ============================================================

st.header("Embedding Visualization")

st.caption(
    "UMAP projection of CLIP product embeddings."
)

if faiss_index is None:

    st.info(
        "FAISS index is required for embedding visualization."
    )

else:

    max_sample_size = min(
        10000,
        faiss_index.ntotal,
    )

    default_sample_size = min(
        2000,
        faiss_index.ntotal,
    )

    sample_size = st.slider(
        "Visualization sample size",
        min_value=500,
        max_value=max_sample_size,
        value=default_sample_size,
        step=500,
    )

    if st.button(
        "Generate Embedding Visualization"
    ):

        with st.spinner(
            "Extracting embeddings and running UMAP..."
        ):

            vectors = faiss_index.reconstruct_n(
                0,
                min(
                    sample_size,
                    faiss_index.ntotal,
                ),
            )

            vectors = np.asarray(
                vectors,
                dtype=np.float32,
            )

            import umap.umap_ as umap

            reducer = umap.UMAP(
                n_components=2,
                random_state=42,
                n_neighbors=15,
                min_dist=0.1,
            )

            projection = reducer.fit_transform(
                vectors
            )

            visualization = pd.DataFrame(
                {
                    "x": projection[:, 0],
                    "y": projection[:, 1],
                }
            )

            # Resolve product categories for the sampled embeddings.
            if embedding_metadata is not None:

                product_ids = embedding_metadata[
                    "product_ids"
                ][:len(visualization)]

                connection = sqlite3.connect(
                    DATABASE_PATH
                )

                try:

                    if product_ids:

                        placeholders = ",".join(
                            "?" for _ in product_ids
                        )

                        category_query = f"""
                            SELECT
                                product_id,
                                category
                            FROM products
                            WHERE product_id IN ({placeholders})
                        """

                        category_df = pd.read_sql_query(
                            category_query,
                            connection,
                            params=list(product_ids),
                        )

                    else:

                        category_df = pd.DataFrame(
                            columns=[
                                "product_id",
                                "category",
                            ]
                        )

                finally:

                    connection.close()

                category_map = dict(
                    zip(
                        category_df["product_id"],
                        category_df["category"],
                    )
                )

                visualization["category"] = [
                    category_map.get(
                        product_id,
                        "UNKNOWN",
                    )
                    for product_id in product_ids
                ]

            st.success(
                f"UMAP projection generated for "
                f"{len(visualization):,} embeddings."
            )

            if "category" in visualization.columns:

                st.scatter_chart(
                    visualization,
                    x="x",
                    y="y",
                    color="category",
                )

            else:

                st.scatter_chart(
                    visualization,
                    x="x",
                    y="y",
                )


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "VisualMind • CLIP + FAISS + SQLite + Kafka + Airflow"
)