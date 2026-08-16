"""
VisualMind daily data pipeline.

Runs the FAISS index rebuild followed by recommendation generation.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator


# Resolve the VisualMind project root from this DAG's location.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Add the project root to Python's import path so Airflow can import
# VisualMind's pipeline modules regardless of the current working directory.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def rebuild_faiss_index() -> None:
    """
    Rebuild the FAISS index using the current product dataset.
    """

    # Import inside the task so heavyweight dependencies are only
    # initialized when the task actually executes.
    from pipeline.index_builder import build_index, save_index
    from api.core.clip_encoder import CLIPEncoder
    from api.database.database import SessionLocal

    # Create the CLIP encoder once for the entire indexing operation.
    encoder = CLIPEncoder()

    # Open the SQLite database.
    db = SessionLocal()

    try:
        # Build the FAISS index using the database and encoder.
        index, product_ids = build_index(
            db=db,
            encoder=encoder,
        )

        # Persist the FAISS index and product ID mapping.
        save_index(
            index=index,
            product_ids=product_ids,
        )

    finally:
        # Always close the database session.
        db.close()
    """
    Rebuild the FAISS index using the current product dataset.
    """

    # Import inside the task so Airflow's DAG parser does not initialize
    # FAISS or other heavyweight pipeline dependencies during DAG parsing.
    from pipeline.index_builder import build_index

    # Execute the existing index-building pipeline.
    build_index()


def rebuild_recommendations() -> None:
    """
    Rebuild product-to-product recommendations from the FAISS index.
    """

    # Import inside the task for the same reason as the FAISS task:
    # heavyweight dependencies should only load when the task executes.
    from pipeline.recommendation_builder import rebuild_recommendations as build_recommendations

    # Execute the recommendation builder created in Step 6.1.
    build_recommendations()


# Read the schedule from .env when available, while keeping the required
# 02:00 daily schedule as the fallback.
DAG_SCHEDULE = os.getenv(
    "AIRFLOW_DAG_SCHEDULE",
    "0 2 * * *",
)


# Define the VisualMind daily pipeline.
with DAG(
    dag_id="visualmind_pipeline",
    description=(
        "Daily VisualMind FAISS index rebuild and recommendation generation."
    ),
    schedule=DAG_SCHEDULE,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["visualmind", "faiss", "recommendations"],
) as dag:

    # Rebuild the vector index first so recommendations always use
    # the latest available product embeddings.
    rebuild_faiss = PythonOperator(
        task_id="rebuild_faiss_index",
        python_callable=rebuild_faiss_index,
    )

    # Generate recommendations from the freshly rebuilt FAISS index.
    rebuild_recommendations_task = PythonOperator(
        task_id="rebuild_recommendations",
        python_callable=rebuild_recommendations,
    )

    # The recommendation task must run after the FAISS rebuild completes.
    rebuild_faiss >> rebuild_recommendations_task