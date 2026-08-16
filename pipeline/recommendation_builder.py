"""
VisualMind recommendation builder.

Builds product-to-product recommendations from the existing FAISS
embedding index and persists them into the SQLite recommendations table.
"""

from __future__ import annotations

import os
import pickle
import sqlite3
import time
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv


# Load configuration from the project-level .env file.
load_dotenv()

# Resolve paths relative to the VisualMind project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Persistent FAISS index containing all product embeddings.
FAISS_INDEX_PATH = PROJECT_ROOT / os.getenv(
    "FAISS_INDEX_PATH",
    "./data/faiss_index.bin",
).lstrip("./")

# Mapping between FAISS positions and product IDs.
METADATA_PATH = PROJECT_ROOT / os.getenv(
    "EMBEDDINGS_METADATA_PATH",
    "./data/embeddings_metadata.pkl",
).lstrip("./")

# SQLite database containing product metadata and recommendations.
DATABASE_PATH = PROJECT_ROOT / "data" / "products.db"

# Number of nearest neighbors to retrieve for each product.
RECOMMENDATION_COUNT = 20

# FAISS search batch size prevents unnecessary memory consumption.
BATCH_SIZE = 1024


def load_index() -> tuple[faiss.Index, list[str]]:
    """
    Load the persistent FAISS index and product-position mapping.
    """

    # Load the already-built FAISS index from disk.
    index = faiss.read_index(str(FAISS_INDEX_PATH))

    # Load metadata containing the product ID for every FAISS position.
    with METADATA_PATH.open("rb") as file:
        metadata = pickle.load(file)

    # Extract the ordered product IDs used when building the index.
    product_ids = metadata["product_ids"]

    # Ensure the mapping and FAISS index contain exactly the same number
    # of vectors so recommendation IDs cannot become misaligned.
    if index.ntotal != len(product_ids):
        raise RuntimeError(
            f"FAISS/metadata mismatch: "
            f"{index.ntotal} vectors vs {len(product_ids)} product IDs"
        )

    return index, product_ids


def rebuild_recommendations() -> None:
    """
    Rebuild recommendations for every product in the FAISS index.
    """

    start_time = time.perf_counter()

    print("=" * 60)
    print("VISUALMIND RECOMMENDATION BUILDER")
    print("=" * 60)

    # Load the existing index instead of recomputing product embeddings.
    index, product_ids = load_index()

    total_products = index.ntotal

    print(f"FAISS vectors: {total_products}")
    print(f"Recommendation count: {RECOMMENDATION_COUNT}")
    print(f"Batch size: {BATCH_SIZE}")
    print()

    # Open SQLite directly so the complete recommendation rebuild can
    # use efficient bulk operations without creating thousands of ORM
    # objects and sessions.
    connection = sqlite3.connect(str(DATABASE_PATH))

    try:
        cursor = connection.cursor()

        # Enable foreign-key enforcement for this SQLite connection.
        cursor.execute("PRAGMA foreign_keys = ON")

        # Remove previous recommendations because this function performs
        # a complete deterministic rebuild of the recommendation table.
        cursor.execute("DELETE FROM recommendations")

        connection.commit()

        # Retrieve every embedding from the FAISS index in batches.
        # IndexFlatIP stores the vectors directly, so reconstruct_n gives
        # us the normalized vectors used during the original index build.
        for start in range(0, total_products, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total_products)
            batch_size = end - start

            # Reconstruct the current batch of product embeddings.
            vectors = np.empty(
                (batch_size, index.d),
                dtype=np.float32,
            )

            index.reconstruct_n(
                start,
                batch_size,
                vectors,
            )

            # Search the batch against the complete product index.
            # We request one extra result because the first result is the
            # product itself with similarity approximately equal to 1.0.
            similarities, positions = index.search(
                vectors,
                RECOMMENDATION_COUNT + 1,
            )

            rows: list[tuple[str, str, float, int]] = []

            for local_position in range(batch_size):
                source_position = start + local_position
                source_product_id = product_ids[source_position]

                rank = 1

                for neighbor_index in range(
                    RECOMMENDATION_COUNT + 1
                ):
                    neighbor_position = int(
                        positions[local_position][neighbor_index]
                    )

                    # FAISS returns -1 when a neighbor does not exist.
                    if neighbor_position < 0:
                        continue

                    # Never recommend the same product as itself.
                    if neighbor_position == source_position:
                        continue

                    # Convert the FAISS position back to the stable
                    # product ID stored in SQLite.
                    recommended_product_id = product_ids[
                        neighbor_position
                    ]

                    # FAISS IndexFlatIP returns the inner product.
                    # Because our embeddings are normalized, this is cosine
                    # similarity.
                    similarity = float(
                        similarities[local_position][neighbor_index]
                    )

                    rows.append(
                        (
                            source_product_id,
                            recommended_product_id,
                            similarity,
                            rank,
                        )
                    )

                    rank += 1

                    # Stop once the requested number of recommendations
                    # has been collected for this product.
                    if rank > RECOMMENDATION_COUNT:
                        break

            # Insert the whole batch using SQLite executemany for much
            # better performance than individual INSERT statements.
            cursor.executemany(
                """
                INSERT INTO recommendations (
                    product_id,
                    recommended_product_id,
                    similarity_score,
                    rank
                )
                VALUES (?, ?, ?, ?)
                """,
                rows,
            )

            # Commit each batch so progress is durable and memory remains
            # bounded during the long-running rebuild.
            connection.commit()

            processed = end
            elapsed = time.perf_counter() - start_time
            rate = processed / elapsed if elapsed > 0 else 0.0

            print(
                f"[PROGRESS] "
                f"{processed:,}/{total_products:,} "
                f"({processed / total_products * 100:.1f}%) | "
                f"{rate:.2f} products/sec"
            )

        # Count the final number of generated recommendation records.
        cursor.execute("SELECT COUNT(*) FROM recommendations")
        recommendation_count = cursor.fetchone()[0]

        elapsed = time.perf_counter() - start_time

        print()
        print("=" * 60)
        print("RECOMMENDATION BUILD COMPLETE")
        print("=" * 60)
        print(f"Products:          {total_products:,}")
        print(f"Recommendations:   {recommendation_count:,}")
        print(f"Expected:          {total_products * RECOMMENDATION_COUNT:,}")
        print(f"Elapsed:            {elapsed / 60:.2f} minutes")

    finally:
        # Always close the SQLite connection even if the build fails.
        connection.close()


if __name__ == "__main__":
    rebuild_recommendations()