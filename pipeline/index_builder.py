"""
Build the VisualMind FAISS image-search index.

"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import faiss
import numpy as np
from PIL import Image, UnidentifiedImageError
from sqlalchemy.orm import Session


# Ensure project-root imports work when this script is executed directly
# via "python pipeline/index_builder.py".
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.core.clip_encoder import CLIPEncoder
from api.database.database import Product, SessionLocal


# Project root directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# SQLite product image paths are stored relative to the project root.
IMAGE_ROOT = PROJECT_ROOT

# FAISS index output location.
INDEX_PATH = PROJECT_ROOT / "data" / "faiss_index.bin"

# FAISS position → product ID mapping.
METADATA_PATH = PROJECT_ROOT / "data" / "embeddings_metadata.pkl"

# Number of images encoded before printing progress.
PROGRESS_INTERVAL = 500


def build_index(
    db: Session,
    encoder: CLIPEncoder,
) -> tuple[faiss.Index, list[str]]:
    """
    Generate CLIP embeddings for all products and build the FAISS index.

    Args:
        db: SQLAlchemy database session.
        encoder: Already-loaded CLIP encoder.

    Returns:
        A tuple containing the FAISS index and product ID mapping.
    """

    # Retrieve only products that have a usable image path.
    products = (
        db.query(Product)
        .filter(Product.image_path.isnot(None))
        .order_by(Product.product_id)
        .all()
    )

    print(f"Products with images: {len(products):,}")

    # Store generated embeddings temporarily so FAISS can receive them
    # as one contiguous float32 matrix.
    embeddings: list[np.ndarray] = []

    # Store product IDs in exactly the same order as the embeddings.
    product_ids: list[str] = []

    # Track products that could not be encoded.
    failed = 0

    # Process every product image.
    for position, product in enumerate(products, start=1):
        # Convert the database-relative path into an absolute filesystem path.
        image_path = IMAGE_ROOT / product.image_path

        try:
            # Open the product image.
            with Image.open(image_path) as image:
                # Generate the normalized 512-dimensional CLIP embedding.
                embedding = encoder.encode_image(image)

            # Validate the embedding shape before adding it to the index.
            if embedding.shape != (512,):
                raise ValueError(
                    f"Unexpected embedding shape: {embedding.shape}"
                )

            # Add the embedding to our temporary collection.
            embeddings.append(embedding)

            # Keep the product ID aligned with the FAISS vector position.
            product_ids.append(product.product_id)

        except (
            FileNotFoundError,
            UnidentifiedImageError,
            OSError,
            ValueError,
        ) as exc:
            # One corrupted/missing image should not abort a full index build.
            failed += 1

            print(
                f"[WARNING] Failed product {product.product_id}: "
                f"{exc}"
            )

        # Print progress periodically because 145k images will take time.
        if position % PROGRESS_INTERVAL == 0:
            print(
                f"Processed: {position:,}/{len(products):,} | "
                f"Encoded: {len(embeddings):,} | "
                f"Failed: {failed:,}"
            )

    # Fail explicitly if no embeddings were successfully generated.
    if not embeddings:
        raise RuntimeError(
            "No product embeddings were generated."
        )

    # Convert the list of vectors into the contiguous float32 matrix
    # required by FAISS.
    embedding_matrix = np.ascontiguousarray(
        np.vstack(embeddings),
        dtype=np.float32,
    )

    # Verify that every vector has the expected dimensionality.
    if embedding_matrix.shape[1] != 512:
        raise ValueError(
            f"Unexpected embedding matrix shape: "
            f"{embedding_matrix.shape}"
        )

    # Create an inner-product index.
    #
    # Because every embedding has L2 norm 1, inner product is equivalent
    # to cosine similarity.
    index = faiss.IndexFlatIP(512)

    # Add all normalized CLIP vectors to the FAISS index.
    index.add(embedding_matrix)

    # Verify that FAISS contains exactly the same number of vectors
    # as our product ID mapping.
    if index.ntotal != len(product_ids):
        raise RuntimeError(
            f"FAISS contains {index.ntotal} vectors but there are "
            f"{len(product_ids)} product IDs."
        )

    print()
    print("FAISS index built successfully.")
    print(f"Vectors: {index.ntotal:,}")
    print(f"Dimension: {index.d}")
    print(f"Failed images: {failed:,}")

    return index, product_ids


def save_index(
    index: faiss.Index,
    product_ids: list[str],
) -> None:
    """
    Persist the FAISS index and product mapping to disk.
    """

    # Make sure the output directory exists before writing files.
    INDEX_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save the FAISS binary index.
    faiss.write_index(
        index,
        str(INDEX_PATH),
    )

    # Save the FAISS position → product ID mapping.
    #
    # Position 0 corresponds to product_ids[0],
    # position 1 corresponds to product_ids[1], etc.
    with METADATA_PATH.open("wb") as file:
        pickle.dump(
            {
                "product_ids": product_ids,
                "dimension": index.d,
                "index_type": "IndexFlatIP",
            },
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print()
    print("Index files saved:")
    print(f"FAISS:    {INDEX_PATH}")
    print(f"Metadata: {METADATA_PATH}")


def main() -> None:
    """
    Build and save the complete VisualMind FAISS index.
    """

    print("=" * 60)
    print("VisualMind FAISS Index Builder")
    print("=" * 60)

    # Create the CLIP encoder once for the entire indexing operation.
    encoder = CLIPEncoder()

    # Open the SQLite database.
    db = SessionLocal()

    try:
        # Generate embeddings and construct the FAISS index.
        index, product_ids = build_index(
            db=db,
            encoder=encoder,
        )

        # Persist both the vector index and product mapping.
        save_index(
            index=index,
            product_ids=product_ids,
        )

    finally:
        # Always close the database session.
        db.close()

    print()
    print("=" * 60)
    print("INDEX BUILD COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    # Execute the index builder when this file is run directly.
    main()