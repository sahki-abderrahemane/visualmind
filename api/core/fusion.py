"""
VisualMind multimodal embedding fusion.

Combines CLIP image and text embeddings into a single normalized
embedding that can be searched directly against the existing FAISS index.
"""

from __future__ import annotations

import os

import numpy as np
from dotenv import load_dotenv


# Load the project's .env configuration before reading fusion settings.
load_dotenv()


# Read the configured image contribution from the environment.
IMAGE_ALPHA = float(
    os.getenv("FUSION_IMAGE_ALPHA", "0.6")
)


# Read the configured text contribution from the environment.
TEXT_ALPHA = float(
    os.getenv("FUSION_TEXT_ALPHA", "0.4")
)


# Both weights must be valid proportions.
if not 0.0 <= IMAGE_ALPHA <= 1.0:
    raise ValueError(
        "FUSION_IMAGE_ALPHA must be between 0.0 and 1.0."
    )


if not 0.0 <= TEXT_ALPHA <= 1.0:
    raise ValueError(
        "FUSION_TEXT_ALPHA must be between 0.0 and 1.0."
    )


# The two contributions must form a complete weighted average.
if not np.isclose(IMAGE_ALPHA + TEXT_ALPHA, 1.0):
    raise ValueError(
        "FUSION_IMAGE_ALPHA and FUSION_TEXT_ALPHA must sum to 1.0."
    )


def _normalize(vector: np.ndarray) -> np.ndarray:
    """
    L2-normalize an embedding vector.

    FAISS uses IndexFlatIP, so normalized vectors make inner product
    equivalent to cosine similarity.
    """

    # Convert to float32 because FAISS uses float32 vectors.
    vector = np.asarray(
        vector,
        dtype=np.float32,
    )

    # Calculate the Euclidean norm of the vector.
    norm = np.linalg.norm(vector)

    # Prevent division by zero for an invalid zero embedding.
    if norm == 0.0:
        raise ValueError(
            "Cannot normalize a zero-length embedding."
        )

    # Return the unit-length embedding.
    return vector / norm


def fuse_embeddings(
    image_embedding: np.ndarray,
    text_embedding: np.ndarray,
    image_alpha: float | None = None,
    text_alpha: float | None = None,
) -> np.ndarray:
    """
    Combine image and text embeddings using configurable weights.

    When explicit weights are not provided, the values from .env are used.
    """

    # Use environment configuration when the caller does not provide
    # an explicit runtime override.
    image_weight = (
        IMAGE_ALPHA
        if image_alpha is None
        else float(image_alpha)
    )

    text_weight = (
        TEXT_ALPHA
        if text_alpha is None
        else float(text_alpha)
    )

    # Validate runtime image weight overrides.
    if not 0.0 <= image_weight <= 1.0:
        raise ValueError(
            "image_alpha must be between 0.0 and 1.0."
        )

    # Validate runtime text weight overrides.
    if not 0.0 <= text_weight <= 1.0:
        raise ValueError(
            "text_alpha must be between 0.0 and 1.0."
        )

    # A weighted average requires the two contributions to sum to one.
    if not np.isclose(image_weight + text_weight, 1.0):
        raise ValueError(
            "image_alpha and text_alpha must sum to 1.0."
        )

    # Convert both embeddings to float32 for consistent numerical behavior.
    image_embedding = np.asarray(
        image_embedding,
        dtype=np.float32,
    )

    text_embedding = np.asarray(
        text_embedding,
        dtype=np.float32,
    )

    # Both CLIP embeddings must have exactly the same dimensions.
    if image_embedding.shape != text_embedding.shape:
        raise ValueError(
            "Image and text embeddings must have identical shapes."
        )

    # Calculate the weighted multimodal representation.
    fused_embedding = (
        image_weight * image_embedding
        + text_weight * text_embedding
    )

    # Normalize again because the weighted sum is not necessarily unit length.
    return _normalize(fused_embedding)