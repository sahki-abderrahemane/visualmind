"""
VisualMind zero-shot product tagger.

Uses CLIP zero-shot classification to generate category, color,
and style tags for product images.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from api.core.clip_encoder import CLIPEncoder


# Candidate product categories used by zero-shot classification.
CATEGORY_LABELS = [
    "clothing",
    "electronics",
    "furniture",
    "footwear",
    "bags",
    "home decor",
    "sports",
    "books",
    "toys",
]


# Candidate colors used by zero-shot classification.
COLOR_LABELS = [
    "red",
    "blue",
    "green",
    "black",
    "white",
    "yellow",
    "brown",
    "grey",
    "pink",
    "orange",
    "purple",
]


# Candidate styles used by zero-shot classification.
STYLE_LABELS = [
    "modern",
    "vintage",
    "casual",
    "formal",
    "minimalist",
    "luxury",
    "sporty",
]


class ProductTagger:
    """
    CLIP-based zero-shot product tagger.

    Candidate text embeddings are generated once during initialization
    and reused for every product image.
    """

    def __init__(
        self,
        encoder: CLIPEncoder,
    ) -> None:
        # Reuse the application's existing CLIP encoder instead of
        # loading another copy of the model into GPU memory.
        self.encoder = encoder

        # Precompute category text embeddings once.
        self.category_embeddings = self._encode_labels(
            CATEGORY_LABELS
        )

        # Precompute color text embeddings once.
        self.color_embeddings = self._encode_labels(
            COLOR_LABELS
        )

        # Precompute style text embeddings once.
        self.style_embeddings = self._encode_labels(
            STYLE_LABELS
        )

    def _encode_labels(
        self,
        labels: list[str],
    ) -> np.ndarray:
        """
        Encode candidate labels into a matrix of normalized CLIP vectors.
        """

        # Add a natural image-description prompt because CLIP performs
        # zero-shot classification better with a descriptive text prompt.
        embeddings = [
            self.encoder.encode_text(
                f"a photo of {label}"
            )
            for label in labels
        ]

        # Stack individual 512-dimensional vectors into one matrix.
        matrix = np.vstack(embeddings).astype(
            np.float32,
            copy=False,
        )

        # Normalize defensively so cosine similarity remains valid even
        # if the encoder implementation changes in the future.
        norms = np.linalg.norm(
            matrix,
            axis=1,
            keepdims=True,
        )

        # Prevent division by zero for any unexpected zero vector.
        matrix = matrix / np.maximum(norms, 1e-12)

        return matrix

    def _classify(
        self,
        image_embedding: np.ndarray,
        labels: list[str],
        label_embeddings: np.ndarray,
    ) -> tuple[str, float]:
        """
        Select the highest-scoring candidate label for an image.
        """

        # Calculate cosine similarity between the image and every
        # candidate text embedding using a matrix-vector product.
        scores = label_embeddings @ image_embedding

        # Find the candidate with the highest similarity score.
        best_index = int(np.argmax(scores))

        # Return the selected label and its similarity score.
        return (
            labels[best_index],
            float(scores[best_index]),
        )

    def tag_image(
        self,
        image: Image.Image,
    ) -> dict[str, dict[str, Any]]:
        """
        Generate category, color, and style tags for one product image.
        """

        # Ensure the image has the RGB format expected by CLIP.
        image = image.convert("RGB")

        # Encode the image exactly once because the same embedding
        # is reused for category, color, and style classification.
        image_embedding = self.encoder.encode_image(image)

        # Ensure the image vector is normalized before cosine similarity.
        image_norm = np.linalg.norm(image_embedding)

        # Defensively normalize the image embedding.
        image_embedding = image_embedding / max(
            float(image_norm),
            1e-12,
        )

        # Determine the most visually compatible product category.
        category, category_score = self._classify(
            image_embedding,
            CATEGORY_LABELS,
            self.category_embeddings,
        )

        # Determine the most visually compatible dominant color.
        color, color_score = self._classify(
            image_embedding,
            COLOR_LABELS,
            self.color_embeddings,
        )

        # Determine the most visually compatible product style.
        style, style_score = self._classify(
            image_embedding,
            STYLE_LABELS,
            self.style_embeddings,
        )

        # Return structured results for the SQLite ingestion pipeline.
        return {
            "category": {
                "value": category,
                "score": category_score,
            },
            "color": {
                "value": color,
                "score": color_score,
            },
            "style": {
                "value": style,
                "score": style_score,
            },
        }