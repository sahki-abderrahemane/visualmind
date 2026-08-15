

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class CLIPEncoder:
    """
    Encode images and text using OpenAI's CLIP ViT-B/32 model.
    """

    MODEL_NAME = "openai/clip-vit-base-patch32"
    EMBEDDING_DIMENSION = 512

    def __init__(self, device: str | None = None) -> None:
        """
        Load the CLIP processor and model once.

        Args:
            device: Optional device override. If omitted, CUDA is used
                    when available, otherwise CPU.
        """

        # Automatically select the GPU when CUDA is available.
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        print(f"Loading CLIP model: {self.MODEL_NAME}")
        print(f"Using device: {self.device}")

        # The processor handles image resizing, normalization and text
        # tokenization exactly as expected by the pretrained CLIP model.
        self.processor = CLIPProcessor.from_pretrained(
            self.MODEL_NAME
        )

        # Load the pretrained CLIP model only once.
        self.model = CLIPModel.from_pretrained(
            self.MODEL_NAME
        )

        # Move the model to the selected device.
        self.model.to(self.device)

        # Disable training-specific behavior because VisualMind only
        # performs inference with the pretrained model.
        self.model.eval()

        print("CLIP model loaded successfully.")

    @torch.inference_mode()
    def encode_image(
        self,
        image: Image.Image | str | Path,
    ) -> np.ndarray:
        """
        Convert an image into a normalized 512-dimensional CLIP vector.

        Args:
            image: PIL Image or path to an image file.

        Returns:
            NumPy array with shape (512,) and L2 norm approximately 1.
        """

        # Load the image when the caller provides a filesystem path.
        if isinstance(image, (str, Path)):
            image = Image.open(image)

        # CLIP expects RGB images.
        image = image.convert("RGB")

        # Convert the image into the tensor format expected by CLIP.
        inputs = self.processor(
            images=image,
            return_tensors="pt",
        )

        # Move every generated tensor to the selected device.
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        # Generate the image embedding without calculating gradients.
        embedding = self.model.get_image_features(
            **inputs
        )

        embedding = torch.nn.functional.normalize(
            embedding,
            p=2,
            dim=-1,
        )

        # Convert the GPU tensor to a CPU NumPy array.
        vector = embedding[0].cpu().numpy().astype(
            np.float32
        )

        # Validate the expected CLIP embedding dimension.
        if vector.shape != (self.EMBEDDING_DIMENSION,):
            raise ValueError(
                f"Unexpected image embedding shape: {vector.shape}. "
                f"Expected ({self.EMBEDDING_DIMENSION},)."
            )

        return vector

    @torch.inference_mode()
    def encode_text(
        self,
        text: str,
    ) -> np.ndarray:
        """
        Convert text into a normalized 512-dimensional CLIP vector.

        Args:
            text: Product search query.

        Returns:
            NumPy array with shape (512,) and L2 norm approximately 1.
        """

        # Reject empty queries because CLIP cannot produce a meaningful
        # search embedding from an empty string.
        if not text or not text.strip():
            raise ValueError(
                "Text query cannot be empty."
            )

        # Tokenize the text using CLIP's official processor.
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        # Move the token tensors to the selected device.
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        # Generate the text embedding without calculating gradients.
        embedding = self.model.get_text_features(
            **inputs
        )

        # Normalize the vector so it can be compared with image vectors
        # using cosine similarity / FAISS inner product.
        embedding = torch.nn.functional.normalize(
            embedding,
            p=2,
            dim=-1,
        )

        # Convert the tensor into a float32 NumPy vector.
        vector = embedding[0].cpu().numpy().astype(
            np.float32
        )

        # Validate the expected CLIP embedding dimension.
        if vector.shape != (self.EMBEDDING_DIMENSION,):
            raise ValueError(
                f"Unexpected text embedding shape: {vector.shape}. "
                f"Expected ({self.EMBEDDING_DIMENSION},)."
            )

        return vector