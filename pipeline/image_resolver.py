"""
ABO image metadata resolver for VisualMind.

This module converts ABO image IDs into paths inside the extracted
small-image dataset.
"""

from __future__ import annotations

import csv
import gzip
from pathlib import Path


class ABOImageResolver:
    """
    Resolve ABO image IDs to local image paths.

    The resolver loads images.csv.gz once into memory because the file
    contains roughly 398k lightweight mappings and is only a few MB
    compressed. This avoids repeatedly scanning the CSV during ingestion.
    """

    def __init__(
        self,
        metadata_path: str | Path,
        images_root: str | Path,
    ) -> None:
        # Store the compressed ABO image metadata location.
        self.metadata_path = Path(metadata_path)

        # Store the root directory containing the extracted small images.
        self.images_root = Path(images_root)

        # Map image IDs to their relative ABO image paths.
        self._image_paths: dict[str, str] = {}

    def load(self) -> None:
        """
        Load the ABO image metadata into an in-memory dictionary.
        """

        # Fail immediately if the metadata file is missing.
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Image metadata file not found: {self.metadata_path}"
            )

        # Open the gzip-compressed CSV as UTF-8 text.
        with gzip.open(
            self.metadata_path,
            mode="rt",
            encoding="utf-8",
            newline="",
        ) as file:
            # DictReader maps CSV column names to values automatically.
            reader = csv.DictReader(file)

            # Process each image metadata record.
            for row in reader:
                # Extract the unique ABO image identifier.
                image_id = row.get("image_id")

                # Extract the relative path provided by ABO.
                relative_path = row.get("path")

                # Ignore malformed rows rather than creating invalid mappings.
                if not image_id or not relative_path:
                    continue

                # Store the normalized mapping for constant-time lookup later.
                self._image_paths[image_id] = relative_path

    def resolve(
        self,
        image_id: str,
    ) -> Path | None:
        """
        Resolve an ABO image ID to its local extracted image path.

        Returns None when the image ID doesn't exist in the metadata or
        when the corresponding image hasn't been extracted locally.
        """

        # Look up the relative path associated with this image ID.
        relative_path = self._image_paths.get(image_id)

        # Return None when ABO doesn't know this image ID.
        if relative_path is None:
            return None

        # The archive stores images under images/small/, while the CSV
        # provides paths relative to that directory.
        image_path = self.images_root / relative_path

        # Return the path only when the actual image exists locally.
        if not image_path.is_file():
            return None

        return image_path

    @property
    def size(self) -> int:
        """
        Return the number of loaded image mappings.
        """

        # Expose the number of known image IDs for validation.
        return len(self._image_paths)