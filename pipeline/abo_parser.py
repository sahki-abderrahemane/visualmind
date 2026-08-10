"""
ABO dataset parser for VisualMind.

This module converts one raw Amazon Berkeley Objects listing into the
normalized Product representation used by the VisualMind database.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _extract_values(
    field: list[dict[str, Any]] | None,
) -> list[str]:
    """
    Extract textual values from an ABO multilingual field.

    ABO represents fields such as item_name, color, style, and
    bullet_point as arrays of dictionaries containing a "value" key.
    """

    # Return an empty list when the ABO field doesn't exist.
    if not field:
        return []

    # Extract only non-empty string values from the field.
    return [
        str(entry["value"]).strip()
        for entry in field
        if entry.get("value") is not None
        and str(entry["value"]).strip()
    ]


def _first_value(
    field: list[dict[str, Any]] | None,
) -> str | None:
    """
    Return the first available textual value from an ABO field.

    For VisualMind's normalized product representation, we use the
    first available localized value rather than storing every locale.
    """

    # Extract all usable values from the ABO field.
    values = _extract_values(field)

    # Return the first value when one exists.
    return values[0] if values else None


def parse_listing(record: dict[str, Any]) -> dict[str, Any]:
    """
    Convert one raw ABO listing into the VisualMind product schema.

    The resulting dictionary contains only fields required by the
    current VisualMind products table.
    """

    # Extract the stable ABO product identifier.
    product_id = record.get("item_id")

    # Reject malformed records because a product without an ID cannot
    # be inserted into our products table.
    if not product_id:
        raise ValueError("ABO listing is missing item_id")

    # Extract the product's primary image identifier.
    main_image_id = record.get("main_image_id")

    # Extract the product title from ABO's multilingual item_name field.
    title = _first_value(record.get("item_name"))

    # Extract the product category from ABO's product_type field.
    category = _first_value(record.get("product_type"))

    # Extract the color from ABO's multilingual color field.
    color = _first_value(record.get("color"))

    # Extract bullet-point descriptions from the listing.
    bullet_points = _extract_values(record.get("bullet_point"))

    # Extract search keywords from the listing.
    keywords = _extract_values(record.get("item_keywords"))

    # Extract style information from the listing.
    styles = _extract_values(record.get("style"))

    # Combine the textual attributes into one searchable description.
    # Duplicates are removed while preserving their original order.
    description_parts: list[str] = []
    seen: set[str] = set()

    for value in bullet_points + keywords + styles:
        # Normalize whitespace before checking for duplicates.
        normalized_value = " ".join(value.split())

        # Skip empty values and duplicate text.
        if not normalized_value or normalized_value in seen:
            continue

        # Remember the value so repeated metadata isn't stored twice.
        seen.add(normalized_value)

        # Preserve the cleaned value in the final description.
        description_parts.append(normalized_value)

    # Join all useful textual attributes into a single searchable field.
    description = " ".join(description_parts) or None

    # The image path is resolved later from images.csv.gz because the
    # listing itself contains an image ID rather than a filesystem path.
    image_path = None

    # Return the normalized product representation expected by SQLite.
    return {
        "product_id": str(product_id),
        "category": category,
        "color": color,
        "title": title,
        "description": description,
        "image_path": image_path,
        "main_image_id": (
            str(main_image_id)
            if main_image_id
            else None
        ),
    }


def load_first_listing(
    listings_file: str | Path,
) -> dict[str, Any]:
    """
    Load and parse the first valid listing from a JSONL ABO file.

    This helper is intentionally limited to one record for the current
    validation step; the full ingestion loop will be added afterward.
    """

    # Convert the input path into a Path object for reliable filesystem access.
    path = Path(listings_file)

    # Fail immediately if the metadata shard doesn't exist.
    if not path.exists():
        raise FileNotFoundError(
            f"ABO listings file not found: {path}"
        )

    # Open the JSONL file as UTF-8 text.
    with path.open("r", encoding="utf-8") as file:
        # Process records one at a time instead of loading the entire
        # dataset into memory.
        for line in file:
            # Remove whitespace surrounding the JSON record.
            line = line.strip()

            # Ignore empty lines.
            if not line:
                continue

            # Parse the JSON object.
            record = json.loads(line)

            # Convert the raw ABO record into our normalized structure.
            return parse_listing(record)

    # Raise an error if the file contains no usable records.
    raise ValueError(
        f"No valid listings found in {path}"
    )