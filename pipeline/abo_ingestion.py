"""
ABO product ingestion for VisualMind.

This module contains the database insertion logic for normalized ABO
products. The current validation step intentionally supports ingesting
a single product before scaling to the complete dataset.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import Session

from api.database.database import Product
from pipeline.abo_parser import load_first_listing
from pipeline.image_resolver import ABOImageResolver


def ingest_first_product(
    db: Session,
    listings_path: str | Path,
    image_resolver: ABOImageResolver,
) -> Product:
    """
    Parse and insert the first ABO product into SQLite.

    This function is intentionally limited to one product for the
    current validation stage.
    """

    # Parse the first valid ABO listing into our normalized product structure.
    product_data = load_first_listing(listings_path)

    # Resolve the product's ABO image ID to its local image path.
    image_path = image_resolver.resolve(
        product_data["main_image_id"]
    )

    # Stop if the listing references an image that isn't available locally.
    if image_path is None:
        raise FileNotFoundError(
            "Could not resolve local image for "
            f"image ID: {product_data['main_image_id']}"
        )

    # The resolver already returns a project-relative path, so we can
    # store it directly without calling Path.relative_to().
    product_data["image_path"] = str(image_path)

    # Check whether this product has already been inserted.
    existing_product = (
        db.query(Product)
        .filter(
            Product.product_id == product_data["product_id"]
        )
        .first()
    )

    # Return the existing product instead of creating a duplicate.
    if existing_product is not None:
        return existing_product

    # Create the SQLAlchemy Product object from the normalized ABO data.
    product = Product(**product_data)

    # Stage the product for insertion into SQLite.
    db.add(product)

    # Commit the transaction so the product is permanently stored.
    db.commit()

    # Refresh the object so SQLAlchemy reflects the committed database state.
    db.refresh(product)

    return product