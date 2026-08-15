"""
ABO product ingestion for VisualMind.

This module contains the database insertion logic for normalized ABO
products. The current validation step intentionally supports ingesting
a single product before scaling to the complete dataset.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from api.database.database import Product
from pipeline.abo_parser import load_first_listing, parse_listing
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


def ingest_shard(
    db: Session,
    listings_path: str | Path,
    image_resolver: ABOImageResolver,
    *,
    batch_size: int = 1000,
) -> dict[str, int]:
    """
    Ingest one ABO listings shard into SQLite and return ingestion stats.

    The shard can be plain JSONL or gzip-compressed JSONL (.gz).
    """

    path = Path(listings_path)

    if not path.exists():
        raise FileNotFoundError(
            f"ABO listings file not found: {path}"
        )

    stats = {
        "processed": 0,
        "inserted": 0,
        "skipped_missing_image": 0,
        "skipped_invalid": 0,
    }

    pending_inserts = 0
    seen_product_ids_in_shard: set[str] = set()

    opener = gzip.open if path.suffix == ".gz" else open

    with opener(path, mode="rt", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            stats["processed"] += 1

            try:
                record: dict[str, Any] = json.loads(line)
                product_data = parse_listing(record)
            except Exception:
                stats["skipped_invalid"] += 1
                continue

            main_image_id = product_data.get("main_image_id")
            if not main_image_id:
                stats["skipped_invalid"] += 1
                continue

            image_path = image_resolver.resolve(str(main_image_id))
            if image_path is None:
                stats["skipped_missing_image"] += 1
                continue

            product_data["image_path"] = str(image_path)

            product_id = product_data["product_id"]
            if product_id in seen_product_ids_in_shard:
                continue

            already_exists = (
                db.query(Product.product_id)
                .filter(Product.product_id == product_id)
                .first()
            )
            if already_exists is not None:
                seen_product_ids_in_shard.add(product_id)
                continue

            db.add(Product(**product_data))
            seen_product_ids_in_shard.add(product_id)
            pending_inserts += 1
            stats["inserted"] += 1

            if pending_inserts >= batch_size:
                db.commit()
                pending_inserts = 0

    if pending_inserts:
        db.commit()

    return stats