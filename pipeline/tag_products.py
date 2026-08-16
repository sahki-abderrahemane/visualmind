"""
VisualMind product tagging pipeline.

Generates CLIP zero-shot tags for ABO product images and stores them
in SQLite. The pipeline is resumable and supports controlled test runs.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from PIL import Image
from sqlalchemy.orm import Session

# Add the project root to Python's import path when this file is executed
# directly with `python pipeline/tag_products.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.core.clip_encoder import CLIPEncoder
from api.core.tagger import ProductTagger
from api.database.database import Product, SessionLocal, Tag


# Number of products processed between SQLite commits.
BATCH_SIZE = 100


def parse_args() -> argparse.Namespace:
    """
    Parse command-line options for controlled pipeline execution.
    """

    parser = argparse.ArgumentParser(
        description="Generate CLIP zero-shot tags for VisualMind products."
    )

    # --limit allows us to safely test the pipeline on a small number
    # of products before processing the complete dataset.
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of products to process.",
    )

    return parser.parse_args()


def get_product_batch(
    db: Session,
    offset: int,
    limit: int,
) -> list[Product]:
    """
    Load one ordered batch of products from SQLite.
    """

    # Load only a bounded number of products so memory usage remains low.
    return (
        db.query(Product)
        .order_by(Product.product_id)
        .offset(offset)
        .limit(limit)
        .all()
    )


def has_tags(
    db: Session,
    product_id: str,
) -> bool:
    """
    Return True when a product already has generated tags.
    """

    # Existing tags allow interrupted executions to resume without
    # repeating CLIP inference for completed products.
    return (
        db.query(Tag.id)
        .filter(Tag.product_id == product_id)
        .first()
        is not None
    )


def tag_product(
    db: Session,
    tagger: ProductTagger,
    product: Product,
) -> tuple[str, str | None]:
    """
    Generate and persist tags for one product.

    Returns:
        ("tagged", None) on success.
        ("skipped", reason) when processing cannot be performed.
    """

    # Do not regenerate tags that were already successfully persisted.
    if has_tags(db, product.product_id):
        return "skipped", "already tagged"

    # Reject database records without an associated image.
    if not product.image_path:
        return "skipped", "missing image path"

    # Convert the stored relative image path into a filesystem path.
    image_path = Path(product.image_path)

    # Protect the entire batch from broken or missing image files.
    if not image_path.is_file():
        return "skipped", f"image not found: {image_path}"

    try:
        # Open the product image for CLIP inference.
        image = Image.open(image_path)

        try:
            # Encode the image once and classify it against the cached
            # category, color, and style text embeddings.
            tag_results = tagger.tag_image(image)
        finally:
            # Release the underlying image file descriptor immediately.
            image.close()

        # Convert each generated tag into its own relational database row.
        for tag_type, result in tag_results.items():
            db.add(
                Tag(
                    product_id=product.product_id,
                    tag_type=tag_type,
                    tag_value=result["value"],
                )
            )

        return "tagged", None

    except Exception as exc:
        # Roll back this product's transaction state so a bad image or
        # inference error cannot break the next product.
        db.rollback()

        return "skipped", f"{type(exc).__name__}: {exc}"


def main() -> None:
    """
    Run the VisualMind product tagging pipeline.
    """

    # Read command-line options before loading the large CLIP model.
    args = parse_args()

    print("=" * 60)
    print("VisualMind — Product Auto-Tagging Pipeline")
    print("=" * 60)

    # Load CLIP exactly once for the entire pipeline process.
    encoder = CLIPEncoder()

    # Build the tagger and cache all candidate text embeddings.
    tagger = ProductTagger(encoder)

    # Open the SQLite session used for product and tag persistence.
    db = SessionLocal()

    start_time = time.perf_counter()

    processed = 0
    tagged = 0
    skipped = 0

    try:
        # Count all products so the normal full run knows its dataset size.
        total_products = db.query(Product).count()

        # Apply the optional command-line limit.
        target_products = total_products

        if args.limit is not None:
            # Reject invalid limits before touching the database.
            if args.limit <= 0:
                raise ValueError(
                    "--limit must be greater than zero."
                )

            # Never process more than the actual database size.
            target_products = min(
                args.limit,
                total_products,
            )

        print(f"Products in database: {total_products:,}")
        print(f"Products targeted:    {target_products:,}")
        print(f"Batch size:           {BATCH_SIZE}")
        print()

        offset = 0

        while offset < target_products:
            # Request only the remaining number of products needed for
            # the current limited or full execution.
            current_batch_size = min(
                BATCH_SIZE,
                target_products - offset,
            )

            # Load the next deterministic batch.
            products = get_product_batch(
                db,
                offset,
                current_batch_size,
            )

            # Stop defensively if SQLite returns no more products.
            if not products:
                break

            for product in products:
                # Generate and persist tags for the current product.
                status, reason = tag_product(
                    db,
                    tagger,
                    product,
                )

                processed += 1

                if status == "tagged":
                    tagged += 1
                else:
                    skipped += 1

                    # Print skipped records so failures can be diagnosed.
                    print(
                        f"[SKIP] {product.product_id}: {reason}"
                    )

            # Persist this completed batch before continuing.
            db.commit()

            offset += len(products)

            # Calculate current processing throughput.
            elapsed = time.perf_counter() - start_time
            rate = processed / elapsed if elapsed > 0 else 0.0
            percent = (
                processed / target_products * 100
                if target_products
                else 100.0
            )

            print(
                f"[PROGRESS] "
                f"{processed:,}/{target_products:,} "
                f"({percent:.2f}%) | "
                f"tagged={tagged:,} | "
                f"skipped={skipped:,} | "
                f"{rate:.2f} products/sec"
            )

    except KeyboardInterrupt:
        # Preserve all completed database work when manually interrupted.
        db.commit()

        print()
        print("Pipeline interrupted by user.")
        print("Completed batches have been committed.")

    finally:
        # Always close the SQLite session.
        db.close()

    elapsed = time.perf_counter() - start_time

    print()
    print("=" * 60)
    print("TAGGING RUN COMPLETE")
    print("=" * 60)
    print(f"Processed: {processed:,}")
    print(f"Tagged:    {tagged:,}")
    print(f"Skipped:   {skipped:,}")
    print(f"Elapsed:   {elapsed / 60:.2f} minutes")

    if elapsed > 0:
        print(
            f"Rate:      "
            f"{processed / elapsed:.2f} products/sec"
        )


if __name__ == "__main__":
    main()