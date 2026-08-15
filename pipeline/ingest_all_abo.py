"""
Ingest all ABO listing shards into the VisualMind SQLite database.
"""

from pathlib import Path
import sys


# Ensure project-root imports work when this script is executed directly
# via "python pipeline/ingest_all_abo.py".
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.database.database import SessionLocal
from pipeline.abo_ingestion import ingest_shard
from pipeline.image_resolver import ABOImageResolver


# Directory containing the compressed ABO listing shards.
METADATA_DIR = Path("data/abo/metadata")

# Compressed ABO image metadata used to resolve image IDs.
IMAGE_METADATA = METADATA_DIR / "images.csv.gz"

# Root directory containing the extracted small ABO images.
IMAGE_ROOT = Path("data/abo/images/small")


def main() -> None:
    """Ingest all available ABO listing shards."""

    # Create the image resolver once because all shards use the same
    # images.csv.gz mapping.
    resolver = ABOImageResolver(
        IMAGE_METADATA,
        IMAGE_ROOT,
    )

    # Load the ~398k image ID → path mappings once instead of scanning
    # images.csv.gz for every product.
    resolver.load()

    print(f"Loaded {resolver.size:,} image mappings.")

    # Discover available listing shards from metadata files to avoid
    # hard-failing when only a subset of shards exists locally.
    shard_candidates = sorted(
        METADATA_DIR.glob("listings_*.json.gz")
    )
    if not shard_candidates:
        shard_candidates = sorted(
            METADATA_DIR.glob("listings_*.json")
        )

    if not shard_candidates:
        raise FileNotFoundError(
            "No ABO listing shards found under data/abo/metadata"
        )

    # Open one database session for the complete ingestion operation.
    db = SessionLocal()

    # Keep aggregate statistics across all shards.
    total_processed = 0
    total_inserted = 0
    total_missing_images = 0
    total_invalid = 0

    try:
        # Process each ABO listing shard sequentially.
        for shard_path in shard_candidates:
            shard_name = shard_path.stem.replace(
                "listings_",
                "",
            ).replace(".json", "")

            print()
            print("=" * 60)
            print(f"Processing {shard_path}")
            print("=" * 60)

            # Ingest the current shard using batched SQLite transactions.
            stats = ingest_shard(
                db=db,
                listings_path=shard_path,
                image_resolver=resolver,
            )

            # Add the shard statistics to the global totals.
            total_processed += stats["processed"]
            total_inserted += stats["inserted"]
            total_missing_images += stats["skipped_missing_image"]
            total_invalid += stats["skipped_invalid"]

            # Display the result immediately so we know which shard
            # completed successfully.
            print()
            print(f"Shard {shard_name} complete:")
            print(f"  Processed:       {stats['processed']:,}")
            print(f"  Inserted:        {stats['inserted']:,}")
            print(
                f"  Missing images:  "
                f"{stats['skipped_missing_image']:,}"
            )
            print(
                f"  Invalid records: "
                f"{stats['skipped_invalid']:,}"
            )

    finally:
        # Always close the database session even if a shard fails.
        db.close()

    # Print the final ingestion summary.
    print()
    print("=" * 60)
    print("ABO INGESTION COMPLETE")
    print("=" * 60)
    print(f"Total processed:       {total_processed:,}")
    print(f"Total inserted:        {total_inserted:,}")
    print(f"Total missing images:  {total_missing_images:,}")
    print(f"Total invalid records: {total_invalid:,}")


if __name__ == "__main__":
    # Execute the complete ingestion when this file is run directly.
    main()