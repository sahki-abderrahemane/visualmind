"""
VisualMind product recommendation API routes.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.database.database import Product, Recommendation, get_db


# Create the router used by the main FastAPI application.
router = APIRouter(
    prefix="/recommend",
    tags=["recommendations"],
)


@router.get("/{product_id}")
def get_recommendations(
    product_id: str,
    db: Session = Depends(get_db),
) -> dict:
    """
    Return the precomputed nearest-neighbor recommendations for a product.
    """

    # Verify that the requested source product exists in the database.
    product = (
        db.query(Product)
        .filter(Product.product_id == product_id)
        .first()
    )

    if product is None:
        # Return a proper HTTP 404 instead of silently returning an empty
        # recommendation list for an invalid product ID.
        raise HTTPException(
            status_code=404,
            detail=f"Product '{product_id}' not found",
        )

    # Retrieve recommendations in their precomputed FAISS ranking order.
    recommendations = (
        db.query(Recommendation, Product)
        .join(
            Product,
            Product.product_id
            == Recommendation.recommended_product_id,
        )
        .filter(
            Recommendation.product_id == product_id,
        )
        .order_by(
            Recommendation.rank.asc(),
        )
        .all()
    )

    # Convert SQLAlchemy objects into the JSON structure expected by the API.
    results = [
        {
            "rank": recommendation.rank,
            "product_id": recommended_product.product_id,
            "title": recommended_product.title,
            "category": recommended_product.category,
            "color": recommended_product.color,
            "image_path": recommended_product.image_path,
            "similarity": recommendation.similarity_score,
        }
        for recommendation, recommended_product in recommendations
    ]

    return {
        "product_id": product_id,
        "result_count": len(results),
        "recommendations": results,
    }