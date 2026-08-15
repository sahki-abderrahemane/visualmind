VisualMind — Full Project Schedule
Step 0 — Environment Setup ✅
Create full folder structure
Create Python virtual environment
Create requirements.txt
Create .env
Create Docker Compose
Kafka
Zookeeper
Kafka UI
Resolve Python 3.13 / SQLAlchemy / Airflow dependency setup
Kafka running
Step 1 — Dataset Setup ✅
ABO Dataset
Download ABO metadata
Download all 16 listing shards
Download images.csv.gz
Download abo-images-small.tar
Extract small images
Validate image files
Build ABO image ID → local path resolver
Parse ABO product metadata
Create SQLite database
Ingest all listing shards
Validate database records
Current dataset
ABO records processed:    147,702
Newly inserted:            135,860
Invalid records:               575
Missing images:                 0


Products in SQLite:        145,050
Products with images:      145,050
Distinct categories:           573
Database size:                167 MB
Step 2 — CLIP Encoder + FAISS Index ⏳ NEXT
2.1 CLIP Encoder

File:

api/core/clip_encoder.py
Load openai/clip-vit-base-patch32
Load model only once
Implement encode_image(image)
Implement encode_text(text)
Produce 512-dimensional embeddings
L2-normalize embeddings
Test image encoding
Test text encoding
2.2 FAISS Index Builder

File:

pipeline/index_builder.py
Load products from SQLite
Iterate through product images
Generate CLIP embeddings
Build faiss.IndexFlatIP
Store normalized vectors
Save:
data/faiss_index.bin
Save product ↔ FAISS position mapping:
data/embeddings_metadata.pkl
Verify index size
Verify similarity search
Step 3 — FastAPI Search ⏳
API

Files:

api/main.py
api/routes/search.py
api/routes/products.py
Image Search
POST /search/image
Image upload
CLIP encoding
FAISS search
Top-k products
Similarity scores
Text Search
POST /search/text
Text input
CLIP encoding
FAISS search
Top-k products
Similarity scores
Startup architecture
FastAPI startup
      │
      ├── Load CLIP ONCE
      │
      └── Load FAISS ONCE

Never reload either model/index per request.

Step 4 — Multimodal Fusion ⏳

Endpoint:

POST /search/multimodal
Pipeline
Image ──→ CLIP ──┐
                 ├──→ Weighted Fusion ──→ Normalize ──→ FAISS
Text ───→ CLIP ──┘
Default weights
Image: 0.6
Text:  0.4

Configurable through .env.

Image encoding
Text encoding
Weighted average
L2 normalization
FAISS search
Return combined results
Step 5 — Auto-Tagging ⏳

File:

api/core/tagger.py

Use CLIP zero-shot classification.

Categories
clothing
electronics
furniture
footwear
bags
home decor
sports
books
toys
Colors
red
blue
green
black
white
yellow
brown
grey
pink
orange
purple
Styles
modern
vintage
casual
formal
minimalist
luxury
sporty
Database

Store generated tags in:

tags
API
GET /products/{id}/tags
Step 6 — Recommendations + Airflow ⏳
Recommendation Engine

File:

pipeline/recommendation_builder.py

For every product:

Product embedding
       ↓
FAISS
       ↓
Top 20 nearest products
       ↓
SQLite recommendations
Generate recommendations
Exclude the product itself
Store top-20 neighbors
Validate recommendations
Airflow

File:

pipeline/dags/visualmind_pipeline.py

Daily at:

02:00

Tasks:

rebuild_faiss_index
        ↓
rebuild_recommendations
API
GET /recommend/{product_id}
Step 7 — Kafka + Streamlit Analytics ⏳
Kafka

File:

api/core/kafka_producer.py

Every search produces an event to:

visualmind.search.events

Event:

{
  "query_type": "image",
  "query_text": "...",
  "result_count": 10,
  "top_result_id": "...",
  "response_time_ms": 42,
  "timestamp": "..."
}

Every search must do both:

Search
 ├──→ SQLite search_events
 │
 └──→ Kafka visualmind.search.events

These Kafka events become the data source for NexusFlow.

Streamlit Dashboard

File:

dashboard/app.py

Dashboard sections:

Search Metrics
Total searches today
Total searches this week
Image vs text vs multimodal
Search Analytics
Top 10 queries
Zero-result queries
Search trends
Index Health
FAISS index size
Last rebuild time
Embedding statistics
Embedding Visualization
CLIP embeddings
      ↓
UMAP
      ↓
2D visualization
      ↓
Colored by product category
Final VisualMind Architecture
                    ┌──────────────────┐
                    │   User / Client  │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
           Image           Text        Image + Text
              │              │              │
              └──────────────┼──────────────┘
                             ↓
                    ┌─────────────────┐
                    │   FastAPI       │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │      CLIP       │
                    │  512D vectors   │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │      FAISS      │
                    │  IndexFlatIP    │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Product Results │
                    └─────────────────┘
                             │
              ┌──────────────┼──────────────┐
              ↓              ↓              ↓
           SQLite          Kafka       Recommendations
              │              │              │
              │              ↓              │
              │         NexusFlow           │
              │                             │
              └──────────────┬──────────────┘
                             ↓
                    ┌─────────────────┐
                    │    Streamlit    │
                    │    Dashboard    │
                    └─────────────────┘


                    Airflow
                       │
             ┌─────────┴─────────┐
             ↓                   ↓
       FAISS rebuild      Recommendations

Current position:

Step 0 ✅
Step 1 ✅
Step 2 ⏳ ← WE ARE HERE
Step 3
Step 4
Step 5
Step 6
Step 7

Next action: api/core/clip_encoder.py — nothing beyond that until we verify it.