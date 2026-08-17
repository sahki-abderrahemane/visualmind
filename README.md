# VisualMind

**Multimodal visual product search and recommendation engine built for integration into modern applications.**

VisualMind is an end-to-end AI search platform that turns product images and natural-language descriptions into semantic search results and recommendations.

It combines **CLIP embeddings, FAISS vector search, multimodal fusion, zero-shot tagging, recommendation generation, FastAPI, Apache Airflow, Kafka, SQLite, and Streamlit analytics** into a modular architecture that can be integrated into an existing e-commerce platform, marketplace, catalog, ERP, mobile application, or other product-based system.

The core AI capabilities are exposed through a REST API, making VisualMind independent from the frontend or application consuming it.

---

## What VisualMind Does

VisualMind supports three search modes:

### Image Search

Upload a product image and retrieve visually similar products.

```text
Image
  ↓
CLIP
  ↓
512D embedding
  ↓
FAISS
  ↓
Top-K products
```

### Text Search

Search using natural language.

```text
"women's blue heels"
        ↓
      CLIP
        ↓
  Text embedding
        ↓
      FAISS
        ↓
   Product results
```

### Multimodal Search

Combine an image with a textual refinement.

```text
              ┌──→ Image → CLIP ──┐
User Query ───┤                    ├──→ Fusion → FAISS
              └──→ Text  → CLIP ──┘
```

The default fusion configuration is:

```text
Image: 60%
Text:  40%
```

The weights are configurable through environment variables.

---

# Designed as an Integration Layer

VisualMind is not tied to a specific frontend or e-commerce application.

Its core functionality is exposed through HTTP endpoints, meaning it can be integrated into almost any system capable of making REST requests.

```text
┌───────────────────────┐
│ Existing Application  │
│                       │
│ Web / Mobile / ERP    │
│ Marketplace / SaaS    │
└───────────┬───────────┘
            │
            │ HTTP
            ▼
┌──────────────────────────┐
│      VisualMind API      │
│         FastAPI          │
├──────────────────────────┤
│ Image Search             │
│ Text Search               │
│ Multimodal Search         │
│ Recommendations           │
│ Product Tags              │
└────────────┬─────────────┘
             │
             ▼
       ┌─────────────┐
       │ CLIP + FAISS│
       └─────────────┘
```

This allows VisualMind to act as a **standalone AI search service** behind an existing product.

For example:

```text
Existing E-Commerce
        │
        ├── Product Catalog
        ├── Authentication
        ├── Payments
        └── Orders
                │
                ▼
          VisualMind API
                │
                ├── Visual Search
                ├── Semantic Search
                ├── Multimodal Search
                └── Recommendations
```

The consuming application does not need to know how CLIP, FAISS, or the recommendation pipeline works.

---

# API

### Image Search

```http
POST /search/image
```

Upload an image and retrieve the most visually similar products.

### Text Search

```http
POST /search/text
```

Search the product catalog using natural language.

### Multimodal Search

```http
POST /search/multimodal
```

Combine image similarity with textual intent.

### Product Recommendations

```http
GET /recommend/{product_id}
```

Retrieve precomputed recommendations for a product.

### Product Tags

```http
GET /products/{id}/tags
```

Retrieve automatically generated CLIP-based tags.

### Health

```http
GET /health
```

Used to verify API and AI infrastructure availability.

---

# Core Architecture

```text
                           Client Applications
                                  │
                   ┌──────────────┼──────────────┐
                   │              │              │
                 Image           Text       Image + Text
                   │              │              │
                   └──────────────┼──────────────┘
                                  ▼
                         ┌─────────────────┐
                         │     FastAPI     │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │      CLIP       │
                         │     512D        │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │      FAISS      │
                         │   IndexFlatIP   │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ Product Results │
                         └────────┬────────┘
                                  │
             ┌────────────────────┼────────────────────┐
             │                    │                    │
             ▼                    ▼                    ▼
          SQLite                Kafka            Recommendations
             │                    │                    │
             └───────────────────┬─────────────────────┘
                                 ▼
                         ┌─────────────────┐
                         │    Streamlit    │
                         │    Analytics    │
                         └─────────────────┘
                         ┌─────────────────┐
                         │     Airflow     │
                         └────────┬────────┘
                                  │
                     ┌────────────┴────────────┐
                     ▼                         ▼
              FAISS Rebuild             Recommendations
```

---

# AI Search Engine

VisualMind uses:

**CLIP**

```text
openai/clip-vit-base-patch32
```

Images and text are transformed into a shared **512-dimensional embedding space**.
Embeddings are L2-normalized before indexing.

**FAISS**

```text
IndexFlatIP
```

The final catalog contains:

```text
145,050 products
145,050 product images
512-dimensional embeddings
```

The resulting vector index contains **145,050 vectors**.

This architecture allows the search engine to work with semantic similarity rather than depending exclusively on exact keyword matching.

---

# Multimodal Fusion

VisualMind can combine independent image and text representations:

```text
Image embedding ──┐
                  ├──→ Weighted Average
Text embedding ───┘
                         ↓
                  L2 Normalization
                         ↓
                       FAISS
```

This makes queries such as:

> "Find something similar to this image, but in blue"

possible without requiring a separate multimodal model or search infrastructure.

---

# Recommendation Engine

Recommendations are generated from the same semantic embedding infrastructure used for search.

```text
Product
   ↓
CLIP embedding
   ↓
FAISS nearest-neighbor search
   ↓
Remove source product
   ↓
Top 20
   ↓
SQLite
```

Recommendations are computed offline and stored in the database.

This keeps the recommendation API lightweight:

```text
GET /recommend/{product_id}
```

The API does not need to perform an expensive vector search every time a product page is opened.

---

# Automated Product Tagging

VisualMind uses CLIP zero-shot classification to generate product metadata without training a dedicated classifier.

Supported tag dimensions include:

**Categories**

```text
clothing
electronics
furniture
footwear
bags
home decor
sports
books
toys
```

**Colors**

```text
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
```

**Styles**

```text
modern
vintage
casual
formal
minimalist
luxury
sporty
```

Tags are persisted in the product database and exposed through the API.

---

# Data Pipeline

The system was built using the **Amazon Berkeley Objects (ABO)** dataset.

The ingestion pipeline handles:

* Metadata shard processing
* Image metadata
* Image extraction
* Image validation
* Image ID resolution
* Product normalization
* SQLite ingestion
* Dataset consistency checks

Final catalog:

```text
Records processed:       147,702
Newly inserted:          135,860
Invalid records:             575
Missing images:                0
Products:                145,050
Products with images:    145,050
Categories:                  573
Database size:             ~167 MB
```

---

# Workflow Automation

Apache Airflow handles the recurring offline pipeline.

The production-oriented flow is:

```text
Daily Schedule
      │
      ▼
rebuild_faiss_index
      │
      ▼
rebuild_recommendations
```

This separates expensive batch processing from real-time API requests.

The system can therefore refresh its search and recommendation data without requiring the API to rebuild everything during a user request.

---

# Event-Driven Architecture

Every search is persisted locally and published as an event.

```text
                         Search
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
        SQLite event               Kafka event
                                      │
                                      ▼
                         visualmind.search.events
```

Example:

```json
{
  "query_type": "text",
  "query_text": "women's blue heels",
  "result_count": 10,
  "top_result_id": "B078FBX4HK",
  "response_time_ms": 277.595,
  "timestamp": "2026-08-17T14:33:21.704162+00:00"
}
```

Kafka decouples search analytics from the API itself, allowing additional consumers to be added later without modifying the search engine.

---

# Analytics

The included Streamlit dashboard provides visibility into the system.

### Search Metrics

* Total searches
* Daily activity
* Weekly activity
* Search modality distribution

### Search Analytics

* Top queries
* Zero-result searches
* Search trends

### Index Health

* FAISS vector count
* Embedding statistics
* Index status
* Rebuild information

### Embedding Visualization

```text
512D CLIP Embeddings
        ↓
       UMAP
        ↓
       2D
        ↓
Category Visualization
```

The dashboard is primarily an observability and exploration layer; the core search engine remains independent of Streamlit.

---

# Project Structure

```text
visualmind/
│
├── api/
│   ├── core/
│   │   ├── clip_encoder.py
│   │   ├── faiss_index.py
│   │   ├── fusion.py
│   │   ├── kafka_producer.py
│   │   └── tagger.py
│   │
│   ├── database/
│   │   └── database.py
│   │
│   ├── routes/
│   │   ├── analytics.py
│   │   ├── products.py
│   │   ├── recommendations.py
│   │   └── search.py
│   │
│   └── main.py
│
├── dashboard/
│   └── app.py
│
├── pipeline/
│   ├── dags/
│   │   └── visualmind_pipeline.py
│   ├── index_builder.py
│   └── recommendation_builder.py
│
├── data/
│   ├── abo/
│   ├── products.db
│   ├── faiss_index.bin
│   └── embeddings_metadata.pkl
│
├── docker/
│   └── docker-compose.yml
│
├── tests/
│
├── requirements.txt
├── .env
└── README.md
```

---

# Technology Stack

| Layer            | Technologies                             |
| ---------------- | ----------------------------------------- |
| AI / Embeddings  | PyTorch, Hugging Face Transformers, CLIP |
| Vector Search    | FAISS                                     |
| Backend          | FastAPI, Uvicorn                          |
| Database         | SQLite, SQLAlchemy                        |
| Data Processing  | Pandas, PyArrow, Pillow                   |
| Workflow         | Apache Airflow                            |
| Streaming        | Apache Kafka, Zookeeper                   |
| Analytics        | Streamlit, Plotly, UMAP                   |
| Infrastructure   | Docker, Docker Compose                    |
| Testing          | Pytest, HTTPX                             |

---

# Engineering Focus

VisualMind is intentionally built as more than a machine-learning demo.

The project explores the engineering problems involved in turning a multimodal model into a reusable AI service:

* Large-scale dataset ingestion
* Persistent vector indexing
* Model lifecycle management
* Efficient similarity search
* Multimodal embedding fusion
* Offline recommendation generation
* API/service separation
* Workflow orchestration
* Event-driven architecture
* Search telemetry
* Embedding visualization
* Dependency and infrastructure management

The result is a modular architecture where each component can evolve independently.

For example:

```text
             ┌───────────────┐
             │    Frontend   │
             └───────┬───────┘
                     │
                     ▼
             ┌───────────────┐
             │  VisualMind   │
             │      API      │
             └───────┬───────┘
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
        Search   Recommend    Tags
          │          │          │
          └──────────┼──────────┘
                     ▼
              Product Database
```

The frontend, database, catalog system, or business application around VisualMind can therefore be replaced without changing the fundamental search architecture.

---

# Integration Possibilities

VisualMind can serve as an AI layer for:

* E-commerce platforms
* Online marketplaces
* Product catalogs
* Retail applications
* Mobile shopping applications
* ERP systems with product catalogs
* Fashion platforms
* Furniture marketplaces
* Electronics catalogs
* Internal enterprise search systems
* SaaS products requiring semantic product discovery

A consuming application only needs to communicate with the API.

For example:

```text
Mobile App
    │
    │ POST /search/image
    ▼
VisualMind API
    │
    ▼
AI Search
    │
    ▼
JSON Results
```

This makes the system suitable for integration with existing applications rather than requiring the entire application to be rebuilt around VisualMind.

---

# Current Capabilities

```text
┌─────────────────────────────────────────────┐
│              VisualMind                     │
├─────────────────────────────────────────────┤
│                                             │
│  ✓ Image Search                             │
│  ✓ Text Search                              │
│  ✓ Multimodal Search                        │
│  ✓ CLIP Embeddings                          │
│  ✓ FAISS Vector Search                      │
│  ✓ Zero-Shot Product Tagging                │
│  ✓ Product Recommendations                  │
│  ✓ FastAPI REST API                         │
│  ✓ SQLite Persistence                       │
│  ✓ Airflow Pipeline                         │
│  ✓ Kafka Search Events                      │
│  ✓ Streamlit Analytics                      │
│  ✓ UMAP Embedding Visualization             │
│  ✓ Dockerized Kafka Infrastructure          │
│  ✓ API / AI Layer Separation                │
│                                             │
└─────────────────────────────────────────────┘
```

---

# Status

**Completed — v1**

All planned components from the initial project architecture have been implemented and validated.

```text
Step 0  Environment & Infrastructure       ✓
Step 1  Dataset Engineering                ✓
Step 2  CLIP + FAISS                       ✓
Step 3  FastAPI Search                     ✓
Step 4  Multimodal Fusion                  ✓
Step 5  Auto-Tagging                       ✓
Step 6  Recommendations + Airflow          ✓
Step 7  Kafka + Streamlit Analytics        ✓
```

---

# License

Add your preferred license here.