# Garuda ML Pipeline: Enterprise-Grade Data Science Platform

![ML Pipeline Architecture](Screenshot%202025-07-24%20182358.png)

## Executive Summary

**Garuda ML Pipeline** is a comprehensive, production-ready machine learning platform that automates the data science workflow from data ingestion to model deployment. Built with modern web technologies and an enterprise-grade architecture, the platform streamlines ML operations for data scientists and organizations.

### Current Scope

The **numeric pipeline** (ingestion → preprocessing → augmentation → EDA → modeling → deployment) is fully implemented and production-ready. The **end-to-end text workflow** and **Google OAuth–based authentication** are planned for a future release.

### Key Achievements

* Full-stack implementation: React frontend with FastAPI backend
* End-to-end ML pipeline from data upload to model training and deployment
* Production database: Supabase integration with robust data persistence
* Advanced data processing: automated preprocessing, augmentation, and EDA
* Multi-model support: integrated multiple ML algorithms and frameworks
* Cloud-ready: scalable architecture with cloud storage integration

---

## Implementation Status (Overview)

| Area                    | Numeric Data | Text Data |
| ----------------------- | ------------ | --------- |
| Ingestion & Storage     | Implemented  | Planned   |
| Preprocessing           | Implemented  | Planned   |
| Augmentation            | Implemented  | Planned   |
| EDA                     | Implemented  | Planned   |
| Modeling & Evaluation   | Implemented  | Planned   |
| Deployment & Versioning | Implemented  | Planned   |
| Identity & Access       | Planned      | Planned   |

---

## Architecture Overview

The Garuda ML Pipeline follows a modern microservices architecture with clear separation of concerns.

### Frontend Layer (React + TypeScript)

```
└── React Application
    ├── Data Management Interface
    ├── Preprocessing Controls
    ├── Augmentation Configuration
    ├── EDA Visualization Dashboard
    └── Model Training & Evaluation
```

### Backend Layer (FastAPI + Python)

```
└── FastAPI Application
    ├── RESTful API Endpoints
    ├── Data Processing Modules
    ├── ML Algorithm Integration
    ├── Database Operations
    └── Cloud Storage Management
```

### Data Layer (Supabase + Cloud Storage)

```
└── Supabase Platform
    ├── PostgreSQL Database
    ├── File Storage Buckets
    └── Real-time Subscriptions
```

---

## Core Features

### 1) Intelligent Data Management

* Multi-format support: CSV, JSON, TXT
* Smart data type detection (numeric/text)
* Pipeline configuration and workflow customization
* Input validation, sanitization, and error handling

### 2) Advanced Data Preprocessing

* **Numeric (Implemented)**

  * Missing value imputation (mean, median, mode, forward-fill)
  * Outlier handling (IQR, Z-score)
  * Feature scaling (StandardScaler, MinMaxScaler, RobustScaler)
  * Feature transforms (log, square root, reciprocal)
* **Text (Planned)**

  * Cleaning and normalization
  * Tokenization strategies
  * Stopword removal
  * N-gram analysis

### 3) Data Augmentation Suite

* **Numeric (Implemented)**

  * SMOTE for class imbalance
  * Gaussian noise injection
  * Scale and jitter transformations
* **Text (Planned)**

  * Mixup
  * Synonym replacement
  * Masked language model (MLM) augmentation

### 4) Exploratory Data Analysis (EDA)

* **Numeric (Implemented)**

  * Descriptive statistics and distribution analysis
  * Correlation matrices
  * Feature importance ranking
  * Interactive plots and charts (heatmaps, distributions, box/violin plots)
* **Text (Planned)**

  * Class balance and text-length distributions
  * Top n-grams and keyphrase extraction
  * Keyword-in-class heatmaps
  * Embedding-based similarity explorations

### 5) Multi-Algorithm Model Training

* **Algorithms (Implemented for Numeric)**

  * XGBoost
  * Random Forest
  * Logistic Regression
  * Gradient Boosting
* **Model Features**

  * Hyperparameter tuning
  * Cross-validation
  * Performance metrics and reporting
  * Model persistence and versioning

---

## Technical Stack

### Frontend

* React 18 with TypeScript
* Tailwind CSS
* React Router v6
* React Context API
* Lucide React icons
* Vite build tool

### Backend

* FastAPI (Python)
* Supabase client integration
* Pandas, NumPy, scikit-learn
* XGBoost, Transformers, Sentence-Transformers
* Automatic OpenAPI/Swagger docs

### Database & Storage

* Supabase (PostgreSQL)
* Supabase Storage (multiple buckets)
* Supabase Realtime subscriptions

### DevOps & Deployment

* Python virtual environments
* `requirements.txt` dependency management
* Environment configuration via `.env`
* FastAPI TestClient for API testing

---

## Project Structure

```
Garuda/
├── Frontend (React + TypeScript)
│   └── frontenddata 1.1/project/
│       ├── src/
│       │   ├── components/          # React components
│       │   ├── services/            # API clients and utilities
│       │   ├── contexts/            # React contexts
│       │   └── lib/                 # Library configurations
│       ├── package.json
│       └── tailwind.config.js
│
├── Backend (FastAPI + Python)
│   └── api/
│       ├── routes/
│       │   ├── upload.py
│       │   ├── preprocess.py
│       │   ├── augment.py
│       │   ├── eda.py
│       │   ├── model.py
│       │   └── datasets.py
│       ├── shared/
│       │   ├── types.py
│       │   ├── supabase_utils.py
│       │   └── module_runner.py
│       ├── models/
│       └── main.py
│
├── ML Modules
│   ├── 2and3week/
│   │   ├── PPnumeric/
│   │   └── PPtext/
│   ├── 4and5week/
│   │   └── Augmentation/
│   │       ├── numericaug/
│   │       └── textaug/
│   ├── 6week/
│   │   └── models/
│   └── EDA/
│       ├── numeric_EDA/
│       └── text_EDA/
│
└── Data & Configuration
    ├── package.json
    └── requirements.txt
```

---

## Data Pipeline Workflow

Following the architecture diagram, the pipeline implements a structured data flow.

### 1) Authentication & Access Control (Planned)

```
User Login → OAuth 2.0 (Google via Supabase Auth) → Session Management → RLS Policies
```

### 2) Data Type Classification

```
File Upload → Format Detection → Data Type Analysis → Pipeline Route Selection
```

### 3) Data Ingestion & Storage

```
Raw Data → Validation → Supabase Storage → Database Metadata → Processing Queue
```

### 4) Preprocessing Pipeline

```
Raw Data → Missing Value Handling → Outlier Treatment → Feature Scaling → Clean Data
```

### 5) Augmentation Pipeline

```
Clean Data → Strategy Selection → SMOTE/Noise/Mixup → Augmented Dataset
```

### 6) EDA & Analysis

```
Processed Data → Statistical Analysis → Visualization Generation → Insights Dashboard
```

### 7) Model Training Pipeline

```
Final Dataset → Algorithm Selection → Hyperparameter Tuning → Model Training → Evaluation
```

### 8) Model Deployment & Storage

```
Trained Model → Validation → Cloud Storage → Version Control → Production Ready
```

---

## Database Schema

**Core Tables**

* `users`: Authentication and profile management (planned with OAuth)
* `datasets`: Dataset metadata and lineage tracking
* `pipeline_artifacts`: Processing step results and metadata
* `processing_logs`: Detailed operation logs and debugging
* `data_samples`: Quick data previews and samples
* `data_quality_metrics`: Data quality assessments and metrics

**Storage Buckets**

* `datasets`: Raw uploaded files
* `preprocessed`: Cleaned and processed data
* `augmented`: Augmented dataset variations
* `eda-results`: Analysis results and visualizations
* `models`: Trained model artifacts and metadata

---

## Getting Started

### Prerequisites

* Python 3.10+
* Node.js 18+
* Supabase account

### Backend Setup

```bash
# 1. Set up Python virtual environment
python -m venv data
data\Scripts\activate  # Windows
source data/bin/activate  # Linux/Mac

# 2. Install dependencies
pip install -r api/requirements.txt

# 3. Configure environment
cp api/.env.example api/.env
# Edit .env with your Supabase credentials

# 4. Run FastAPI server
cd api
uvicorn main:app --reload --port 8000
```

### Frontend Setup

```bash
# 1. Navigate to frontend directory
cd "frontenddata 1.1/project"

# 2. Install dependencies
npm install

# 3. Configure environment
cp .env.example .env
# Edit .env with your Supabase and API configuration

# 4. Start development server
npm run dev
```

### Database Migration

```bash
python api/run_migration.py
```

---

## API Documentation

* Swagger UI: `http://localhost:8000/docs`
* ReDoc: `http://localhost:8000/redoc`
* OpenAPI Schema: `http://localhost:8000/openapi.json`

### Key Endpoints

**Data Management**

* `POST /api/upload/file` — Upload and process files
* `GET /api/datasets/user/{user_id}` — Get user datasets
* `GET /api/data-retrieval/artifact/{artifact_id}` — Retrieve processing results

**Data Processing**

* `POST /api/preprocess/numeric` — Numeric data preprocessing
* `POST /api/preprocess/text` — Text data preprocessing
* `POST /api/augment/smote` — SMOTE augmentation
* `POST /api/augment/noise` — Gaussian noise augmentation

**Analysis & Modeling**

* `POST /api/eda/statistical` — Statistical analysis
* `POST /api/eda/correlation` — Correlation analysis
* `POST /api/model/train/xgboost` — XGBoost model training
* `POST /api/model/train/random-forest` — Random Forest training

---

## Testing & Validation

**Automated Testing**

* API testing with FastAPI TestClient
* Unit tests for ML components
* Integration tests for end-to-end pipeline validation

**Manual Testing**

* UI testing across core user journeys
* Data quality checks at each pipeline stage
* Load and performance testing for concurrent users

---

## Roadmap

### Text Workflow (Planned)

* **Ingestion & Schema**: Column tagging for text/label fields; CSV/JSON/TXT support; dataset lineage.
* **Preprocessing**: Normalization, tokenization, stopword removal, lemmatization; configurable pipelines.
* **Augmentation**: Synonym replacement, Mixup, MLM augmentation; optional back-translation.
* **EDA**: Class balance, text length distributions, top n-grams, keyword-in-class heatmaps.
* **Modeling**: Baselines (LogReg/SVM with TF-IDF), transformer models (e.g., DistilBERT); cross-validation and hyperparameter tuning.
* **Metrics**: Accuracy, macro/micro F1, ROC-AUC (binary), confusion matrices.
* **Serving**: FastAPI inference endpoints, model cards, versioning, drift monitoring.
* **Governance**: PII detection/redaction, audit logs.

### Identity & Access (Planned)

* OAuth 2.0 with Google via Supabase Auth
* JWT-based sessions
* Row-Level Security (RLS) policies for dataset isolation
* Role- and tenant-level access controls
* Audit logging and SSO expansion (Okta/Azure AD)

### Platform Enhancements

* AutoML integration for automated model selection and tuning
* Advanced visualizations with D3.js
* Comprehensive model versioning and MLOps lifecycle management
* React Native mobile application
* Multi-tenancy for enterprise organizations
* Containerized deployment with Docker
* Cloud deployment templates for AWS/GCP
* Async processing and caching for performance
* Enhanced authorization layers
* Application performance monitoring and logging

---

## Team & Acknowledgments

**Lead Developer**: \[Arnav Gupta]
**Architecture**: Full-stack ML pipeline implementation
**Technologies**: React, FastAPI, Supabase, Python ML Stack

**Key Accomplishments**

* Designed and implemented the complete numeric ML pipeline
* Built production-ready frontend with the modern React ecosystem
* Developed a scalable FastAPI backend with comprehensive APIs
* Integrated multiple ML algorithms and processing techniques
* Created a comprehensive data processing and augmentation suite
* Built interactive EDA and visualization capabilities

---

## Support & Documentation

* Technical documentation: `/docs` directory
* API reference: interactive Swagger documentation
* Video demos: complete feature walkthroughs
* Issue tracking: GitHub Issues for bugs and feature requests

---

## License

This project is proprietary software developed for \[Garuda]. All rights reserved.

---

**Ready to transform your data science workflow? The Garuda ML Pipeline’s numeric workflow is production-ready; the text workflow and Google OAuth–based authentication are on the roadmap.**
