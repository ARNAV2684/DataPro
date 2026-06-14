# Garuda ML Pipeline - Quick Start Guide

## One-Command Deployment

Anyone with Docker Desktop installed can run the entire Garuda ML Pipeline with just **ONE COMMAND**!

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- Internet connection to download images (~3.2GB total)

### Quick Start

1. **Download the deployment file:**
   ```bash
   curl -O https://raw.githubusercontent.com/ARNAV2684/DataPro/main/docker-compose.hub.yml
   ```

2. **Run the application:**
   ```bash
   docker-compose -f docker-compose.hub.yml up -d
   ```

3. **Access your application:**
   - **Frontend:** http://localhost:3000
   - **Backend API:** http://localhost:8000

That's it! The entire ML pipeline is now running on your localhost! 🎉

### What You Get

- ✅ **Complete ML Pipeline:** Upload datasets, preprocess data, run EDA, train models
- ✅ **Text, Numeric & Image Data:** Full support across all three data types
- ✅ **Multiple ML Models:** Logistic Regression, Random Forest, XGBoost, DistilBERT, and image classifiers (CNN, ResNet, EfficientNet, ViT)
- ✅ **Data Augmentation:** SMOTE, Mixup, Noise injection, image rotation/jitter/cutout/elastic, and more
- ✅ **Advanced EDA:** Statistical analysis, correlation analysis, topic modeling, and image analysis (color, quality, similarity, PCA embeddings)
- ✅ **Production Ready:** Nginx frontend, FastAPI backend with health checks

### Available Docker Images

- **Backend:** `arnav2684/garuda-backend:latest` (~3.1GB)
- **Frontend:** `arnav2684/garuda-frontend:latest` (~80MB)

### Commands Reference

```bash
# Start the application
docker-compose -f docker-compose.hub.yml up -d

# Check logs
docker-compose -f docker-compose.hub.yml logs

# Stop the application
docker-compose -f docker-compose.hub.yml down

# Update to latest images
docker-compose -f docker-compose.hub.yml pull
docker-compose -f docker-compose.hub.yml up -d

# Remove everything (including data)
docker-compose -f docker-compose.hub.yml down -v
docker rmi arnav2684/garuda-backend:latest arnav2684/garuda-frontend:latest
```

### Troubleshooting

**Port 3000 already in use?**
```bash
# Use different port
docker-compose -f docker-compose.hub.yml up -d --scale frontend=0
docker run -d -p 8080:80 --name garuda-frontend-custom arnav2684/garuda-frontend:latest
# Access at http://localhost:8080
```

**Backend health check failing?**
```bash
# Check backend logs
docker-compose -f docker-compose.hub.yml logs backend

# Restart backend
docker-compose -f docker-compose.hub.yml restart backend
```

### Build From Source

If you want to build the images yourself (instead of pulling from Docker Hub):

1. **Configure Supabase credentials:**
   ```bash
   cp api/.env.example api/.env
   # edit api/.env with your SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY
   ```
   See [`SUPABASE_SETUP.md`](SUPABASE_SETUP.md) for the required storage buckets
   (the `eda` bucket must allow public read for visualizations to display).

2. **Build and run:**
   ```bash
   docker compose up --build
   ```
   - **Frontend:** http://localhost:3000
   - **Backend API:** http://localhost:8000 (interactive docs at `/docs`)

   The backend image bundles the full pipeline (numeric, text **and image**
   scripts) and CPU-only PyTorch. The first build downloads ~2–3 GB of Python
   dependencies.

> **Image datasets:** upload a `.zip` of images (optionally with class
> sub-folders for training) or single images. See
> [`IMAGE_PIPELINE.md`](IMAGE_PIPELINE.md) for the full image flow reference.

### Architecture

```
┌─────────────────┐    ┌──────────────────┐
│   Frontend      │    │     Backend      │
│   (nginx)       │───▶│   (FastAPI)      │
│   Port 3000     │    │   Port 8000      │
└─────────────────┘    └──────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   Supabase DB    │
                       │   (PostgreSQL)   │
                       └──────────────────┘
```
### Support

For issues or questions:
- GitHub: [ARNAV2684/DataPro](https://github.com/ARNAV2684/DataPro)
- Create an issue with your problem description and logs
