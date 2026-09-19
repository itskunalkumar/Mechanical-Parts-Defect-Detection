# 🔧 Mechanical Parts Defect Detection (Deep Learning)

An end-to-end deep learning project that classifies casting/manufacturing parts as
**defective** or **OK** from images — using transfer learning (MobileNetV2), served
via a FastAPI REST API, with explainability, monitoring, batch inference, and a
CI/CD pipeline that auto-builds and pushes the Docker image.

## Overview

Manufacturing quality control traditionally relies on manual visual inspection.
This project automates defect detection from part images, combining a mechanical
engineering problem with a practical computer vision / MLOps pipeline.

## Dataset

[Casting Product Image Data for Quality Inspection](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)
(Kaggle) — binary classes: `def_front` (defective) and `ok_front` (ok).

Download and place under:
```
data/casting_data/train/{def_front, ok_front}/
data/casting_data/test/{def_front, ok_front}/
```

## Results

| Metric | Score |
|---|---|
| Test Accuracy | 99.3% |
| Precision | 98.1% |
| Recall | 100% |

## Features

- **Transfer learning** — MobileNetV2 backbone, fine-tuned classification head
- **Grad-CAM explainability** — every prediction in the UI shows a heatmap of
  which region of the part image the model focused on, instead of a black-box label
- **REST API** — `/predict` (single image, JSON) and `/predict-batch` (multiple
  images in one request, with a summary count of defective vs ok)
- **Monitoring dashboard** (`/dashboard`) — logs every prediction to SQLite and
  tracks volume, class balance, average confidence, and flags low-confidence
  predictions for manual review
- **CI/CD** — GitHub Actions runs a model smoke test on every push to `main`,
  then builds and pushes the Docker image to Docker Hub automatically
- **Dockerized** — self-contained image, runs anywhere

## Pipeline

| Stage | File | What it does |
|---|---|---|
| Data Ingestion | `src/components/data_ingestion.py` | Validates dataset structure, reports class counts |
| Data Transformation | `src/components/data_transformation.py` | Builds augmented train/val/test generators |
| Model Training | `src/components/model_trainer.py` | Transfer learning on frozen MobileNetV2 + custom head |
| Explainability | `src/pipeline/gradcam.py` | Grad-CAM heatmap generation |
| Monitoring | `src/components/monitoring.py` | SQLite prediction logging + stats |
| Prediction Pipeline | `src/pipeline/predict_pipeline.py` | Loads saved model, predicts (single + batch) |
| Serving | `app/main.py` | FastAPI app — UI (`/`), API (`/predict`, `/predict-batch`), dashboard (`/dashboard`) |

## Tech Stack

Python, TensorFlow/Keras, FastAPI, Docker, SQLite, GitHub Actions, Matplotlib (Grad-CAM)

## Run locally

```bash
pip install -r requirements.txt

# Train (after placing dataset under data/casting_data/)
python -m src.pipeline.train_pipeline

# Serve
uvicorn app.main:app --reload
```

Visit `http://localhost:8000` for the UI, `http://localhost:8000/dashboard` for monitoring.

## Run with Docker

```bash
docker build -t defect-detection .
docker run -p 8000:8000 defect-detection
```

Or pull the pre-built image:
```bash
docker pull kunalbtech2024/defect-detection:latest
docker run -p 8000:8000 kunalbtech2024/defect-detection:latest
```

## API

**Single prediction** — `POST /predict`, multipart form-data with a `file` field.
```json
{"label": "Defective", "confidence": 92.4}
```

**Batch prediction** — `POST /predict-batch`, multipart form-data with multiple `files`.
```json
{
  "results": [{"filename": "part1.jpg", "label": "OK", "confidence": 98.1}, ...],
  "summary": {"total": 5, "defective": 1, "ok": 4}
}
```

## CI/CD Setup

To enable the GitHub Actions pipeline, add these repository secrets
(Settings → Secrets and variables → Actions):
- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN` (a Docker Hub access token, not your password)

Every push to `main` will then: run a model smoke test → build the Docker image →
push it to Docker Hub automatically.

## Future Improvements

- [ ] Multi-class defect classification (not just binary)
- [ ] Data drift detection comparing live traffic to training distribution
- [ ] Auto-deploy to Render via webhook after a successful image push
