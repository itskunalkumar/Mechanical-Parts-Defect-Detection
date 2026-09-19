# 🔧 AURA — AI Casting Defect Inspection

An end-to-end deep learning system that inspects casting/manufacturing parts from
images and classifies them as **Defective** or **OK** in real time — with visual
explainability (Grad-CAM), a live monitoring dashboard, a REST API, and a fully
automated CI/CD pipeline that builds and ships the Docker image on every push.

**🔗 Live demo: [defect-detection-5e9g.onrender.com](https://defect-detection-5e9g.onrender.com/)**
*(hosted on Render's free tier — the first request after inactivity may take ~30–50s to wake the server)*

---

## Why this project

Manufacturing quality control traditionally relies on manual visual inspection,
which is slow and inconsistent. This project automates defect detection from
part images using computer vision — combining a mechanical engineering problem
with a practical, deployed ML/MLOps pipeline, end to end: data → model →
explainability → API → monitoring → CI/CD → live deployment.

## Results

| Metric | Score |
|---|---|
| Test Accuracy | 99.3% |
| Precision | 98.1% |
| **Recall (catches every defective part)** | **100%** |

Model: MobileNetV2 backbone (transfer learning, frozen), fine-tuned classification
head, trained on the [Casting Product Image Data for Quality Inspection](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)
dataset (~6,600 images, binary: defective / ok).

## Features

- **3D interactive inspection console** — upload a part image and watch it move
  through a live scanning pipeline UI before the model returns a verdict
- **Grad-CAM explainability** — every prediction includes a heatmap showing
  exactly which region of the part influenced the model's decision, instead of
  a black-box label
- **REST API** — `/predict` (single image) and `/predict-batch` (multiple images
  in one request, with a pass/fail summary count)
- **Live monitoring dashboard** (`/dashboard`) — logs every inspection to SQLite
  and tracks total volume, pass/fail split, average confidence, and flags
  low-confidence predictions for manual review
- **CI/CD pipeline** — GitHub Actions runs a model smoke test on every push to
  `main`, then automatically builds and pushes the Docker image to Docker Hub
- **Dockerized & deployed** — self-contained image, live on Render

## Architecture

```
data_ingestion → data_transformation → model_trainer → artifacts/defect_model.h5
                                                              │
                                                              ▼
                                            predict_pipeline (+ Grad-CAM)
                                                              │
                                                              ▼
                                    FastAPI (/, /predict, /predict-batch, /dashboard)
                                                              │
                                              ┌───────────────┴───────────────┐
                                              ▼                               ▼
                                     SQLite (monitoring)              Docker → Docker Hub
                                                                              │
                                                                              ▼
                                                                        Render (live)
```

| Stage | File | What it does |
|---|---|---|
| Data Ingestion | `src/components/data_ingestion.py` | Validates dataset structure, reports class counts |
| Data Transformation | `src/components/data_transformation.py` | Builds augmented train/val/test generators |
| Model Training | `src/components/model_trainer.py` | Transfer learning on frozen MobileNetV2 + custom head |
| Explainability | `src/pipeline/gradcam.py` | Grad-CAM heatmap generation |
| Monitoring | `src/components/monitoring.py` | SQLite prediction logging + stats |
| Prediction Pipeline | `src/pipeline/predict_pipeline.py` | Loads saved model, predicts (single + batch) |
| Serving | `app/main.py` | FastAPI app — UI, JSON API, monitoring dashboard |

## Tech Stack

Python · TensorFlow/Keras · FastAPI · Docker · SQLite · GitHub Actions · Tailwind CSS · Three.js · Render

## Run locally

```bash
git clone https://github.com/itskunalkumar/Mechanical-Parts-Defect-Detection.git
cd Mechanical-Parts-Defect-Detection
pip install -r requirements.txt

# Download the Kaggle dataset and place it under data/casting_data/{train,test}/{def_front,ok_front}/
python -m src.pipeline.train_pipeline   # trains and saves artifacts/defect_model.h5

uvicorn app.main:app --reload
```

Visit `http://localhost:8000` for the UI, `http://localhost:8000/dashboard` for monitoring,
`http://localhost:8000/docs` for interactive API docs.

## Run with Docker

```bash
docker pull kunalbtech2024/defect-detection:latest
docker run -p 8000:8000 kunalbtech2024/defect-detection:latest
```

## API

**Single prediction** — `POST /predict`, multipart form-data, field `file`.
```json
{"label": "Defective", "confidence": 92.4}
```

**Batch prediction** — `POST /predict-batch`, multipart form-data, field `files` (multiple).
```json
{
  "results": [{"filename": "part1.jpg", "label": "OK", "confidence": 98.1}],
  "summary": {"total": 5, "defective": 1, "ok": 4}
}
```

## CI/CD

Every push to `main` triggers `.github/workflows/ci-cd.yml`:
1. Installs dependencies and runs a model smoke test (loads the model, checks output shape)
2. On success, builds the Docker image and pushes it to Docker Hub

Requires repo secrets `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN`.

## Limitations & Future Improvements

- Binary classifier only — does not output defect sub-type, exact location, or
  dimensional measurements (the UI is explicit about this; only label,
  confidence, and the Grad-CAM heatmap are real model output)
- [ ] Multi-class defect classification
- [ ] Data drift detection comparing live traffic to training distribution
- [ ] Auto-deploy to Render triggered directly from the CI/CD pipeline
