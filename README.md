# 🔧 Mechanical Parts Defect Detection (Deep Learning)

An end-to-end deep learning project that classifies casting/manufacturing parts as
**defective** or **OK** from images — using transfer learning (MobileNetV2), served
via a FastAPI REST API, and containerized with Docker.

## Overview

Manufacturing quality control traditionally relies on manual visual inspection.
This project automates defect detection from part images, combining a mechanical
engineering problem with a practical computer vision / deployment pipeline.

## Dataset

[Casting Product Image Data for Quality Inspection](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)
(Kaggle) — binary classes: `def_front` (defective) and `ok_front` (ok).

Download and place under:
```
data/casting_data/train/{def_front, ok_front}/
data/casting_data/test/{def_front, ok_front}/
```

## Pipeline

| Stage | File | What it does |
|---|---|---|
| Data Ingestion | `src/components/data_ingestion.py` | Validates dataset structure, reports class counts |
| Data Transformation | `src/components/data_transformation.py` | Builds augmented train/val/test generators |
| Model Training | `src/components/model_trainer.py` | Transfer learning on frozen MobileNetV2 + custom head |
| Prediction Pipeline | `src/pipeline/predict_pipeline.py` | Loads saved model, predicts on a single image |
| Serving | `app/main.py` | FastAPI app — HTML form (`/`) + JSON API (`/predict`) |

## Tech Stack

Python, TensorFlow/Keras, FastAPI, Docker, Pillow

## Run locally

```bash
pip install -r requirements.txt

# Train (after placing dataset under data/casting_data/)
python -m src.pipeline.train_pipeline

# Serve
uvicorn app.main:app --reload
```

Visit `http://localhost:8000`.

## Run with Docker

```bash
docker build -t defect-detection .
docker run -p 8000:8000 defect-detection
```

## API

`POST /predict` — multipart form-data with a `file` field (image).
Returns:
```json
{"label": "Defective", "confidence": 92.4}
```

## Future Improvements

- [ ] Grad-CAM visualization to show *where* the model sees the defect
- [ ] Model monitoring / drift detection
- [ ] CI/CD (GitHub Actions → Docker build → deploy)
- [ ] Multi-class defect classification (not just binary)
