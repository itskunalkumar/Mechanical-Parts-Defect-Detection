import io

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image

from src.components.monitoring import log_prediction, get_stats
from src.pipeline.predict_pipeline import PredictPipeline

app = FastAPI(title="Mechanical Parts Defect Detection")
templates = Jinja2Templates(directory="app/templates")
pipeline = PredictPipeline()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})


@app.post("/predict-ui", response_class=HTMLResponse)
async def predict_ui(request: Request, file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    result = pipeline.predict(image, explain=True)  # Grad-CAM heatmap included in UI
    log_prediction(result["label"], result["confidence"])
    return templates.TemplateResponse("index.html", {"request": request, "result": result})


@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    """JSON API endpoint for a single image (no heatmap, lightweight)."""
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    result = pipeline.predict(image, explain=False)
    log_prediction(result["label"], result["confidence"])
    return result


@app.post("/predict-explain")
async def predict_explain_api(file: UploadFile = File(...)):
    """JSON API endpoint used by the interactive UI -- includes Grad-CAM heatmap."""
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    result = pipeline.predict(image, explain=True)
    log_prediction(result["label"], result["confidence"])
    return result


@app.post("/predict-batch")
async def predict_batch_api(files: list[UploadFile] = File(...)):
    """Batch JSON API endpoint -- accepts multiple images in one request."""
    images = []
    filenames = []
    for f in files:
        image_bytes = await f.read()
        images.append(Image.open(io.BytesIO(image_bytes)))
        filenames.append(f.filename)

    results = pipeline.predict_batch(images)
    for r in results:
        log_prediction(r["label"], r["confidence"])

    defective_count = sum(1 for r in results if r["label"] == "Defective")

    return {
        "results": [
            {"filename": fn, **r} for fn, r in zip(filenames, results)
        ],
        "summary": {
            "total": len(results),
            "defective": defective_count,
            "ok": len(results) - defective_count,
        },
    }


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Simple monitoring dashboard -- prediction volume, class balance, low-confidence flags."""
    stats = get_stats()
    return templates.TemplateResponse("dashboard.html", {"request": request, "stats": stats})


@app.get("/health")
async def health():
    return {"status": "ok"}
