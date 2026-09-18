import io

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image

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
    result = pipeline.predict(image)
    return templates.TemplateResponse("index.html", {"request": request, "result": result})


@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    """JSON API endpoint -- this is the piece the previous project was missing."""
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    result = pipeline.predict(image)
    return result


@app.get("/health")
async def health():
    return {"status": "ok"}
