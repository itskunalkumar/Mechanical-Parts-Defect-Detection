import base64
import io
import os
import sys

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

from src.exception import CustomException
from src.logger import logging
from src.pipeline.gradcam import GradCAM

MODEL_PATH = os.path.join("artifacts", "defect_model.h5")
IMG_HEIGHT, IMG_WIDTH = 224, 224

# class_indices from training: {'def_front': 0, 'ok_front': 1} -- confirmed via
# flow_from_directory during training
CLASS_LABELS = {0: "Defective", 1: "OK"}


class PredictPipeline:
    def __init__(self):
        self.model = None
        self.gradcam = None

    def load(self):
        if self.model is None:
            logging.info(f"Loading model from {MODEL_PATH}")
            self.model = load_model(MODEL_PATH)
            self.gradcam = GradCAM(self.model)
        return self.model

    def preprocess(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB").resize((IMG_WIDTH, IMG_HEIGHT))
        arr = np.array(image) / 255.0
        arr = np.expand_dims(arr, axis=0)  # add batch dimension
        return arr

    def _heatmap_to_base64(self, original_image: Image.Image, arr: np.ndarray) -> str:
        heatmap = self.gradcam.compute_heatmap(arr)
        overlaid = self.gradcam.overlay_heatmap(original_image, heatmap)
        buf = io.BytesIO()
        overlaid.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def predict(self, image: Image.Image, explain: bool = False) -> dict:
        try:
            model = self.load()
            arr = self.preprocess(image)
            prob = float(model.predict(arr, verbose=0)[0][0])

            predicted_class = 1 if prob >= 0.5 else 0
            confidence = prob if predicted_class == 1 else 1 - prob

            result = {
                "label": CLASS_LABELS[predicted_class],
                "confidence": round(confidence * 100, 2),
            }

            if explain:
                try:
                    result["heatmap_base64"] = self._heatmap_to_base64(image, arr)
                except Exception as gc_err:
                    # Grad-CAM failing should never break the core prediction
                    logging.info(f"Grad-CAM generation skipped: {gc_err}")

            return result
        except Exception as e:
            raise CustomException(e, sys)

    def predict_batch(self, images: list) -> list:
        """Runs prediction on a list of PIL images, returns list of results."""
        try:
            return [self.predict(img, explain=False) for img in images]
        except Exception as e:
            raise CustomException(e, sys)
