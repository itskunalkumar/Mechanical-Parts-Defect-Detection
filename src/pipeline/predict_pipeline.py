import os
import sys

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

from src.exception import CustomException
from src.logger import logging

MODEL_PATH = os.path.join("artifacts", "defect_model.h5")
IMG_HEIGHT, IMG_WIDTH = 224, 224

# class_indices from training (0 = def_front, 1 = ok_front) -- confirm against
# train_generator.class_indices after training and update if different
CLASS_LABELS = {0: "Defective", 1: "OK"}


class PredictPipeline:
    def __init__(self):
        self.model = None

    def load(self):
        if self.model is None:
            logging.info(f"Loading model from {MODEL_PATH}")
            self.model = load_model(MODEL_PATH)
        return self.model

    def preprocess(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB").resize((IMG_WIDTH, IMG_HEIGHT))
        arr = np.array(image) / 255.0
        arr = np.expand_dims(arr, axis=0)  # add batch dimension
        return arr

    def predict(self, image: Image.Image) -> dict:
        try:
            model = self.load()
            arr = self.preprocess(image)
            prob = float(model.predict(arr)[0][0])

            predicted_class = 1 if prob >= 0.5 else 0
            confidence = prob if predicted_class == 1 else 1 - prob

            return {
                "label": CLASS_LABELS[predicted_class],
                "confidence": round(confidence * 100, 2),
            }
        except Exception as e:
            raise CustomException(e, sys)
