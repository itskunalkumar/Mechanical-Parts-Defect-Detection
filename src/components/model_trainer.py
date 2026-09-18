import os
import sys
from dataclasses import dataclass

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src.exception import CustomException
from src.logger import logging


@dataclass
class ModelTrainerConfig:
    trained_model_path: str = os.path.join("artifacts", "defect_model.h5")
    img_height: int = 224
    img_width: int = 224
    epochs: int = 15
    learning_rate: float = 1e-4


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def build_model(self):
        """
        Transfer learning: MobileNetV2 backbone (frozen) + custom classification head.
        Frozen backbone = fast training, works well with small defect datasets (~7k images).
        """
        base_model = MobileNetV2(
            input_shape=(self.config.img_height, self.config.img_width, 3),
            include_top=False,
            weights="imagenet",
        )
        base_model.trainable = False  # freeze pretrained layers

        model = models.Sequential(
            [
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dense(64, activation="relu"),
                layers.Dropout(0.3),
                layers.Dense(1, activation="sigmoid"),  # binary: defective vs ok
            ]
        )

        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy", "Precision", "Recall"],
        )
        return model

    def initiate_model_training(self, train_generator, val_generator):
        try:
            logging.info("Building model")
            model = self.build_model()

            os.makedirs(os.path.dirname(self.config.trained_model_path), exist_ok=True)

            callbacks = [
                EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
                ModelCheckpoint(
                    self.config.trained_model_path, monitor="val_loss", save_best_only=True
                ),
            ]

            logging.info("Starting training")
            history = model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=self.config.epochs,
                callbacks=callbacks,
            )

            logging.info(f"Model saved at {self.config.trained_model_path}")
            return history, self.config.trained_model_path

        except Exception as e:
            raise CustomException(e, sys)
