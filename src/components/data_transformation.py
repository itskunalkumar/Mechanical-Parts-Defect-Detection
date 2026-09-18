import sys
from dataclasses import dataclass

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataTransformationConfig:
    img_height: int = 224
    img_width: int = 224
    batch_size: int = 32


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_data_generators(self, train_dir: str, test_dir: str):
        """
        Builds train/validation/test generators.
        Augmentation is applied ONLY on training data (defect datasets are small,
        so augmentation helps the model generalize instead of memorizing).
        """
        try:
            logging.info("Creating ImageDataGenerators with augmentation for training set")

            train_datagen = ImageDataGenerator(
                rescale=1.0 / 255,
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                brightness_range=(0.8, 1.2),
                zoom_range=0.1,
                horizontal_flip=True,
                validation_split=0.15,
            )

            test_datagen = ImageDataGenerator(rescale=1.0 / 255)

            target_size = (self.config.img_height, self.config.img_width)

            train_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=target_size,
                batch_size=self.config.batch_size,
                class_mode="binary",
                subset="training",
                shuffle=True,
            )

            val_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=target_size,
                batch_size=self.config.batch_size,
                class_mode="binary",
                subset="validation",
                shuffle=False,
            )

            test_generator = test_datagen.flow_from_directory(
                test_dir,
                target_size=target_size,
                batch_size=self.config.batch_size,
                class_mode="binary",
                shuffle=False,
            )

            logging.info(f"Class indices: {train_generator.class_indices}")

            return train_generator, val_generator, test_generator

        except Exception as e:
            raise CustomException(e, sys)
