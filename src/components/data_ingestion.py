import os
import sys
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    # Expected structure after downloading the Kaggle "Casting Product" dataset:
    # data/casting_data/train/{ok_front, def_front}/*.jpeg
    # data/casting_data/test/{ok_front, def_front}/*.jpeg
    train_dir: str = os.path.join("data", "casting_data", "train")
    test_dir: str = os.path.join("data", "casting_data", "test")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion")
        try:
            train_dir = self.ingestion_config.train_dir
            test_dir = self.ingestion_config.test_dir

            if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
                raise FileNotFoundError(
                    f"Expected dataset at '{train_dir}' and '{test_dir}'. "
                    "Download from Kaggle: 'Casting Product Image Data for Quality Inspection' "
                    "and place it under data/casting_data/"
                )

            classes = sorted(os.listdir(train_dir))
            logging.info(f"Found classes: {classes}")

            counts = {}
            for cls in classes:
                cls_path = os.path.join(train_dir, cls)
                counts[cls] = len(os.listdir(cls_path))
            logging.info(f"Training image counts per class: {counts}")

            return train_dir, test_dir, classes

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    ingestion = DataIngestion()
    train_dir, test_dir, classes = ingestion.initiate_data_ingestion()
    print(f"Train dir: {train_dir}")
    print(f"Test dir: {test_dir}")
    print(f"Classes: {classes}")
