import sys

from tensorflow.keras.models import load_model

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging


def run_training():
    try:
        logging.info("=== Training pipeline started ===")

        train_dir, test_dir, classes = DataIngestion().initiate_data_ingestion()

        train_gen, val_gen, test_gen = DataTransformation().get_data_generators(
            train_dir, test_dir
        )

        trainer = ModelTrainer()
        history, model_path = trainer.initiate_model_training(train_gen, val_gen)

        logging.info("Evaluating best saved model on test set")
        best_model = load_model(model_path)
        test_loss, test_acc, test_precision, test_recall = best_model.evaluate(test_gen)
        logging.info(
            f"Test loss={test_loss:.4f} acc={test_acc:.4f} "
            f"precision={test_precision:.4f} recall={test_recall:.4f}"
        )

        print(f"Model trained and saved to: {model_path}")
        print(
            f"Test accuracy: {test_acc:.4f} | precision: {test_precision:.4f} | "
            f"recall: {test_recall:.4f}"
        )
        print(f"Classes: {classes}")

        logging.info("=== Training pipeline finished ===")

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_training()
