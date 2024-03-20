# Make a trainig pipeline from data ingestion to model training.

# Path: src/pipeline/train_pipeline.py
# import os
import sys
from src.components.data_ingestion import DataIngestionConfig
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformConfig
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer
# from src.utils import load_object, save_object, evaluate_models 
from src.logger import logging
from src.exception import CustomException
# from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report)
# import pandas as pd 


if __name__ == "__main__":
    try:
        logging.info("Starting Training Pipeline")
        logging.info("Initiating Data Ingestion")
        data_ingestion_config = DataIngestionConfig()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        raw_data_path, train_data_path, test_data_path = data_ingestion.get_data() 
        logging.info("Successfully Data Ingestion Completed")
        
        logging.info("Initiating Data Transformation")
        data_transformation_config = DataTransformConfig()
        data_transformation = DataTransformation(config=data_transformation_config)
        train_array, test_array, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
        logging.info("Successfully Data Transformation Completed")
        logging.info("Initiating Model Training")
        model_trainer_config = ModelTrainerConfig()
        model_trainer = ModelTrainer(config=model_trainer_config)
        model_trainer.initiate_model_training(train_array, test_array)
        logging.info("Successfully Model Training Completed")
        
        logging.info("Training Pipeline Completed")
    except CustomException as e:
        logging.error("Error in Training Pipeline: {}".format(e))
        sys.exit(1)
    except Exception as e:
        logging.error("Error in Training Pipeline: {}".format(e))
        sys.exit(1)
