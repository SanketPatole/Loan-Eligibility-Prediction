import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
                            RandomForestClassifier, 
                            AdaBoostClassifier,
                            GradientBoostingClassifier
                            )

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report)

from src.components.data_transformation import DataTransformConfig 
from src.components.data_transformation import DataTransformation
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object, save_object, evaluate_models 


# @dataclass
# class ModelTrainerConfig:
#     model_obj_file_path=os.path.join('artifacts',"model.pkl")

# class ModelTrainer: 
#     def __init__(self, config: ModelTrainerConfig):
#         self.config = config

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', "model.pkl")
    model_performance_file_path = os.path.join('artifacts', "model_performance.csv")

class ModelTrainer:
    
    def __init__(self, config: ModelTrainerConfig): 
        self.config = config

    def initiate_model_training(self, train_array, test_array):
         
         try:
            logging.info("Splitting dependent and independent variables from train and test data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            logging.info("Successfully dependent and independent splitted in training and testset")
            print(set(y_train))
            print(set(y_test))

            logging.info("Successfully dependent and independent splitted in training and testset")

            logging.info("Ready to do model")

            models = {
                        "Logistic Regression": LogisticRegression(),
                        "Random Forest": RandomForestClassifier(class_weight='balanced')
                    }       
            
            params = {"Logistic Regression": {"solver": ['liblinear'], "C": [0.1, 1], "penalty": ["l1", "l2"]},
                      "Random Forest": {"n_estimators": [10, 50], "max_depth": [9, 10]}
                      }

            logging.info("Successfully Model was fitted")

            logging.info("Evaluation Model")

            data_report, train_model_report, test_model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)
            print('\n====================================================================================\n')
            logging.info(f'Data Report : {data_report}') 
            logging.info(f'Test Model Report : {test_model_report}')
            logging.info(f'Train Model Report : {train_model_report}')

            ## To get the best model score from dictionary
            # best_model_score = max(sorted(model_report.values()))
            best_model_score = max(test_model_report.items(), key=lambda x: x[1]['test_f1'])
            
            best_model_name = max(test_model_report.items(), key=lambda x: x[1]['test_f1'])[0]
 
            # best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , F1_Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , F1_Score : {best_model_score}')

            data_report.to_csv(self.config.model_performance_file_path, index=False, header=True)

            save_object(
                    obj=best_model,
                    file_path=self.config.trained_model_file_path             
            )

        

         except Exception as e:
            logging.info("Exception occured at model training")
            raise CustomException(e, sys)



if __name__ == "__main__":
    data_transform_config = DataTransformConfig()
    data_transform = DataTransformation(data_transform_config)
    train_array, test_array, _ = data_transform.initiate_data_transformation('artifacts/train_data.csv', 'artifacts/test_data.csv')
    model_trainer_config = ModelTrainerConfig()
    model_trainer = ModelTrainer(model_trainer_config)
    model_trainer.initiate_model_training(train_array, test_array)