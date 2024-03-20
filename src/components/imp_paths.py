# for data transformation


# if __name__ == "__main__": 
#     logging.info("Data Transformation has been started")
#     data_transformation_config = DataTransformationConfig()
#     data_transformation = DataTransformation(config=data_transformation_config)
#     data_transformation.transform_data('artifacts/train_data.csv', 'artifacts/test_data.csv', 'artifacts/raw_data.csv')
#     logging.info("Data Transformation has been ended")
#     self.config.preprocessor_obj_file_path)    


# Data Engineering

# if __name__ == "__main__":
#     logging.info("Data Engineering has been started")
#     data_engineering_config = DataEngineeringConfig()
#     data_engineering = DataEngineering(config=data_engineering_config)
#     data_engineering.preprocess_data('artifacts/raw_data.csv')
#     logging.info("Data Engineering has been ended")

# Utils 
# def save_object(obj, file_path):
#     """Save the object to the file."""
#     try:
#         dir_path = os.path.dirname(file_path) 
#         os.makedirs(dir_path, exist_ok=True)
#         with open(file_path, 'wb') as f:
#             dill.dump(obj, f)
#     except Exception as e:
#         raise CustomException(e, sys) 
    
# def evaluate_model(X_train, y_train, X_test, y_test, models, model_params):
#     """Evaluate the model."""
#     try:
#         report = {}
#         for model_name in models:
#             model = models[model_name]
#             if model_params:
#                 Rs = RandomizedSearchCV(model, model_params[model_name], cv=5, n_jobs=-1)            
#             Rs.fit(X_train, y_train) 
#             model = Rs.set_params(**Rs.best_params_)
#             model.fit(X_train, y_train)

#             y_train_pred = model.predict(X_train)
#             y_test_pred = model.predict(X_test) 

#             report[model_name] = {
#                 "accuracy_score": accuracy_score(y_test, y_test_pred),
#                 "precision_score": precision_score(y_test, y_test_pred),
#                 "recall_score": recall_score(y_test, y_test_pred),
#                 "f1_score": f1_score(y_test, y_test_pred),
#                 "confusion_matrix": confusion_matrix(y_test, y_test_pred),
#                 "classification_report": classification_report(y_test, y_test_pred)
#             }
#     except Exception as e:
#         raise CustomException(e, sys) 
    
# def load_object(file_path):
#     """
#     Load the object from the file.

#     Args:
#         file_path (str): The path to the file containing the object.

#     Returns:
#         object: The loaded object.

#     Raises:
#         CustomException: If there is an error while loading the object.
#     """
#     try:
#         with open(file_path, 'rb') as f:
#             return dill.load(f)
#     except Exception as e:
#         raise CustomException(e, sys) 