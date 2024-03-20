import sys
print(sys.path)
import os
from logger import logging
# from logger import logging
from src.exception import CustomException 
import pandas as pd
from src.utils import load_object
from src.utils import save_object
from sklearn.model_selection import train_test_split
from dataclasses import dataclass 

# If you want to do data transformation then in same way you can create a 
#Data TransfornationConfig class and write the code for transformation

#Here my DataIngestionConfig class know where to store the train and test data and raw data 
# using dataclass decorator we can directly assign the variables to the class 
# using dataclass decorator we can access the variables using self.config.variable_name 
@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts', 'train_data.csv')
    test_data_path: str=os.path.join('artifacts', 'test_data.csv')
    raw_data_path: str=os.path.join('artifacts', 'raw_data.csv')

# If you are just defining the variables then you can go with dataclass decorator 
# otherwise if you want to define some function and operations then you can go with class
class DataIngestion: 
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def get_data(self):
        logging.info("Reading data from the source")
        try:
            data = pd.read_csv('notebook/data/loan_data.csv')
            logging.info("Data is read from the source")
            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True) 
            data.to_csv(self.config.raw_data_path, index=False, header=True)
            logging.info("Raw data is stored in the path: {}".format(self.config.raw_data_path))

            train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
            train_set.to_csv(self.config.train_data_path, index=False, header=True)
            test_set.to_csv(self.config.test_data_path, index=False, header=True)

            logging.info("Train and Test data is stored in the path: {} and {}".format(self.config.train_data_path, self.config.test_data_path))
            return (self.config.raw_data_path, self.config.train_data_path, self.config.test_data_path)
        except Exception as e:
            logging.error("Error while reading data from the source")
            raise CustomException(e)
        
    
if __name__ == "__main__":
    data_ingestion_config = DataIngestionConfig()
    data_ingestion = DataIngestion(config=data_ingestion_config)
    data_ingestion.get_data() 


               
    