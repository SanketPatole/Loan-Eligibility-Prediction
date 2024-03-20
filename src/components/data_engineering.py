import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

import pandas as pd
import numpy as np
from sklearn.preprocessing import (StandardScaler, RobustScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split



from src.utils import save_object 
from src.utils import load_object 

from src.components.data_ingestion import DataIngestionConfig
from src.components.data_ingestion import DataIngestion



@dataclass
class DataEngineeringConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"dataengineering.pkl")

class DataEngineering:
    def __init__(self, config: DataEngineeringConfig):
        self.config = config 
        
    def preprocess_data(self, raw_data_path):

        """Preprocess the data."""
        try:
            self.df = pd.read_csv(raw_data_path)
            logging.info("Preprocessing the data")
            # Drop the columns
            logging.info("Dropping the columns")
            drop_columns = ['issue_d', 'emp_title', 'title',   
                    'earliest_cr_line']  
            self.df.drop(columns=drop_columns, axis=1, inplace=True)
            logging.info("Columns dropped")

            # drop the missing values from all columns
            logging.info("Dropping the missing values")
            self.df.dropna(axis=0, inplace=True)
            logging.info("Missing values dropped")

            # Convert the rows
            logging.info("Converting the home_ownership from ANY NONE to OTHER")
            self.df.loc[(self.df['home_ownership'] == 'ANY') | (self.df['home_ownership'] == 'NONE'), 'home_ownership'] = 'OTHER'
            # self.df['issue_d'] = pd.to_datetime(self.df['issue_d'])
            # self.df['earliest_cr_line'] = pd.to_datetime(self.df['earliest_cr_line'])

            def pub_rec(number):
                if number == 0.0:
                    return 0
                else:
                    return 1

            def mort_acc(number):
                if number == 0.0:
                    return 0
                elif number >= 1.0:
                    return 1
                else:
                    return number

            def pub_rec_bankruptcies(number):
                if number == 0.0:
                    return 0
                elif number >= 1.0:
                    return 1
                else:
                    return number

            logging.info("Converting the pub_rec, mort_acc, pub_rec_bankruptcies")
            self.df['pub_rec'] = self.df['pub_rec'].apply(pub_rec)
            self.df['mort_acc'] = self.df['mort_acc'].apply(mort_acc)
            self.df['pub_rec_bankruptcies'] = self.df['pub_rec_bankruptcies'].apply(pub_rec_bankruptcies)
            logging.info("Conversion succesfull of pub_rec, mort_acc, pub_rec_bankruptcies")

            # Convert the columns
            logging.info("Mapping the term, list_status, zip_code, grade, emp_length, sub_grade, loan_status")
            term_values = {' 36 months': 36, ' 60 months': 60}
            self.df['term'] = self.df['term'].map(term_values)
            self.df['term'] = self.df['term'].astype('int64')

            list_status = {'w': 0, 'f': 1}
            self.df['initial_list_status'] = self.df['initial_list_status'].map(list_status)
            self.df['initial_list_status'] = self.df['initial_list_status'].astype('int64')

            self.df['zip_code'] = self.df['address'].apply(lambda x: x[-5:])
            self.drop_columns = ['address']
            self.df.drop(columns=self.drop_columns, axis=1, inplace=True)

            grade_to_int = {'A':7, 'B':6, 'C':5, 'D':4, 'E':3, 'F':2, 'G':1}    
            self.df['grade'] =self.df['grade'].apply(lambda x: grade_to_int[str(x).split('-')[0]]) 
            self.df['grade'] = self.df['grade'].astype('int64')

            emp_len_to_int = {'10+ years': 10, '4 years': 4, '< 1 year': 0, '6 years': 6, '9 years': 9, '2 years': 2,
                    '3 years': 3, '8 years': 8, '7 years': 7, '5 years': 5, '1 year': 1, 'nan': np.nan}
            self.df['emp_length'] = self.df['emp_length'].apply(lambda x: emp_len_to_int[str(x).split('-')[0]]) 
            self.df['emp_length'] = self.df['emp_length'].astype('int64')

            sub_grade_to_int = {'A1':35, 'A2':34, 'A3':33, 'A4':32, 'A5':31,
                        'B1':30, 'B2':29, 'B3':28, 'B4':27, 'B5':26,
                        'C1':25, 'C2':24, 'C3':23, 'C4':22, 'C5':21,
                        'D1':20, 'D2':19, 'D3':18, 'D4':17, 'D5':16,
                        'E1':15, 'E2':14, 'E3':13, 'E4':12, 'E5':11,
                        'F1':10, 'F2':9, 'F3':8, 'F4':7, 'F5':6,
                        'G1':5, 'G2':4, 'G3':3, 'G4':2, 'G5':1}
              
            self.df['sub_grade'] = self.df['sub_grade'].apply(lambda x: sub_grade_to_int[str(x).split('-')[0]])
            self.df['sub_grade'] = self.df['sub_grade'].astype('int64')

            self.df['loan_status'] = self.df['loan_status'].apply(lambda x: 0 if x == 'Fully Paid' else 1)
            self.df['loan_status'] = self.df['loan_status'].astype('int64')

            logging.info("Mapping the term, list_status, zip_code, grade, emp_length, sub_grade, loan_status succesfull")

            # Write a code to remove outliers from data for all the numerical columns
            logging.info("Removing outliers from the data")
            # def remove_outlier(df_1 = self.df):
            #     numerical_data = df_1.select_dtypes(include='number')
            #     for column in numerical_data.columns:
            #         q1 = df_1[column].quantile(0.25)
            #         q3 = df_1[column].quantile(0.75)
            #         iqr = q3 - q1
            #         lower_bound  = q1 - (1.5 * iqr)
            #         upper_bound = q3 + (1.5 * iqr)
            #         df_1 = df_1.loc[(df_1[column] > lower_bound) & (df_1[column] < upper_bound)]
            #     return df_1    
            
            logging.info("Outliers removed from the data")

            save_object(self.config.preprocessor_obj_file_path, self.df)

            return self.df 

        except Exception as e:
            raise CustomException(e, sys) 
        

if __name__ == "__main__":
    logging.info("Data Engineering has been started")
    data_engineering_config = DataEngineeringConfig()
    data_engineering = DataEngineering(config=data_engineering_config)
    data_engineering.preprocess_data('artifacts/raw_data.csv')
    logging.info("Data Engineering has been ended")


        

        




   


