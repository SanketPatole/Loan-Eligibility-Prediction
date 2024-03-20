import sys
import os
from src.logger import logging
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from src.exception import CustomException
from src.utils import save_object
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.exception import CustomException
from src.utils import load_object
from src.utils import save_object
from src.components.data_ingestion import DataIngestionConfig
from src.components.data_ingestion import DataIngestion 

@dataclass
class DataTransformConfig:
    preprocess_obj_file_path = os.path.join("artifacts", "preprocessor.pkl") #It will be a folder and file name

class DataTransformation:

    def __init__(self, config: DataTransformConfig):
        self.config = config

    def get_transformation_object(self):

        try:
            logging.info("Data Transformation Initiated")

            ## Separating Numerical features

            numerical_features = ['loan_amnt', 'term', 'int_rate', 'grade', 'sub_grade', 
                                  'emp_length', 'annual_inc', 'dti', 'open_acc', 
                                  'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'initial_list_status', 
                                  'mort_acc', 'pub_rec_bankruptcies'] 
            
            categorical_features = ['home_ownership', 'verification_status', 'purpose', 'application_type', 'zip_code']

            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())]) 
            
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))]) 
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_features),
                    ('cat', categorical_transformer, categorical_features)]) 
            return preprocessor 
        
        except Exception as e:
            raise CustomException(e, sys) 
        
    def initiate_data_transformation(self, train_path, test_path):

        try:
            logging.info("Read train set and test set is started")

            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            print(f"train_data.shape: {train_data.shape}")
            print(f"test_data.shape: {test_data.shape}")

            logging.info("Preprocessing the data")
            # Drop the columns
            logging.info("Dropping the columns")
            drop_columns = ['issue_d', 'emp_title', 'title',   
                    'earliest_cr_line']  
            train_data.drop(columns=drop_columns, axis=1, inplace=True) 
            test_data.drop(columns=drop_columns, axis=1, inplace=True)
            logging.info("Columns dropped")

            # drop the missing values from all columns
            logging.info("Dropping the missing values")
            train_data.dropna(axis=0, inplace=True)
            test_data.dropna(axis=0, inplace=True)
            logging.info("Missing values dropped")
            cols = [column for column in train_data.columns if column != 'object']
            print(cols)
            # remove the outliers from the data
            logging.info("Removing the outliers")
            # def remove_outlier(dataframe):
            #     numerical_data = dataframe.select_dtypes(include='number')
            #     for column in numerical_data.columns:
            #         q1 = dataframe[column].quantile(0.25)
            #         q3 = dataframe[column].quantile(0.75)
            #         iqr = q3 - q1
            #         lower_bound  = q1 - (1.5 * iqr)
            #         upper_bound = q3 + (1.5 * iqr)
            #         dataframe = dataframe.loc[(dataframe[column] > lower_bound) & (dataframe[column] < upper_bound)]
            #     return dataframe
            
            # train_data = remove_outlier(train_data)
            # test_data = remove_outlier(test_data)

            # Convert the rows
            logging.info("Converting the home_ownership from ANY NONE to OTHER")
            train_data.loc[(train_data['home_ownership'] == 'ANY') | (train_data['home_ownership'] == 'NONE'), 'home_ownership'] = 'OTHER'
            test_data.loc[(test_data['home_ownership'] == 'ANY') | (test_data['home_ownership'] == 'NONE'), 'home_ownership'] = 'OTHER'
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
            train_data['pub_rec'] = train_data['pub_rec'].apply(pub_rec)
            test_data['pub_rec'] = test_data['pub_rec'].apply(pub_rec)
 
            train_data['mort_acc'] = train_data['mort_acc'].apply(mort_acc)
            test_data['mort_acc'] = test_data['mort_acc'].apply(mort_acc)

            train_data['pub_rec_bankruptcies'] = train_data['pub_rec_bankruptcies'].apply(pub_rec_bankruptcies)
            test_data['pub_rec_bankruptcies'] = test_data['pub_rec_bankruptcies'].apply(pub_rec_bankruptcies)
            logging.info("Conversion succesfull of pub_rec, mort_acc, pub_rec_bankruptcies")
            print(f"train_data.shape: {train_data.shape}")
            print(f"test_data.shape: {test_data.shape}")
            # Convert the columns
            logging.info("Mapping the term, list_status, zip_code, grade, emp_length, sub_grade, loan_status")
            term_values = {' 36 months': 36, ' 60 months': 60}

            train_data['term'] = train_data['term'].map(term_values)
            test_data['term'] = test_data['term'].map(term_values)

            list_status = {'w': 0, 'f': 1}
            train_data['initial_list_status'] = train_data['initial_list_status'].map(list_status)
            train_data['initial_list_status'] = train_data['initial_list_status'].astype('int64')

            test_data['initial_list_status'] = test_data['initial_list_status'].map(list_status)
            test_data['initial_list_status'] = test_data['initial_list_status'].astype('int64')

            train_data['zip_code'] = train_data['address'].apply(lambda x: x[-5:])
            drop_columns = ['address']
            train_data.drop(columns= drop_columns, axis=1, inplace=True)

            test_data['zip_code'] = test_data['address'].apply(lambda x: x[-5:])
            test_data.drop(columns= drop_columns, axis=1, inplace=True)
            print(f"train_data.shape: {train_data.shape}")
            print(f"test_data.shape: {test_data.shape}")

            grade_to_int = {'A':7, 'B':6, 'C':5, 'D':4, 'E':3, 'F':2, 'G':1}    
            train_data['grade'] = train_data['grade'].apply(lambda x: grade_to_int[str(x).split('-')[0]])
            train_data['grade'] = train_data['grade'].astype('int64')
            test_data['grade'] = test_data['grade'].apply(lambda x: grade_to_int[str(x).split('-')[0]])
            test_data['grade'] = test_data['grade'].astype('int64')

            emp_len_to_int = {'10+ years': 10, '4 years': 4, '< 1 year': 0, '6 years': 6, '9 years': 9, '2 years': 2,
                    '3 years': 3, '8 years': 8, '7 years': 7, '5 years': 5, '1 year': 1, 'nan': np.nan}
            
            train_data['emp_length'] = train_data['emp_length'].apply(lambda x: emp_len_to_int[str(x).split('-')[0]])
            train_data['emp_length'] = train_data['emp_length'].astype('int64')
            test_data['emp_length'] = test_data['emp_length'].apply(lambda x: emp_len_to_int[str(x).split('-')[0]])
            test_data['emp_length'] = test_data['emp_length'].astype('int64')

            sub_grade_to_int = {'A1':35, 'A2':34, 'A3':33, 'A4':32, 'A5':31,
                        'B1':30, 'B2':29, 'B3':28, 'B4':27, 'B5':26,
                        'C1':25, 'C2':24, 'C3':23, 'C4':22, 'C5':21,
                        'D1':20, 'D2':19, 'D3':18, 'D4':17, 'D5':16,
                        'E1':15, 'E2':14, 'E3':13, 'E4':12, 'E5':11,
                        'F1':10, 'F2':9, 'F3':8, 'F4':7, 'F5':6,
                        'G1':5, 'G2':4, 'G3':3, 'G4':2, 'G5':1}
            
            train_data['sub_grade'] = train_data['sub_grade'].apply(lambda x: sub_grade_to_int[str(x).split('-')[0]])
            train_data['sub_grade'] = train_data['sub_grade'].astype('int64')

            test_data['sub_grade'] = test_data['sub_grade'].apply(lambda x: sub_grade_to_int[str(x).split('-')[0]])
            test_data['sub_grade'] = test_data['sub_grade'].astype('int64')

            train_data['loan_status'] = train_data['loan_status'].apply(lambda x: 0 if x == 'Fully Paid' else 1)
            train_data['loan_status'] = train_data['loan_status'].astype('int64')

            test_data['loan_status'] = test_data['loan_status'].apply(lambda x: 0 if x == 'Fully Paid' else 1)
            test_data['loan_status'] = test_data['loan_status'].astype('int64')
            print(f"train_data.shape: {train_data.shape}")
            print(f"test_data.shape: {test_data.shape}")

            logging.info("Mapping the term, list_status, zip_code, grade, emp_length, sub_grade, loan_status succesfull")

            # Write a code to remove outliers from data for all the numerical columns
            logging.info("Removing outliers from the data")
            # check data types pf all columns in the dataframe
   

            print(f"train_data.shape: {train_data.shape}")
            print(f"test_data.shape: {test_data.shape}") 
            
            logging.info("Outliers removed from the data")

    
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_transformation_object()

            logging.info("Splitting the dataset into training set and test set")
            

            target_column_name = "loan_status"
            drop_columns = target_column_name

            input_feature_train_df = train_data.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_data[target_column_name]

            input_feature_test_df=test_data.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_data[target_column_name]

            print(f"input_feature_train_df.shape: {input_feature_train_df.shape}")
            print(f"target_feature_train_df.shape: {target_feature_train_df.shape}")
            print(f"input_feature_test_df.shape: {input_feature_test_df.shape}")
            print(f"target_feature_test_df.shape: {target_feature_test_df.shape}")
            ## Transforming using preprocessor object
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr= preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)] # We are just concatenating our input train and output train
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)] # we are just concatenating our input test and output test

            save_object(

                file_path= self.config.preprocess_obj_file_path,
                obj= preprocessing_obj

            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.config.preprocess_obj_file_path,
            )

        except Exception as e:
            logging.info("Error occured in the initiate data_transformation")
            raise CustomException(e, sys) 
        



if __name__ == "__main__":
    data_transform_config = DataTransformConfig()
    data_transform = DataTransformation(data_transform_config)
    train_array, test_array, _ = data_transform.initiate_data_transformation('artifacts/train_data.csv', 'artifacts/test_data.csv')
    print(train_array.shape)
    print(test_array.shape)

        

