import sys
import pandas as pd

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self) -> None:
        pass
    
    def predict(self,new_data):
        try:
            preprocessor_path='artifacts\dataengineering.pkl'
            transform_path='artifacts\preprocessor.pkl'
            model_path='artifacts\model.pkl'

            data_preprocessor=load_object(file_path=preprocessor_path)
            data_transform=load_object(file_path=transform_path)
            data_model = load_object(file_path=model_path)
            
            preprocessed_data = data_preprocessor.preprocess_data(new_data)
            transformed_data = data_transform.transform(preprocessed_data)
            predictions = data_model.predict(transformed_data)

            preds = pd.DataFrame(predictions, columns=['predictions'])
            # print(preds)
            return predictions
        except Exception as e:
            raise CustomException(e,sys)
         

class CustomData:
    '''
    Responsible for mapping all the data points we receive in the front end to the back end.
    '''

    def __init__(self,
                 loan_amnt: float, 
                 term: str, 
                 int_rate: float, 
                 installment: float, 
                    grade: str, 
                    sub_grade: str, 
                    emp_title: str, 
                    emp_length: str, 
                    home_ownership: str, 
                    annual_inc: float, 
                    verification_status: str, 
                    issue_d: str, 
                    loan_status: str, 
                    purpose: str, 
                    title: str, 
                    dti: float, 
                    earliest_cr_line: str, 
                    open_acc: float, 
                    pub_rec: float, 
                    revol_bal: float,  
                    revol_util: float, 
                    total_acc: float, 
                    initial_list_status: str, 
                    application_type: str, 
                    mort_acc: float, 
                    pub_rec_bankruptcies: float, 
                    address: str) -> None: 
        self.loan_amnt = loan_amnt
        self.term = term
        self.int_rate = int_rate
        self.installment = installment
        self.grade = grade
        self.sub_grade = sub_grade
        self.emp_title = emp_title
        self.emp_length = emp_length
        self.home_ownership = home_ownership
        self.annual_inc = annual_inc
        self.verification_status = verification_status
        self.issue_d = issue_d
        self.loan_status = loan_status
        self.purpose = purpose
        self.title = title
        self.dti = dti
        self.earliest_cr_line = earliest_cr_line
        self.open_acc = open_acc
        self.pub_rec = pub_rec
        self.revol_bal = revol_bal
        self.revol_util = revol_util
        self.total_acc = total_acc
        self.initial_list_status = initial_list_status
        self.application_type = application_type
        self.mort_acc = mort_acc
        self.pub_rec_bankruptcies = pub_rec_bankruptcies
        self.address = address


    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                'term': [self.term],
                'int_rate': [self.int_rate],
                'installment': [self.installment],
                'grade': [self.grade],
                'sub_grade': [self.sub_grade],
                'emp_title': [self.emp_title],
                'emp_length': [self.emp_length],
                'home_ownership': [self.home_ownership],
                'annual_inc': [self.annual_inc],
                'verification_status': [self.verification_status],
                'issue_d': [self.issue_d],
                'loan_status': [self.loan_status],
                'purpose': [self.purpose],
                'title': [self.title],
                'dti': [self.dti],
                'earliest_cr_line': [self.earliest_cr_line],
                'open_acc': [self.open_acc],
                'pub_rec': [self.pub_rec],
                'revol_bal': [self.revol_bal],
                'revol_util': [self.revol_util],
                'total_acc': [self.total_acc],
                'initial_list_status': [self.initial_list_status],
                'application_type': [self.application_type],
                'mort_acc': [self.mort_acc],
                'pub_rec_bankruptcies': [self.pub_rec_bankruptcies],
                'address': [self.address]

            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
        
