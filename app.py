from flask import Flask, request, jsonify
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route("/ping", methods= ['GET']) 
def ping() :
    return {"message": "Hi there I am availale flask app..!!"} 

@app.route('/predict', methods=['POST'])
def predict_datapoint():
        data= CustomData(
            loan_amnt = float(request.form.get('loan_amnt')),
            term = request.form.get('term'),
            int_rate = float(request.form.get('int_rate')),
            installment = float(request.form.get('installment')),
            grade = request.form.get('grade'),
            sub_grade = request.form.get('sub_grade'),
            emp_title = request.form.get('emp_title'),
            emp_length = request.form.get('emp_length'),
            home_ownership = request.form.get('home_ownership'),
            annual_inc = float(request.form.get('annual_inc')),
            verification_status = request.form.get('verification_status'),
            issue_d = request.form.get('issue_d'),
            loan_status = request.form.get('loan_status'),
            purpose = request.form.get('purpose'),
            title = request.form.get('title'),
            dti = float(request.form.get('dti')),
            earliest_cr_line = request.form.get('earliest_cr_line'),
            open_acc = float(request.form.get('open_acc')),
            pub_rec = float(request.form.get('pub_rec')),
            revol_bal = float(request.form.get('revol_bal')),
            revol_util = float(request.form.get('revol_util')),
            total_acc = float(request.form.get('total_acc')),
            initial_list_status = request.form.get('initial_list_status'),
            application_type = request.form.get('application_type'),
            mort_acc = float(request.form.get('mort_acc')),
            pub_rec_bankruptcies = float(request.form.get('pub_rec_bankruptcies')),
            address = request.form.get('address')
        )

        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(pred_df)


        if pred == 0:
            results = 'Fully Paid'
        else:
            results = 'Charged Off'

        return jsonify({'result': results})

if __name__=='__main__':
    app.run(host='127.0.0.1', port=5001, debug=True) 
