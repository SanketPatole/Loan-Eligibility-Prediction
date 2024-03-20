from fastapi import FastAPI, Form, HTTPException

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = FastAPI() 
app = application

@app.get("/ping") 
def ping():
    return {"message": "Hi there I am availale ..!!"}


@app.post("/predict")
async def predict_datapoint(
        loan_amnt: float = Form(...),
        term: str = Form(...),
        int_rate: float = Form(...),
        installment: float = Form(...),
        grade: str = Form(...),
        sub_grade: str = Form(...),
        emp_title: str = Form(...),
        emp_length: str = Form(...),
        home_ownership: str = Form(...),
        annual_inc: float = Form(...),
        verification_status: str = Form(...),
        issue_d: str = Form(...),
        loan_status: str = Form(...),
        purpose: str = Form(...),
        title: str = Form(...),
        dti: float = Form(...),
        earliest_cr_line: str = Form(...),
        open_acc: float = Form(...),
        pub_rec: float = Form(...),
        revol_bal: float = Form(...),
        revol_util: float = Form(...),
        total_acc: float = Form(...),
        initial_list_status: str = Form(...),
        application_type: str = Form(...),
        mort_acc: float = Form(...),
        pub_rec_bankruptcies: float = Form(...),
        address: str = Form(...)):

    try:
        data = CustomData(
            loan_amnt=loan_amnt,
            term=term,
            int_rate=int_rate,
            installment=installment,
            grade=grade,
            sub_grade=sub_grade,
            emp_title=emp_title,
            emp_length=emp_length,
            home_ownership=home_ownership,
            annual_inc=annual_inc,
            verification_status=verification_status,
            issue_d=issue_d,
            loan_status=loan_status,
            purpose=purpose,
            title=title,
            dti=dti,
            earliest_cr_line=earliest_cr_line,
            open_acc=open_acc,
            pub_rec=pub_rec,
            revol_bal=revol_bal,
            revol_util=revol_util,
            total_acc=total_acc,
            initial_list_status=initial_list_status,
            application_type=application_type,
            mort_acc=mort_acc,
            pub_rec_bankruptcies=pub_rec_bankruptcies,
            address=address
        )

        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(pred_df)

        if pred == 0:
            results = 'Fully Paid'
        else:
            results = 'Charged Off'

        return {'result': results}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(application, host='0.0.0.0', port=5001, log_level='debug')
