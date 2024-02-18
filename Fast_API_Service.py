from fastapi import FastAPI
import hydra
import joblib
from pydantic import BaseModel
import uvicorn
import pickle
import numpy as np
import json
class Item(BaseModel):
    JoiningYear: int
    PaymentTier: int
    Age: int
    ExperienceInCurrentDomain: int
    Education_Masters: int
    Education_PHD: int
    City_New_Delhi: int
    City_Pune: int
    Gender_Male: int
    EverBenched_Yes: int
   
app = FastAPI()
@app.post('/predict')
def predict_outcome(data: Item):
    data = data.model_dump()
    model = joblib.load('F:\\Machine_Learning_Ops\\mlops_1\\models\\RandomForest')
    Joining_Year = data['JoiningYear']
    PaymentTier = data['PaymentTier']
    Age = data['Age']
    ExperienceInCurrentDomain = data['ExperienceInCurrentDomain']
    Education_Masters = data['Education_Masters']
    Education_PHD = data['Education_PHD']
    City_New_Delhi = data['City_New_Delhi']
    City_Pune = data['City_Pune']
    Gender_Male = data['Gender_Male']
    EverBenched_Yes = data['EverBenched_Yes']
    features = [Joining_Year,PaymentTier,Age,ExperienceInCurrentDomain,Education_Masters,Education_PHD,City_New_Delhi,
                City_Pune,Gender_Male,EverBenched_Yes]
    predictions = model.predict([features])
    response_data = {"prediction": predictions.tolist()}
    return json.dumps(response_data)



if __name__ == '__main__':
    uvicorn.run(app,host='127.0.0.1',port = 8000)


    
