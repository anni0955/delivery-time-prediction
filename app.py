from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
import uvicorn
import pandas as pd 
import mlflow 
import json
import joblib
from mlflow import MlflowClient
from sklearn import set_config

set_config('pandas')

import dagshub
dagshub.init(repo_owner='anni0955', 
             repo_name='delivery-time-prediction', 
             mlflow=True)

mlflow.set_registry_uri('https://dagshub.com/anni0955/delivery-time-prediction.mlflow')

class Data(BaseModel):  
    rider_age: float
    ratings: float
    weather: str
    traffic: str
    vehicle_condition: int
    type_of_order: str
    type_of_vehicle: str
    multiple_deliveries: float
    festival: str
    city_type: str
    is_weekend: int
    pickup_time_minutes: float
    order_time_of_day: str
    distance: float
    distance_type: str

def load_model_information(file_path):
    with open(file_path, 'r') as f:
        run_info = json.load(f)

        return run_info
    

def load_transformer(transformer_path):
    transformer = joblib.load(transformer_path)
    return transformer

num_cols = [
    'rider_age', 'ratings', 'pickup_time_minutes', 'distance'
]

nomial_cat_cols = [
    'weather', 'type_of_order', 'type_of_vehicle', 'festival', 'city_type', 'is_weekend', 'order_time_of_day'
]

ordianl_cat_cols = [
    'traffic', 'distance_type'
]

client = MlflowClient()

model_name = load_model_information('run_information.json')['model_name']
stage = 'Staging'
latest_model_version = client.get_latest_versions(name=model_name, stages=[stage])

model_path = f'models:/{model_name}/{stage}'
model = mlflow.sklearn.load_model(model_path)

preprocessor_path = 'models/preprocessor.joblib'
preprocessor = load_transformer(preprocessor_path)

model_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])

app = FastAPI()

@app.get('/')
def home():
    return 'welcome to the delivery time prediction API'

@app.post('/predict')
def do_prediction(data: Data):
    pred_data = pd.DataFrame({
        'rider_age': data.rider_age, 
        'ratings': data.ratings, 
        'weather': data.weather, 
        'traffic': data.traffic,
        'vehicle_condition': data.vehicle_condition, 
        'type_of_order': data.type_of_order, 
        'type_of_vehicle': data.type_of_vehicle, 
        'multiple_deliveries': data.multiple_deliveries, 
        'festival': data.festival, 
        'city_type': data.city_type, 
        'is_weekend': data.is_weekend, 
        'pickup_time_minutes': data.pickup_time_minutes, 
        'order_time_of_day': data.order_time_of_day, 
        'distance': data.distance, 
        'distance_type': data.distance_type
    }, index=[0])

    predictions = model_pipe.predict(pred_data)[0]
    return predictions

if __name__ == '__main__':
    uvicorn.run(app='app:app')