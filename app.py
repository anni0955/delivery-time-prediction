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
from scripts.data_clean_utils import perform_data_cleanining

set_config('pandas')

import dagshub
dagshub.init(repo_owner='anni0955', 
             repo_name='delivery-time-prediction', 
             mlflow=True)

mlflow.set_tracking_uri('https://dagshub.com/anni0955/delivery-time-prediction.mlflow')

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

MODEL_NAME = 'delivery_time_prediction_model'

model_uri = f'models:/{MODEL_NAME}/latest'
model = mlflow.pyfunc.load_model(model_uri)

preprocessor_path = 'models/preprocessor.joblib'
preprocessor = load_transformer(preprocessor_path)





app = FastAPI()

class Data(BaseModel):  
    ID: str
    Delivery_person_ID: str
    Delivery_person_Age: str
    Delivery_person_Ratings: str
    Restaurant_latitude: float
    Restaurant_longitude: float
    Delivery_location_latitude: float
    Delivery_location_longitude: float
    Order_Date: str
    Time_Orderd: str
    Time_Order_picked: str
    Weatherconditions: str
    Road_traffic_density: str
    Vehicle_condition: int 
    Type_of_order: str
    Type_of_vehicle: str
    multiple_deliveries: str
    Festival: str
    City: str
    

@app.get('/')
def home():
    return 'welcome to the delivery time prediction API'

@app.post('/predict')
def do_prediction(data: Data):
    pred_data = pd.DataFrame([data.model_dump()])

    cleaned_data = perform_data_cleanining(pred_data)
    transformed_data = pd.DataFrame(preprocessor.transform(cleaned_data), columns=preprocessor.get_feature_names_out())
    transformed_data['vehicle_condition'] = transformed_data['vehicle_condition'].astype(int)

    predictions = model.predict(transformed_data)[0]
    return predictions

if __name__ == '__main__':
    uvicorn.run(app='app:app', host='0.0.0.0', port=8000)

