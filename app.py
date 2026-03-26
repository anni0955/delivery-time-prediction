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

mlflow.set_registry_uri('https://dagshub.com/anni0955/delivery-time-prediction.mlflow')

class Data(BaseModel):  
    # rider_age: float
    # ratings: float
    # weather: str
    # traffic: str
    # vehicle_condition: int
    # type_of_order: str
    # type_of_vehicle: str
    # multiple_deliveries: float
    # festival: str
    # city_type: str
    # is_weekend: int
    # pickup_time_minutes: float
    # order_time_of_day: str
    # distance: float
    # distance_type: str
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
        'ID': data.ID,
        'Delivery_person_ID': data.Delivery_person_ID,
        'Delivery_person_Age': data.Delivery_person_Age,
        'Delivery_person_Ratings': data.Delivery_person_Ratings,
        'Restaurant_latitude': data.Restaurant_latitude,
        'Restaurant_longitude': data.Restaurant_longitude,
        'Delivery_location_latitude': data.Delivery_location_latitude,
        'Delivery_location_longitude': data.Delivery_location_longitude,
        'Order_Date': data.Order_Date,
        'Time_Orderd': data.Time_Orderd,
        'Time_Order_picked': data.Time_Order_picked,
        'Weatherconditions': data.Weatherconditions,
        'Road_traffic_density': data.Road_traffic_density,
        'Vehicle_condition': data.Vehicle_condition, 
        'Type_of_order': data.Type_of_order,
        'Type_of_vehicle': data.Type_of_vehicle,
        'multiple_deliveries': data.multiple_deliveries,
        'Festival': data.Festival,
        'City': data.City,
        }, index=[0])


    cleaned_data = perform_data_cleanining(pred_data)
    predictions = model_pipe.predict(cleaned_data)[0]
    return predictions

if __name__ == '__main__':
    uvicorn.run(app='app:app')