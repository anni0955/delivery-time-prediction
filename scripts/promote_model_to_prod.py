import mlflow
import dagshub
import json 
from mlflow import MlflowClient

dagshub.init(repo_owner='anni0955', repo_name='delivery-time-prediction', mlflow=True)

mlflow.set_tracking_uri('https://dagshub.com/anni0955/delivery-time-prediction.mlflow')
    

MODEL_NAME = 'delivery_time_prediction_model'
client =MlflowClient()

versions = client.search_model_versions(f"name='{MODEL_NAME}'")
latest_version = max([int(v.version) for v in versions])

client.set_registered_model_alias(
    name=MODEL_NAME,
    alias='prod',
    version=latest_version
)

print(f'model version {latest_version} is now tagged tas @prod')
