import pytest 
import mlflow
from mlflow import MlflowClient
import dagshub
import json

dagshub.init(repo_owner='anni0955', repo_name='delivery-time-prediction', mlflow=True)

mlflow_tracking_uri = mlflow.set_tracking_uri('https://dagshub.com/anni0955/delivery-time-prediction.mlflow')
    
MODEL_NAME = 'delivery_time_prediction_model'


def test_load_model_from_registry():
    client = MlflowClient()

    versions = client.search_model_versions(f"name='{MODEL_NAME}'")

    assert len(versions) > 0, 'No models in registry'

    latest_version = max([int(v.version) for v in versions])

    model_uri = f'models:/{MODEL_NAME}/{latest_version}'
    model = mlflow.pyfunc.load_model(model_uri)

    assert model is not None, 'failed to load model'

    print(f'model {MODEL_NAME} v{latest_version} loaded successfully')

