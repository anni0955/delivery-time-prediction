import pytest 
import mlflow
from mlflow import MlflowClient
import dagshub
import json

dagshub.init(repo_owner='anni0955', repo_name='delivery-time-prediction', mlflow=True)

mlflow_tracking_uri = mlflow.set_tracking_uri('https://dagshub.com/anni0955/delivery-time-prediction.mlflow')
    
MODEL_NAME = 'delivery_time_prediction_model'
MODEL_ALIAS = 'dev'


def test_load_model_from_registry():
    client = MlflowClient()

    try:
        model_version = client.get_model_version_by_alias(name=MODEL_NAME, alias=MODEL_ALIAS)

    except Exception:
        pytest.fail(f'no model found with alias @{MODEL_ALIAS}')

    version = model_version.version

    assert version is not None, 'No version linked to alias'

    model_uri = f'models/{MODEL_NAME}@{MODEL_ALIAS}'
    model = mlflow.pyfunc.load_model(model_uri)

    assert model is not None, 'Failed to load'

    print(f'Model {MODEL_NAME}@{MODEL_ALIAS} v{version} loaded successfully')

