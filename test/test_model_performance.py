import pytest 
import mlflow
from mlflow import MlflowClient
import dagshub
import joblib
import json 
from pathlib import Path
from sklearn.pipeline import Pipeline
import pandas as pd 
from sklearn.metrics import mean_absolute_error

dagshub.init(repo_owner='anni0955', repo_name='delivery-time-prediction', mlflow=True)

mlflow_tracking_uri = mlflow.set_tracking_uri('https://dagshub.com/anni0955/delivery-time-prediction.mlflow')

def load_model(model_path):
    with open(model_path, 'r') as f:
        run_info = json.load(f)

        return run_info
    
def load_transformer(model_path):
    with open(model_path, 'rb') as f:
        transformer = joblib.load(model_path) 
        return transformer
    
    
MODEL_NAME = 'delivery_time_prediction_model'
MODEL_ALIAS = 'dev'

model = mlflow.pyfunc.load_model(model_uri=f'models:/{MODEL_NAME}@{MODEL_ALIAS}')

root_path = Path(__file__).parent.parent

preprocessor_path = root_path / 'models' / 'preprocessor.joblib'
preprocessor = load_transformer(preprocessor_path)

model_pipe = Pipeline([
    ('transformer', preprocessor),
    ('model', model)
])


@pytest.mark.parametrize(argnames='threshold_error', argvalues=[5])
def test_model_performance(model_pipe, test_data_path, threshold_error):

    test_data_path = root_path / 'data' / 'interim' / 'test_subset.csv' 
    
    df = pd.read_csv(test_data_path)
    df.dropna(inplace=True)

    x = df.drop(columns=['time_taken'])
    y = df['time_taken']

    y_pred = model_pipe.predict(x)

    mean_error = mean_absolute_error(y, y_pred)
    assert mean_error <= threshold_error, f'the model does not pass the perfomance '
    print('avg_mae:', mean_error)

    print(f'the {MODEL_NAME}@{MODEL_ALIAS} model passed the performance test')