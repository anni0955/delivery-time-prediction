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
    
    
model_name = load_model('run_information.json')['model_name']
stage = 'Staging'

model_path = f'models:/{model_name}/{stage}'
model = mlflow.sklearn.load_model(model_path)

root_path = Path(__file__).parent.parent

preprocessor_path = root_path / 'models' / 'preprocessor.joblib'
preprocessor = load_transformer(preprocessor_path)

model_pipe = Pipeline([
    ('transformer', preprocessor),
    ('model', model)
])

test_data_path = root_path / 'data' / 'interim' / 'test_subset.csv'

@pytest.mark.parametrize(argnames='model_pipe, test_data_path, threshold_error', argvalues=[(model_pipe, test_data_path, 5)])
def test_model_performance(model_pipe, test_data_path, threshold_error):
    df = pd.read_csv(test_data_path)
    df.dropna(inplace=True)

    x = df.drop(columns=['time_taken'])
    y = df['time_taken']

    y_pred = model_pipe.predict(x)

    mean_error = mean_absolute_error(y, y_pred)
    assert mean_error <= threshold_error, f'the model does not pass the perfomance '
    print('avg_mae:', mean_error)

    print(f'the {model_name} model passed the performance test')