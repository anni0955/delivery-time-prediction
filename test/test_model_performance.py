import pytest 
import mlflow
from mlflow import MlflowClient
import dagshub
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
import pandas as pd 
from sklearn.metrics import mean_absolute_error

dagshub.init(repo_owner='anni0955', repo_name='delivery-time-prediction', mlflow=True)

mlflow_tracking_uri = mlflow.set_tracking_uri('https://dagshub.com/anni0955/delivery-time-prediction.mlflow')
    
def load_transformer(model_path):
    with open(model_path, 'rb') as f:
        transformer = joblib.load(model_path) 
        return transformer
    
    
MODEL_NAME = 'delivery_time_prediction_model'


client = MlflowClient()

versions = client.search_model_versions(f"name='{MODEL_NAME}'")
latest_version = max([int(v.version) for v in versions])

model = mlflow.pyfunc.load_model(model_uri=f'models:/{MODEL_NAME}/{latest_version}')

root_path = Path(__file__).parent.parent

preprocessor_path = root_path / 'models' / 'preprocessor.joblib'
preprocessor = load_transformer(preprocessor_path)


@pytest.mark.parametrize(argnames='threshold_error', argvalues=[5])
def test_model_performance(threshold_error):

    test_data_path = root_path / 'data' / 'interim' / 'test_subset.csv' 
    
    df = pd.read_csv(test_data_path)
    df.dropna(inplace=True)

    x = df.drop(columns=['time_taken'])
    y = df['time_taken']

    transformed_data = pd.DataFrame(preprocessor.transform(x), columns=preprocessor.get_feature_names_out())
    transformed_data['vehicle_condition'] = transformed_data['vehicle_condition'].astype(int)
    y_pred = model.predict(transformed_data)

    mean_error = mean_absolute_error(y, y_pred)
    assert mean_error <= threshold_error, f'the model does not pass the perfomance '
    print('avg_mae:', mean_error)

    print(f'the {MODEL_NAME} v{latest_version} model passed the performance test')