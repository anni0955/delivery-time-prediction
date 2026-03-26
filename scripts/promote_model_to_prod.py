import mlflow
import dagshub
import json 
from mlflow import MlflowClient

dagshub.init(repo_owner='anni0955', repo_name='delivery-time-prediction', mlflow=True)

mlflow.set_tracking_uri('https://dagshub.com/anni0955/delivery-time-prediction.mlflow')

def load_model_information(file_path):
    with open(file_path, 'r') as f:
        run_info = json.load(f)

        return run_info
    

model_name = load_model_information('run_information.json')['model_name']
stage = 'Staging'

client =MlflowClient()
latest_model_version = client.get_latest_versions(name=model_name, stages=[stage])
latest_model_version_staging = latest_model_version[0].version


promotion_stage = 'Production'
client.transition_model_version_stage(
    name = model_name,
    version=latest_model_version_staging,
    stage=promotion_stage,
    archive_existing_versions=True
)
