import mlflow
import dagshub 

import logging
import json 
from mlflow import MlflowClient
from pathlib import Path

logger = logging.getLogger('register_model')
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

logger.addHandler(handler)

formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)


dagshub.init(repo_owner='anni0955', repo_name='delivery-time-prediction', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/anni0955/delivery-time-prediction.mlflow')

def load_model_information(file_path):
    with open(file_path, 'r') as f:
        run_info = json.load(f)
    
    return run_info


if __name__ == '__main__':
    root_path = Path(__file__).parent.parent.parent
    run_info_path = root_path / 'run_information.json'
    
    run_info = load_model_information(run_info_path)
    model_uri = run_info['model_uri']
    model_name = run_info['model_name']
    logger.info(f'Registering model from URI: {model_uri}')

    model_version = mlflow.register_model(model_uri=model_uri, name=model_name)

    registered_model_version = model_version.version
    registered_model_name = model_version.name
    logger.info(f'The latest mode version in the model registry is {registered_model_version}')

    client = MlflowClient()
    client.transition_model_version_stage(
        name=registered_model_name,
        version=registered_model_version,
        stage='Staging'
    )
    logger.info('Model pushed to staging stage')
