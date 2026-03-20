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

fomatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(fomatter)


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
    client.set_registered_model_alias(name=registered_model_name, alias='best', version=registered_model_version)

    logger.info('Alias best assigned successfully')


    
