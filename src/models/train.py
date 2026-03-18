import pandas as pd 
import joblib 
from pathlib import Path
import logging 

from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer

from sklearn.compose import TransformedTargetRegressor
import yaml

TARGET = 'time_taken'


logger = logging.getLogger('model_training')
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

logger.addHandler(handler)

formatter = logging.Formatter(fmt= '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)



def load_data(data_path: Path) ->pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        return df 

    except FileNotFoundError:
        logger.error('the file to load does not exist')
        raise
    

def read_params(file_path: Path):
    with open(file_path, 'r') as f:
        params_file = yaml.safe_load(f)
    
    return params_file

def save_model(model, save_dir: Path, model_name: str):
    save_location = save_dir / model_name

    joblib.dump(value=model, filename=save_location)

# def save_transformer(transformer, save_dir: Path, transformer_name: str):
#     save_location = save_dir / transformer_name
    
#     joblib.dump(value=transformer, filename=save_location)
    
def make_x_and_y(data: pd.DataFrame, target_col: str):
    x = data.drop(columns=target_col)
    y = data[target_col]

    return x, y


def train_model(model, x_train: pd.DataFrame, y_train: pd.Series):
    model.fit(x_train, y_train)

    return model


if __name__ == '__main__':
    root_path = Path(__file__).parent.parent.parent
    data_path = root_path / 'data' / 'processed' / 'train_trans.csv'
    params_file_path = root_path / 'params.yaml'

    training_data = load_data(data_path)
    logger.info('Data loaded successfully')

    x_train, y_train = make_x_and_y(training_data, TARGET)
    logger.info('x_train and y_train splitting complete')

    models_params = read_params(params_file_path)['Train']
    
    rf_params = models_params['Random_Forest']
    logger.info('Read Random Forest parameters')

    rf = RandomForestRegressor(**rf_params)
    logger.info('RF Model building is successful')
    
    
    lgbm_params = models_params['LightGBM']
    logger.info('Read LGBMRegressor parameters')

    lgbm = LGBMRegressor(**lgbm_params)
    logger.info('LGBM model building is successful')

    lr = LinearRegression()
    logger.info('Build meta model')

    pt = PowerTransformer()
    logger.info('Target transformer built')

    stk_reg = StackingRegressor(
        estimators=[('rf', rf), ('lgbm', lgbm)],
        final_estimator=lr,
        cv=5, n_jobs=-1
    )
    logger.info('Stacking model built')

    model = TransformedTargetRegressor(regressor=stk_reg, transformer=pt)
    logger.info('Model wrapped inside wrapper')

    train_model(model, x_train, y_train)
    logger.info('Model training completed')

    model_filename = 'model.joblib'
    model_save_dir = root_path / 'models' 
    model_save_dir.mkdir(exist_ok=True)

    stacking_model = model.regressor_
    transformer = model.transformer_

    save_model(model, model_save_dir, model_filename)
    logger.info('Model saved succesfully')

    stacking_filename = 'stacking_regressor.joblib'
    save_model(stacking_model, model_save_dir, stacking_filename)
    logger.info('Stacking model saved successfully')

    transformer_filename = 'power_transformer.joblib'
    save_model(transformer, model_save_dir, transformer_filename)
    logger.info('power transformer saved successfully')
    



