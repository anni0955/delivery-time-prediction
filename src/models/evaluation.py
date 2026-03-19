import pandas as pd 
import joblib
from pathlib import Path
import logging 
import dagshub
import mlflow
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.compose import TransformedTargetRegressor


dagshub.init(repo_owner='anni0955',
             repo_name='delivery_time_prediction',
             mlflow=True)

mlflow.set_tracking_uri('https://dagshub.com/anni0955/delivery-time-prediction.mlflow')

mlflow.set_experiment('dvc pipeline')

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
        logger.log('File to loag dies not exist')
        raise


def make_x_and_y(data: pd.DataFrame, target_col: str):
    x = data.drop(columns=target_col)
    y = data[target_col]

    return x, y


def load_model(model_path: Path):
    model = joblib.load(model_path)
    return model


if __name__ == '__main__':
    root_path = Path(__file__).parent.parent.parent

    train_data_path = root_path / 'data' / 'processed' / 'train_trans.csv'
    test_data_path = root_path / 'data' / 'processed' / 'test_trans.csv'
    model_path = root_path / 'models' / 'model.joblib'

    train_data = load_data(train_data_path)
    logger.info('Train data loaded scuccessgully')
    
    test_data = load_data(test_data_path)
    logger.info('Test data loaded scuccessgully')

    model = load_model(model_path)
    logger.info('Model loaded scuccessgully')

    x_train, y_train = make_x_and_y(train_data, TARGET)
    x_test, y_test = make_x_and_y(test_data, TARGET)
    logger.info('Data splitting for both and training and testing set done')

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    logger.info('Prediction generated for both the train and test data')

    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    logger.info(f'The training MAE is {train_mae} and the testing MAE is {test_mae}')
    
    train_r2_score = r2_score(y_train, y_train_pred)
    test_r2_score = r2_score(y_test, y_test_pred)
    logger.info(f'The training R2_score is {train_r2_score} and the testing R2_score is {test_r2_score}')

    cv_scores = cross_val_score(model, x_train, y_train, cv=5, n_jobs=-1, scoring='neg_mean_absolute_error')
    logger.info('Cross validation completed')

    mean_cv_score = -(cv_scores).mean()

    with mlflow.start_run():
        mlflow.set_tag('model', 'Delivery Time Regressor')

        mlflow.log_params(model.get_params())

        mlflow.log_metric('train_MAE', train_mae)
        mlflow.log_metric('test_MAE', test_mae)
        mlflow.log_metric('train_r2_score', train_r2_score)
        mlflow.log_metric('test_r2_score', test_r2_score)
        mlflow.log_metric('mean_cross_val_score', mean_cv_score)

        mlflow.log_metrics({f'cv {num}': score for num, score in enumerate(-cv_scores)})

        train_data_input = mlflow.data.from_pandas(train_data, targets=TARGET)
        test_data_input = mlflow.data.from_pandas(test_data, targets=TARGET)

        mlflow.log_input(dataset=train_data_input, context='training')
        mlflow.log_input(dataset=test_data_input, context='testing')

        model_signature = mlflow.models.infer_signature(model_input=x_train.sample(20, random_state=42), 
                                                        model_output=model.predict(x_train.sample(20, random_state=42)))
        
        mlflow.sklearn.log_model(model, 'model', signature=model_signature)

        logger.info('MLflow logging complete and model logged')


