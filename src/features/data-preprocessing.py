import pandas as pd 
from pathlib import Path
import logging 

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler
import joblib

from sklearn import set_config

set_config(transform_output='pandas')


logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

logger.addHandler(handler)

formatter = logging.Formatter(fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)



num_cols = ['rider_age', 'distance', 'ratings', 'pickup_time_minutes']
nominal_cat_cols = [
    'weather', 'type_of_order', 
    'type_of_vehicle', 'festival', 
    'city_type', 'is_weekend',
    'order_time_of_day' 
]
ordinal_cat_cols = ['traffic', 'distance_type']

TARGET = 'time_taken'

traffic_order = ['low', 'medium', 'high', 'jam']
distance_type_order = ['short', 'medium', 'long', 'very_long']


def load_data(data_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        return df 
    
    except FileNotFoundError:
        logger.error('file to load does not exist')
    


def drop_missing_val(data: pd.DataFrame) -> pd.DataFrame:
    logger.info(f'The original dataset with missing values have shape {data.shape}')
    df_dropped = data.dropna()
    logger.info(f'The new dataset with no missing values have shape {df_dropped.shape}')
    missing_vals = df_dropped.isna().sum().sum()

    if missing_vals > 0:
        raise ValueError('The dataframe has missing values')
    
    return df_dropped


def save_transformer(transformer, save_dir: Path, transformer_name: str):
    save_location = save_dir / transformer_name

    joblib.dump(value=transformer, filename=save_location)

def train_preprocessor(preprocessor, data: pd.DataFrame):
    preprocessor.fit(data)
    return preprocessor

def perform_transformation(preprocessor, data: pd.DataFrame):
    transformed_data = preprocessor.transform(data)
    return transformed_data

def save_data(data: pd.DataFrame, save_path: Path): 
    data.to_csv(save_path, index=False)


def make_x_and_y(data: pd.DataFrame, target_col=TARGET):
    x = data.drop(columns=target_col)
    y = data[target_col]
    return x, y

def join_x_and_y(x: pd.DataFrame, y: pd.Series):
    joined_df = x.join(y, how='inner')
    return joined_df

if __name__ == '__main__':
    root_path = Path(__file__).parent.parent.parent
    train_data_path = root_path / 'data' / 'interim' / 'train_subset.csv'
    test_data_path = root_path / 'data' / 'interim' / 'test_subset.csv'

    save_data_dir = root_path / 'data' / 'processed'
    save_data_dir.mkdir(exist_ok=True, parents=True)

    train_trans_filename = 'train_trans.csv'
    test_trans_filename = 'test_trans.csv'

    save_train_trans_path = save_data_dir / train_trans_filename
    save_test_trans_path = save_data_dir / test_trans_filename


    preprocessor = ColumnTransformer(transformers=[
        ('scaler', MinMaxScaler(), num_cols),
        ('nomial_encode', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), nominal_cat_cols),
        ('ordinal_encode', OrdinalEncoder(categories=[traffic_order, distance_type_order], encoded_missing_value=-999, 
                                          handle_unknown='use_encoded_value', unknown_value=-1), ordinal_cat_cols)
    ], remainder='passthrough', n_jobs=-1, verbose_feature_names_out=False)

    train_df = drop_missing_val(load_data(train_data_path))
    logger.info('Train data loaded succesfully')
    test_df = drop_missing_val(load_data(test_data_path))
    logger.info('Test data loaded succesfully')

    x_train, y_train = make_x_and_y(train_df)
    x_test, y_test = make_x_and_y(test_df)
    logger.info('Data splitting completed')

    train_preprocessor(preprocessor, x_train)
    logger.info('Preprocessor is trained')

    x_train_trans = perform_transformation(preprocessor, x_train)
    logger.info('Train data is transformed')
    x_test_trans = perform_transformation(preprocessor, x_test)
    logger.info('Test data is transformed')


    train_trans_df = join_x_and_y(x_train_trans, y_train)
    test_trans_df = join_x_and_y(x_test_trans, y_test)
    logger.info('Dataset joined')

    data_subsets = [train_trans_df, test_trans_df]
    data_paths = [save_train_trans_path, save_test_trans_path]

    filename_list = [train_trans_filename, test_trans_filename]

    for filename, path, data in zip(filename_list, data_paths, data_subsets):
        save_data(data, path)
        logger.info(f'{filename.replace(".csv", "")} data saved to location')

    transformer_filename = 'preprocessor.joblib'
    transformer_save_dir = root_path / 'models'
    transformer_save_dir.mkdir(exist_ok=True, parents=True)

    save_transformer(preprocessor, transformer_save_dir, transformer_filename)

    logger.info('Preprocessor saved to location')