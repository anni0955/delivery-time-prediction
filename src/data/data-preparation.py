import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import logging
from pathlib import Path

TARGET = 'time_taken'

logger = logging.getLogger('data_preparation')
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

logger.addHandler(handler)

formatter = logging.Formatter(fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)



def load_data(datapath: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(datapath)
    
    except FileNotFoundError:
        logger.error('The file to load does not exist')

    return df 



def split_data(data: pd.DataFrame, test_size: float, random_state: int): 
    train_data, test_data = train_test_split(data, test_size=test_size, 
                                             random_state=random_state)
    
    return train_data, test_data



def read_params(file_path):
    with open(file_path, 'r') as f:
        parmas_file = yaml.safe_load(f)

    return parmas_file



def save_data(data: pd.DataFrame, save_path: Path) -> None:
    data.to_csv(save_path, index=False)



if __name__ == '__main__':
    root_path = Path(__file__).parent.parent.parent
    datapath = root_path / 'data' / 'cleaned' / 'cleaned_data.csv'
    
    save_data_dir = root_path / 'data' / 'interim'
    save_data_dir.mkdir(exist_ok=True, parents=True)
    
    train_filename = 'train_subset.csv'
    test_filename = 'test_subset.csv'

    save_train_path = save_data_dir / train_filename
    save_test_path = save_data_dir / test_filename

    params_file_path = root_path / 'params.yaml'

    df = load_data(datapath=datapath)
    logger.info('Data loaded successfully')

    parameters = read_params(params_file_path)['Data_Preparation']
    test_size = parameters['test_size']
    random_state = parameters['random_state']
    logger.info('Parameters read successfully')

    train_data, test_data = split_data(data=df, test_size=test_size, random_state=random_state)
    logger.info('Dataset split into train and test data')

    data_subsets = [train_data, test_data]
    data_paths = [save_train_path, save_test_path]
    filename_list = [train_filename, test_filename]

    for filename, path, data in zip(filename_list, data_paths, data_subsets):
        save_data(data=data, save_path=path)
        logger.info(f'{filename.replace('.csv', '')} data saved to location')
