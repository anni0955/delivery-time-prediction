import requests
import pandas as pd 
from pathlib import Path

root_dir = Path(__file__).parent.parent
data_path = root_dir / 'data' / 'raw' / 'train.csv'

predict_url = 'http://127.0.0.1:8000/predict'

sample_row = pd.read_csv(data_path).dropna().sample(1)
print(f'taget: {sample_row.iloc[:, -1].values.item().replace('(min) ', '')}')

data = sample_row.drop(columns=[sample_row.columns.tolist()[-1]]).squeeze().to_dict()
print(data)

response = requests.post(url=predict_url, json=data)

print('status code:', response.status_code)

if response.status_code == 200:
    print(f'prediction for that row is {float(response.text): .2f} min')
else:
    print('Error:', response.status_code)
