"""Downloads dataset from UCI and save into data/raw folder"""

import requests
from pathlib import Path


URL = r"https://archive.ics.uci.edu/ml/machine-learning-databases/00640/Occupancy_Estimation.csv"
destination = Path('data/raw/Occupancy_Estimation.csv').resolve()

response = requests.get(URL)
# open(destination, "wb").write(response.content)
destination.write_bytes(response.content)