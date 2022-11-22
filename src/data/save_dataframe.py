"""Loads raw dataset into Pandas DataFrame and saves it as pkl."""

import pandas as pd
from pathlib import Path

in_path = Path('data/raw/Occupancy_Estimation.csv').resolve()
out_path = Path('data/interim/raw_df.pkl')

df = (pd.read_csv(in_path, parse_dates=[['Date', 'Time']])
        .set_index('Date_Time'))
df.to_pickle(out_path)