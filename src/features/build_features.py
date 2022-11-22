"""Scripts to turn raw data into features for modeling."""

from pathlib import Path
import argparse
import pandas as pd


in_path = Path('data/interim/raw_df.pkl').resolve()

def build_binary_independent():
    """Build dataframe for binary classification with independent data points."""
    
    out_path = Path('data/interim/binary_independent/df.pkl')

    df_raw = pd.read_pickle(in_path)

    # create envelopes for temp, light, sound and pir
    df = pd.DataFrame(
        index=df_raw.index,    
    )
    df['mintemp'] = df_raw[['S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp']].min(axis=1)
    df['maxtemp'] = df_raw[['S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp']].max(axis=1)
    df['minlight'] = df_raw[['S1_Light', 'S2_Light', 'S3_Light', 'S4_Light']].min(axis=1)
    df['maxlight'] = df_raw[['S1_Light', 'S2_Light', 'S3_Light', 'S4_Light']].max(axis=1)
    df['minsound'] = df_raw[['S1_Sound', 'S2_Sound', 'S3_Sound', 'S4_Sound']].min(axis=1)
    df['maxsound'] = df_raw[['S1_Sound', 'S2_Sound', 'S3_Sound', 'S4_Sound']].max(axis=1)
    df['minpir'] = df_raw[['S6_PIR', 'S7_PIR']].min(axis=1)
    df['maxpir'] = df_raw[['S6_PIR', 'S7_PIR']].max(axis=1)
    df['co2'] = df_raw['S5_CO2']
    df['co2slope'] = df_raw['S5_CO2_Slope']
    df['target'] = (df_raw['Room_Occupancy_Count'] > 0).astype(int)

    df.to_pickle(out_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    task = ap.add_mutually_exclusive_group(required=True)
    task.add_argument("-b", "--binary", action='store_true', help="binary classification task")
    task.add_argument("-r", "--regression", action='store_true', help="regression task")
    approach = ap.add_mutually_exclusive_group(required=True)
    approach.add_argument("-i", "--independent", action='store_true', help="independent samples")
    approach.add_argument("-t", "--timeseries", action='store_true', help="time series")

    args = vars(ap.parse_args())

    if args['binary'] and args['independent']:
        build_binary_independent()
    else:
        print('not implemented')