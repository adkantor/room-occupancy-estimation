"""Scripts to turn raw data into features for modeling."""

from pathlib import Path
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


in_path = Path('data/interim/raw_df.pkl').resolve()

def build_binary():
    """Build dataframe for binary classification with independent data points."""
    
    in_path = Path('data/interim/binary/df.pkl').resolve()
    X_path = Path('data/processed/binary/X.pkl')
    X_train_path = Path('data/processed/binary/X_train.pkl')
    X_test_path = Path('data/processed/binary/X_test.pkl')
    y_path = Path('data/processed/binary/y.pkl')
    y_train_path = Path('data/processed/binary/y_train.pkl')
    y_test_path = Path('data/processed/binary/y_test.pkl')

    df = pd.read_pickle(in_path)

    # split features and target
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    X.to_pickle(X_path)
    y.to_pickle(y_path)

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train.to_pickle(X_train_path)
    X_test.to_pickle(X_test_path)
    y_train.to_pickle(y_train_path)
    y_test.to_pickle(y_test_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    task = ap.add_mutually_exclusive_group(required=True)
    task.add_argument("-b", "--binary", action='store_true', help="binary classification task")
    task.add_argument("-r", "--regression", action='store_true', help="regression task")

    args = vars(ap.parse_args())

    if args['binary']:
        build_binary()
    else:
        print('task not implemented')
