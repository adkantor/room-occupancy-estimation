import argparse
from typing import Union
import joblib
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC


def train_svc_binary(X_path: str, y_path: str) -> None:

    # model hyperparameters
    params = {
        'kernel': 'rbf',
        'C': 300,
        'gamma': 4.0,
        'random_state': 42,
    }

    X_path = Path(X_path).resolve()
    y_path = Path(y_path).resolve()
    out_path = Path('models/svc/svc-binary.sav')

    real_features = ['mintemp', 'maxtemp', 'minlight', 'maxlight', 'minsound', 'maxsound', 'co2', 'co2slope']

    # read datasets
    X = pd.read_pickle(X_path)
    y = pd.read_pickle(y_path)

    # constuct pipeline
    model = SVC(**params)

    numTransformer = Pipeline(steps=[
        ('scaler', MinMaxScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numTransformer, real_features),
        ]
    )

    estimator = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ]
    )

    # fit model
    estimator.fit(X, y)

    # save fitted model
    joblib.dump(estimator, out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-X", "--X", required=True, help="path to pickled features dataframe")
    ap.add_argument("-y", "--y", required=True, help="path to pickled target dataframe")    
    task = ap.add_mutually_exclusive_group(required=True)
    task.add_argument("-b", "--binary", action='store_true', help="binary classification task")
    task.add_argument("-r", "--regression", action='store_true', help="regression task")
    mod = ap.add_mutually_exclusive_group(required=True)
    mod.add_argument("-s", "--svc", action='store_true', help="Support Vector Classifier")
    mod.add_argument("-d", "--dummy", action='store_true', help="Dummy classifier")

    args = vars(ap.parse_args())

    if args['binary']:
        if args['svc']:
            train_svc_binary(args['X'],args['y'])
        else:
            print('model not implemented')
    else:
        print('task not implemented')
