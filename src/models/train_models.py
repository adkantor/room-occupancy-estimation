import argparse
from typing import Union
import joblib
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


real_features = ['mintemp', 'maxtemp', 'minlight', 'maxlight', 'minsound', 'maxsound', 'co2', 'co2slope']


def get_data(X_path: str, y_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_path = Path(X_path).resolve()
    y_path = Path(y_path).resolve()
    # read datasets
    X = pd.read_pickle(X_path)
    y = pd.read_pickle(y_path)

    return X, y

def train_svc_binary(X_path: str, y_path: str) -> None:

    # model hyperparameters
    params = {
        'kernel': 'rbf',
        'C': 300,
        'gamma': 4.0,
        'random_state': 42,
    }
    # output path
    out_path = Path('models/binary/svc-binary.sav')

    # get datasets
    X, y = get_data(X_path, y_path)

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


def train_knn_binary(X_path: str, y_path: str) -> None:

    # model hyperparameters
    params = {
        'model__n_neighbors': 1,
        'model__weights': 'distance',
        'model__p': 1
    }
    # output path
    out_path = Path('models/binary/knn-binary.sav')

    # get datasets
    X, y = get_data(X_path, y_path)

    # constuct pipeline
    model = KNeighborsClassifier()

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


def train_cart_binary(X_path: str, y_path: str) -> None:

    # model hyperparameters
    params = {
        'model__ccp_alpha': 0.0, 
        'model__criterion': 'log_loss', 
        'model__max_depth': 12, 
        'model__min_samples_split': 4
    }
    # output path
    out_path = Path('models/binary/cart-binary.sav')

    # get datasets
    X, y = get_data(X_path, y_path)

    # constuct pipeline
    model = DecisionTreeClassifier(random_state=42)

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


def train_rf_binary(X_path: str, y_path: str) -> None:

    # model hyperparameters
    params = {
        'model__criterion': 'gini', 
        'model__max_depth': 12, 
        'model__max_features': 'sqrt', 
        'model__min_samples_split': 8, 
        'model__n_estimators': 480
    }
    # output path
    out_path = Path('models/binary/rf-binary.sav')

    # get datasets
    X, y = get_data(X_path, y_path)

    # constuct pipeline
    model = RandomForestClassifier(random_state=42)

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
    mod.add_argument("-s", "--svc", action='store_true', help="Support Vector classifier")
    mod.add_argument("-k", "--knn", action='store_true', help="k-Nearest Neighbors classifier")
    mod.add_argument("-c", "--cart", action='store_true', help="Decision Tree classifier")
    mod.add_argument("-f", "--rf", action='store_true', help="Random Forest classifier")

    args = vars(ap.parse_args())

    if args['binary']:
        if args['svc']:
            train_svc_binary(args['X'],args['y'])
        elif args['knn']:
            train_knn_binary(args['X'],args['y'])
        elif args['cart']:
            train_cart_binary(args['X'],args['y'])
        elif args['rf']:
            train_rf_binary(args['X'],args['y'])
    elif args['regression']:
        print('regression task not implemented')
