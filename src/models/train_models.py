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
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier


real_features_binary = [
    'mintemp', 'maxtemp', 'minlight', 'maxlight', 'minsound', 'maxsound',
    'co2', 'co2slope'
]
real_features_multiclass = [
    'S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp',
    'S1_Light', 'S2_Light', 'S3_Light', 'S4_Light',
    'S1_Sound', 'S2_Sound', 'S3_Sound', 'S4_Sound',
    'S5_CO2', 'S5_CO2_Slope',
    'mintemp', 'maxtemp', 'minlight', 'maxlight', 'minsound', 'maxsound'
]


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
            ('numeric', numTransformer, real_features_binary),
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


def train_svc_multiclass(X_path: str, y_path: str) -> None:

    # model hyperparameters
    params = {
        'kernel': 'rbf',
        'C': 45,
        'gamma': 3.7,
        'random_state': 42,
    }
    # output path
    out_path = Path('models/multiclass/svc-multiclass.sav')

    # get datasets
    X, y = get_data(X_path, y_path)

    # constuct pipeline
    model = SVC(**params)

    numTransformer = Pipeline(steps=[
        ('scaler', MinMaxScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numTransformer, real_features_multiclass),
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
        'n_neighbors': 1,
        'weights': 'distance',
        'p': 1
    }
    # output path
    out_path = Path('models/binary/knn-binary.sav')

    # get datasets
    X, y = get_data(X_path, y_path)

    # constuct pipeline
    model = KNeighborsClassifier(**params)

    numTransformer = Pipeline(steps=[
        ('scaler', MinMaxScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numTransformer, real_features_binary),
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


def train_knn_multiclass(X_path: str, y_path: str) -> None:

    # model hyperparameters
    params = {
        'n_neighbors': 1,
        'weights': 'distance',
        'p': 1
    }
    # output path
    out_path = Path('models/multiclass/knn-multiclass.sav')

    # get datasets
    X, y = get_data(X_path, y_path)

    # constuct pipeline
    model = KNeighborsClassifier(**params)

    numTransformer = Pipeline(steps=[
        ('scaler', MinMaxScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numTransformer, real_features_multiclass),
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
        'ccp_alpha': 0.0, 
        'criterion': 'log_loss', 
        'max_depth': 12, 
        'min_samples_split': 4,
        'random_state': 42,
    }
    # output path
    out_path = Path('models/binary/cart-binary.sav')

    # get datasets
    X, y = get_data(X_path, y_path)

    # constuct pipeline
    model = DecisionTreeClassifier(**params)

    numTransformer = Pipeline(steps=[
        ('scaler', MinMaxScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numTransformer, real_features_binary),
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


def train_cart_multiclass(X_path: str, y_path: str) -> None:

    # model hyperparameters
    params = {
        'ccp_alpha': 0.0, 
        'criterion': 'log_loss', 
        'max_depth': 16, 
        'min_samples_split': 2,
        'random_state': 42,
    }
    # output path
    out_path = Path('models/multiclass/cart-multiclass.sav')

    # get datasets
    X, y = get_data(X_path, y_path)

    # constuct pipeline
    model = DecisionTreeClassifier(**params)

    numTransformer = Pipeline(steps=[
        ('scaler', MinMaxScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numTransformer, real_features_multiclass),
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
        'criterion': 'gini', 
        'max_depth': 12, 
        'max_features': 'sqrt', 
        'min_samples_split': 8, 
        'n_estimators': 480
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
            ('numeric', numTransformer, real_features_binary),
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


def train_rf_multiclass(X_path: str, y_path: str) -> None:

    # model hyperparameters
    params = {
        'criterion': 'entropy', 
        'max_depth': 21, 
        'max_features': 'sqrt', 
        'min_samples_split': 2, 
        'n_estimators': 500
    }
    # output path
    out_path = Path('models/multiclass/rf-multiclass.sav')

    # get datasets
    X, y = get_data(X_path, y_path)

    # constuct pipeline
    model = RandomForestClassifier(random_state=42)

    numTransformer = Pipeline(steps=[
        ('scaler', MinMaxScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numTransformer, real_features_multiclass),
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


def train_adaboost_binary(X_path: str, y_path: str) -> None:

    # model hyperparameters
    params = {
        'learning_rate': 1.486,
        'n_estimators': 500,
        'random_state': 42
    }
    # output path
    out_path = Path('models/binary/adaboost-binary.sav')

    # get datasets
    X, y = get_data(X_path, y_path)

    # constuct pipeline
    model = AdaBoostClassifier(**params)

    numTransformer = Pipeline(steps=[
        ('scaler', MinMaxScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numTransformer, real_features_binary),
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


def train_gradientboost_multiclass(X_path: str, y_path: str) -> None:

    # model hyperparameters
    params = {
        'learning_rate': 0.1,
        'max_depth': 11,
        'max_features': 'sqrt',
        'subsample': 1,
        'n_estimators': 500,
        'n_iter_no_change': 10,
        'validation_fraction': 0.2,
        'random_state':42,
    }
    # output path
    out_path = Path('models/multiclass/gradientboost-multiclass.sav')

    # get datasets
    X, y = get_data(X_path, y_path)

    # constuct pipeline
    model = GradientBoostingClassifier(**params)

    numTransformer = Pipeline(steps=[
        ('scaler', MinMaxScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numTransformer, real_features_multiclass),
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
    task.add_argument("-m", "--multiclass", action='store_true', help="multiclass classification task")
    mod = ap.add_mutually_exclusive_group(required=True)
    mod.add_argument("-s", "--svc", action='store_true', help="Support Vector classifier")
    mod.add_argument("-k", "--knn", action='store_true', help="k-Nearest Neighbors classifier")
    mod.add_argument("-c", "--cart", action='store_true', help="Decision Tree classifier")
    mod.add_argument("-f", "--rf", action='store_true', help="Random Forest classifier")
    mod.add_argument("-a", "--adaboost", action='store_true', help="AdaBoost classifier")
    mod.add_argument("-g", "--gradientboost", action='store_true', help="Gradient Boosting classifier")

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
        elif args['adaboost']:
            train_adaboost_binary(args['X'],args['y'])
        elif args['gradientboost']:
            print('Gradient Boosting for binary classification not implemented')
    elif args['multiclass']:
        if args['svc']:
            train_svc_multiclass(args['X'],args['y'])
        elif args['knn']:
            train_knn_multiclass(args['X'],args['y'])
        elif args['cart']:
            train_cart_multiclass(args['X'],args['y'])
        elif args['rf']:
            train_rf_multiclass(args['X'],args['y'])
        elif args['adaboost']:
            print('AdaBoost for multiclass classification not implemented')
        elif args['gradientboost']:
            train_gradientboost_multiclass(args['X'],args['y'])
    else:
        print('task not implemented')
