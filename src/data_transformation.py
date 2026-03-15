from sklearn.model_selection import train_test_split

import pandas as pd


column_names = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital-gain',
    'capital-loss',
    'hours-pr-week',
    'native-country',
    'income',
]

categorical_cols = [
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country',
]

numeric_cols = [
    'age',
    'fnlwgt',
    'education-num',
    'capital-gain',
    'capital-loss',
    'hours-pr-week',
]


def load_adult_data(data_path='data/adult.data', nrows=1000, names = column_names):
    return pd.read_csv(data_path, nrows=nrows, header=None, names=column_names)


def split_X_y(data, target_col='income'):
    X = data.drop(columns=[target_col])
    y = data[target_col]
    return X, y


def create_train_test_split(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

