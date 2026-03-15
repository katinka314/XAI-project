from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


def build_decision_tree_model(categorical_cols, numeric_cols):
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_cols),
            (
                'cat',
                OneHotEncoder(
                    drop='first',
                    handle_unknown='infrequent_if_exist',
                    min_frequency=0.05,
                    max_categories=10,
                ),
                categorical_cols,
            ),
        ]
    )

    return Pipeline(
        [
            ('prep', preprocessor),
            ('clf', DecisionTreeClassifier(criterion='gini', min_samples_split=2, max_depth=5, random_state=42)),
        ]
    )


def build_logistic_regression_model(categorical_cols, numeric_cols):
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            (
                'cat',
                OneHotEncoder(
                    drop='first',
                    handle_unknown='infrequent_if_exist',
                    min_frequency=0.05,
                    max_categories=10,
                ),
                categorical_cols,
            ),
        ]
    )

    return Pipeline(
        [
            ('prep', preprocessor),
            ('clf', LogisticRegression(max_iter=1000)),
        ]
    )


def model_dt(categorical_cols, numeric_cols):
    return build_decision_tree_model(categorical_cols, numeric_cols)


def model_lr(categorical_cols, numeric_cols):
    return build_logistic_regression_model(categorical_cols, numeric_cols)


def fit_and_score(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)
