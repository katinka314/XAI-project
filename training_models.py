# %%
import os
os.makedirs("src/DataScientist", exist_ok=True)
os.makedirs("src/Director", exist_ok=True)
os.makedirs("src/EndUser", exist_ok=True)

from src.data_transformation import (
    categorical_cols,
    create_train_test_split,
    load_adult_data,
    numeric_cols,
    split_X_y,
)
from src.DataScientist.plots import (
    plot_confusion_matrix,
    plot_decision_tree,
    plot_fairness,
    plot_lime_explanation,
    plot_logistic_coefficients,
    plot_shap_summary,
    plot_class_distribution,
    plot_roc_curve,
    plot_feature_distribution,
    plot_bias_report,
    plot_business_error_summary,
    plot_confusion_matrix_pct
)
from src.model_factory import fit_and_score, model_dt, model_lr

# %%
data = load_adult_data(data_path='data/adult.data', nrows=None)
X, y = split_X_y(data)
X_train, X_test, y_train, y_test = create_train_test_split(X, y, test_size=0.2, random_state=42)


dt_model = model_dt(categorical_cols, numeric_cols)
lr_model = model_lr(categorical_cols, numeric_cols)

dt_score = fit_and_score(dt_model, X_train, y_train, X_test, y_test)
lr_score = fit_and_score(lr_model, X_train, y_train, X_test, y_test)
print(f'Decision Tree Accuracy: {dt_score:.4f}')
print(f'Logistic Regression Accuracy: {lr_score:.4f}')


# %%

