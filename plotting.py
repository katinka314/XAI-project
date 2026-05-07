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


plot_decision_tree(dt_model, output_path='tree.png')
#plot_logistic_coefficients(lr_model, top_k=20, output_path='logreg.png', dir_path='src/DataScientist/')
#plot_logistic_coefficients(lr_model, top_k=10, output_path='logreg.png', dir_path='src/Director/')
from sklearn.metrics import confusion_matrix as _cm
_vmax = max(
    _cm(y_test, dt_model.predict(X_test)).max(),
    _cm(y_test, lr_model.predict(X_test)).max(),
)
plot_confusion_matrix(dt_model, X_test, y_test, 'Decision Tree', vmax=_vmax)
plot_confusion_matrix(lr_model, X_test, y_test, 'Logistic Regression', vmax=_vmax)
ds_pretty_names = {
    "num__education-num": "Years of Education",
    "num__capital-gain": "Capital Gain",
    "num__capital-loss": "Capital Loss",
    "num__age": "Age",
    "num__hours-pr-week": "Hours Worked Per Week",
    "num__fnlwgt": "Census Sampling Weight",
    "cat__marital-status_ Married-civ-spouse": "Married",
    "cat__relationship_ Husband": "Relationship: Husband",
    "cat__relationship_ Not-in-family": "Not in Family",
    "cat__occupation_ Prof-specialty": "Professional Occupation",
    "cat__occupation_ Exec-managerial": "Executive/Managerial",
    "cat__education_ Bachelors": "Bachelor's Degree",
    "cat__education_ Masters": "Master's Degree",
    "cat__sex_ Male": "Male",
    "cat__race_ White": "White",
}
plot_shap_summary(dt_model, X_train, 'Decision Tree', log_scale=False, pretty_names=ds_pretty_names)
plot_shap_summary(lr_model, X_train, 'Logistic Regression', log_scale=False, pretty_names=ds_pretty_names)
misclassified_dt_idx = int((dt_model.predict(X_test) != y_test.values).argmax())
plot_lime_explanation(dt_model, X_train, X_test, 'Decision Tree', instance_idx=misclassified_dt_idx)

misclassified_lr_idx = int((lr_model.predict(X_test) != y_test.values).argmax())
plot_lime_explanation(lr_model, X_train, X_test, 'Logistic Regression', instance_idx=misclassified_lr_idx)

# %%
plot_shap_summary(dt_model, X_train, 'Decision Tree', dir_path = "src/Director/", features=6, pretty_names = {
    "cat__marital-status_ Married-civ-spouse": "Married",
    "num__education-num": "Education Level",
    "num__capital-gain": "Capital Gain",
    "num__age": "Age",
    "num__hours-pr-week": "Hours Per Week",
    "num__capital-loss": "Capital Loss",},
    log_scale = False,)

plot_shap_summary(lr_model, X_train, 'Logistic Regression', dir_path = "src/Director/", features=6, pretty_names = {
    "cat__marital-status_ Married-civ-spouse": "Married",
    "num__education-num": "Education Level",
    "num__capital-gain": "Capital Gain",
    "cat__relationship_ Not-in-family": "Not in Family",
    "cat__sex_ Male": "Male",
    "num__age": "Age",})

plot_fairness(dt_model, X_test, y_test, X_test_original = X_test, category = "sex", model_name = "Decision Tree",
              dir_path = "src/Director/")
plot_fairness(lr_model, X_test, y_test, X_test_original = X_test, category = "sex", model_name = "Logistic Regression", dir_path = \
    "src/Director/")

plot_fairness(dt_model, X_test, y_test, X_test_original = X_test, category = "race", model_name = "Decision Tree",
              dir_path = "src/Director/")

plot_fairness(lr_model, X_test, y_test, X_test_original = X_test, category = "race", model_name = "Logistic Regression",
              dir_path = "src/Director/")

plot_fairness(dt_model, X_test, y_test, X_test_original = X_test, category = "sex", model_name = "Decision Tree",
              dir_path = "src/DataScientist/")
plot_fairness(lr_model, X_test, y_test, X_test_original = X_test, category = "sex", model_name = "Logistic Regression", dir_path = \
    "src/DataScientist/")

plot_fairness(dt_model, X_test, y_test, X_test_original = X_test, category = "race", model_name = "Decision Tree",
              dir_path = "src/DataScientist/")

plot_fairness(lr_model, X_test, y_test, X_test_original = X_test, category = "race", model_name = "Logistic Regression",
              dir_path = "src/DataScientist/")


rejected_idx = int((y_test.reset_index(drop=True) == ' <=50K').idxmax())

plot_lime_explanation(dt_model, X_train, X_test, 'Decision Tree', instance_idx=rejected_idx, num_features=6, dir_path=
"src/EndUser/", features_hidden = ['age', 'sex', 'race', 'native-country', 'fnlwgt',
                                   'cat__relationship_infrequent_sklearn'],
                      pretty_names = {
    'num__capital-gain': 'Investment Income',
    'cat__marital-status_ Married-civ-spouse': 'Married',
    'num__education-num': 'Education Level',
    'num__capital-loss': 'Investment Losses',
    'num__hours-pr-week': 'Hours Worked Per Week',
    'cat__education_ Masters': 'Has Masters Degree',
    'cat__occupation_ Prof-specialty': 'Professional Occupation',
    'cat__occupation_ Adm-clerical': 'Administrative Job',
    'cat__workclass_infrequent_sklearn': 'Uncommon Work Type',
    'cat__relationship_ Not-in-family': 'Not in Family',
})

plot_lime_explanation(lr_model, X_train, X_test, 'Logistic Regression', instance_idx=rejected_idx, num_features=7, dir_path =
"src/EndUser/", features_hidden = ['age', 'sex', 'race', 'native-country', 'fnlwgt',
                                   'cat__relationship_infrequent_sklearn'], pretty_names = {
    'num__capital-gain': 'Investment Income',
    'cat__marital-status_ Married-civ-spouse': 'Married',
    'num__education-num': 'Education Level',
    'num__capital-loss': 'Investment Losses',
    'num__hours-pr-week': 'Hours Worked Per Week',
    'cat__education_ Masters': 'Has Masters Degree',
    'cat__occupation_ Prof-specialty': 'Professional Occupation',
    'cat__occupation_ Adm-clerical': 'Administrative Job',
    'cat__workclass_infrequent_sklearn': 'Uncommon Work Type',
    'cat__relationship_ Not-in-family': 'Not in Family',
    'cat__education_ Some-college': 'Went to College',
})

plot_class_distribution(y_train, dir_path = "src/DataScientist")

plot_roc_curve({"Decision Tree": dt_model, "Logistic Regression": lr_model}, X_test, y_test, dir_path = "src/DataScientist/")

plot_feature_distribution(X_train=X_train, categorical=categorical_cols, numerical=numeric_cols, dir_path="src/DataScientist/",
                          numerical_pretty={
                              'age': 'Age',
                              'fnlwgt': 'Census Sampling Weight',
                              'education-num': 'Years of Education',
                              'capital-gain': 'Capital Gain',
                              'capital-loss': 'Capital Loss',
                              'hours-pr-week': 'Hours Worked Per Week',
                          })


plot_bias_report(dt_model, X_test, y_test, X_test, category='sex', model_name='Decision Tree')
plot_bias_report(lr_model, X_test, y_test, X_test, category='sex', model_name='Logistic Regression')
plot_bias_report(dt_model, X_test, y_test, X_test, category='race', model_name='Decision Tree')
plot_bias_report(lr_model, X_test, y_test, X_test, category='race', model_name='Logistic Regression')

plot_business_error_summary({"Decision Tree": dt_model, "Logistic Regression": lr_model}, X_test, y_test, dir_path = "src/Director/")
