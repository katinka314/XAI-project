import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import sklearn.metrics
from src.DataScientist.plots import plot_confusion_matrix
from src.data_transformation import (
    categorical_cols,
    numeric_cols,
    create_train_test_split,
    load_adult_data,
    split_X_y,
)
from src.model_factory import fit_and_score, model_dt, model_MLP

data = load_adult_data(data_path='data/adult.data', nrows=None)
X, y = split_X_y(data)
X_train, X_test, y_train, y_test = create_train_test_split(X, y, test_size=0.2, random_state=42)
dt_model = model_dt(categorical_cols, numeric_cols)
nn_model = model_MLP(categorical_cols, numeric_cols)

dt_score = fit_and_score(dt_model, X_train, y_train, X_test, y_test)
nn_score = fit_and_score(nn_model, X_train, y_train, X_test, y_test)

#%%
# Fix X_test once, at the top
X_test = X_test.loc[:, ~X_test.columns.duplicated()]

# Then simple, clean masks — no squeeze needed
male_mask = X_test['sex'] == " Male"
female_mask = X_test['sex'] == " Female"

X_male, y_male = X_test[male_mask].copy(), y_test[male_mask]
X_female, y_female = X_test[female_mask].copy(), y_test[female_mask]



# Confusion Matrix - Decision Tree
dtf_confusion = plot_confusion_matrix(dt_model, X_female, y_female, "Female Confusion DT", dir_path='src/DataScientist/', plot=True, vmax=None)
dtm_confusion = plot_confusion_matrix(dt_model, X_male, y_male, "Male Confusion DT", dir_path='src/DataScientist/', plot=True, vmax=None)

#Confusion Matrix - Neural Network
nnf_confusion = plot_confusion_matrix(nn_model, X_female, y_female, "Female Confusion NN", dir_path='src/DataScientist/', plot=True, vmax=None)
nnm_confusion = plot_confusion_matrix(nn_model, X_male, y_male, "Male Confusion NN", dir_path='src/DataScientist/', plot=True, vmax=None)



#%%

# Fairness Metrics - Equalized Odds
dtf_fpr = dtf_confusion["FP"] / (dtf_confusion["TN"] + dtf_confusion["FP"])  # False Positive Rate for Females
dtf_tpr = dtf_confusion["TP"] / (dtf_confusion["FN"] + dtf_confusion["TP"])  # True Positive Rate for Females
dtm_fpr = dtm_confusion["FP"] / (dtm_confusion["TN"] + dtm_confusion["FP"])  # False Positive Rate for Males
dtm_tpr = dtm_confusion["TP"] / (dtm_confusion["FN"] + dtm_confusion["TP"])  # True Positive Rate for Males

nnf_fpr = nnf_confusion["FP"] / (nnf_confusion["TN"] + nnf_confusion["FP"])  # False Positive Rate
nnf_tpr = nnf_confusion["TP"] / (nnf_confusion["FN"] + nnf_confusion["TP"])  # True Positive Rate
nnm_fpr = nnm_confusion["FP"] / (nnm_confusion["TN"] + nnm_confusion["FP"])  # False Positive Rate
nnm_tpr = nnm_confusion["TP"] / (nnm_confusion["FN"] + nnm_confusion["TP"])  # True Positive Rate

print(f"Female FPR: {dtf_fpr:.4f},  Female TPR: {dtf_tpr:.4f}")
print(f"Male FPR: {dtm_fpr:.4f},  Male TPR: {dtm_tpr:.4f}")

print(f"Female FPR: {nnf_fpr:.4f},  Female TPR: {nnf_tpr:.4f}")
print(f"Male FPR: {nnm_fpr:.4f},  Male TPR: {nnm_tpr:.4f}")





# %%
