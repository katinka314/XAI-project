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

#%%
dt_model = model_dt(categorical_cols, numeric_cols)
nn_model = model_MLP(categorical_cols, numeric_cols)

dt_score = fit_and_score(dt_model, X_train, y_train, X_test, y_test)
nn_score = fit_and_score(nn_model, X_train, y_train, X_test, y_test)

#%%

# masks
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

# Fairness Metrics - Equalized Odds -- Sex 
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


white_mask = X_test['race'] == " White"
black_mask = X_test['race'] == " Black"
other_mask = X_test['race'] == " Other"

X_white, y_white = X_test[white_mask].copy(), y_test[white_mask]
X_black, y_black = X_test[black_mask].copy(), y_test[black_mask]
X_other, y_other = X_test[other_mask].copy(), y_test[other_mask]

# Confusion Matrix - Decision Tree
dtw_confusion = plot_confusion_matrix(dt_model, X_white, y_white, "White Confusion DT", dir_path='src/DataScientist/', plot=True, vmax=None)
dtb_confusion = plot_confusion_matrix(dt_model, X_black, y_black, "Black Confusion DT", dir_path='src/DataScientist/', plot=True, vmax=None)
dto_confusion = plot_confusion_matrix(dt_model, X_other, y_other, "Other    Confusion DT", dir_path='src/DataScientist/', plot=True, vmax=None) 
nnw_confusion = plot_confusion_matrix(nn_model, X_white, y_white, "White Confusion NN", dir_path='src/DataScientist/', plot=True, vmax=None)
nnb_confusion = plot_confusion_matrix(nn_model, X_black, y_black, "Black Confusion NN", dir_path='src/DataScientist/', plot=True, vmax=None)
nno_confusion = plot_confusion_matrix(nn_model, X_other, y_other, "Other    Confusion NN", dir_path='src/DataScientist/', plot=True, vmax=None)

dtw_fpr = dtw_confusion["FP"] / (dtw_confusion["TN"] + dtw_confusion["FP"])  # False Positive Rate for White
dtw_tpr = dtw_confusion["TP"] / (dtw_confusion["FN"] + dtw_confusion["TP"])  # True Positive Rate for White
dtb_fpr = dtb_confusion["FP"] / (dtb_confusion["TN"] + dtb_confusion["FP"])  # False Positive Rate for Black
dtb_tpr = dtb_confusion["TP"] / (dtb_confusion["FN"] + dtb_confusion["TP"])  # True Positive Rate for Black
dto_fpr = dto_confusion["FP"] / (dto_confusion["TN"] + dto_confusion["FP"])  # False Positive Rate for Other
dto_tpr = dto_confusion["TP"] / (dto_confusion["FN"] + dto_confusion["TP"])  # True Positive Rate for Other

nnw_fpr = nnw_confusion["FP"] / (nnw_confusion["TN"] + nnw_confusion["FP"])  # False Positive Rate for White
nnw_tpr = nnw_confusion["TP"] / (nnw_confusion["FN"] + nnw_confusion["TP"])  # True Positive Rate for White
nnb_fpr = nnb_confusion["FP"] / (nnb_confusion["TN"] + nnb_confusion["FP"])  # False Positive Rate for Black
nnb_tpr = nnb_confusion["TP"] / (nnb_confusion["FN"] + nnb_confusion["TP"])  # True Positive Rate for Black
nno_fpr = nno_confusion["FP"] / (nno_confusion["TN"] + nno_confusion["FP"])  # False Positive Rate for Other
nno_tpr = nno_confusion["TP"] / (nno_confusion["FN"] + nno_confusion["TP"])  # True Positive Rate for Other

#%%

X_sex_train, X_sex_test = X_train.drop(columns=['sex', 'race']), X_test.drop(columns=['sex', 'race'])
categorical_cols_sex = [col for col in categorical_cols if col != 'sex' and col != 'race']

X_female, X_male = X_female.drop(columns=['sex', 'race']), X_male.drop(columns=['sex', 'race'])
X_white, X_black, X_other = X_white.drop(columns=['sex', 'race']), X_black.drop(columns=['sex', 'race']), X_other.drop(columns=['sex', 'race'])

# %%

dt_model_fair = model_dt(categorical_cols_sex, numeric_cols)
nn_model_fair = model_MLP(categorical_cols_sex, numeric_cols)

dt_score_fair = fit_and_score(model=dt_model_fair, X_train=X_sex_train, y_train=y_train, X_test=X_sex_test, y_test=y_test)
nn_score_fair = fit_and_score(model=nn_model_fair, X_train=X_sex_train, y_train=y_train, X_test=X_sex_test, y_test=y_test)

#plot confusion 
dt_confusion_fair = plot_confusion_matrix(dt_model_fair, X_sex_test, y_test, "Confusion Matrix DT Fair", dir_path='src/DataScientist/', plot=True, vmax=None)
nn_confusion_fair = plot_confusion_matrix(nn_model_fair, X_sex_test, y_test, "Confusion Matrix NN Fair", dir_path='src/DataScientist/', plot=True, vmax=None)

# Gender analyisis
dtf_confusion_fair = plot_confusion_matrix(dt_model_fair, X_female, y_female, "Female Confusion DT Fair", dir_path='src/DataScientist/', plot=True, vmax=None)
dtm_confusion_fair = plot_confusion_matrix(dt_model_fair, X_male, y_male, "Male Confusion DT Fair", dir_path='src/DataScientist/', plot=True, vmax=None)
nnf_confusion_fair = plot_confusion_matrix(nn_model_fair, X_female, y_female, "Female Confusion NN Fair", dir_path='src/DataScientist/', plot=True, vmax=None)
nnm_confusion_fair = plot_confusion_matrix(nn_model_fair, X_male, y_male, "Male Confusion NN Fair", dir_path='src/DataScientist/', plot=True, vmax=None)

# Race analysis
dtw_confusion_fair = plot_confusion_matrix(dt_model_fair, X_white, y_white, "White Confusion DT Fair", dir_path='src/DataScientist/', plot=True, vmax=None)
dtb_confusion_fair = plot_confusion_matrix(dt_model_fair, X_black, y_black, "Black Confusion DT Fair", dir_path='src/DataScientist/', plot=True, vmax=None)
dto_confusion_fair = plot_confusion_matrix(dt_model_fair, X_other, y_other, "Other Confusion DT Fair", dir_path='src/DataScientist/', plot=True, vmax=None)
nnw_confusion_fair = plot_confusion_matrix(nn_model_fair, X_white, y_white, "White Confusion NN Fair", dir_path='src/DataScientist/', plot=True, vmax=None)
nnb_confusion_fair = plot_confusion_matrix(nn_model_fair, X_black, y_black, "Black Confusion NN Fair", dir_path='src/DataScientist/', plot=True, vmax=None)
nno_confusion_fair = plot_confusion_matrix(nn_model_fair, X_other, y_other, "Other Confusion NN Fair", dir_path='src/DataScientist/', plot=True, vmax=None)

#%% Checking for nul hypothesis - Is there significant difference in accuracy between the original and the fair model?
from scipy.stats import ttest_rel

original_accuracies_dt = []  # Replace with actual accuracies from multiple runs
fair_accuracies_dt = []  # Replace with actual accuracies from multiple runs
original_accuracies_nn = []  # Replace with actual accuracies from multiple runs
fair_accuracies_nn = []  # Replace with actual accuracies from multiple runs 

for i in range(10):
    X_train, X_test, y_train, y_test  = create_train_test_split(X, y, test_size=0.2, random_state=42+i)
    X_train_fair,X_test_fair  = X_train.drop(columns=['sex', 'race']), X_test.drop(columns=['sex', 'race'])
    original_accuracies_dt.append(fit_and_score(model=dt_model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test))
    original_accuracies_nn.append(fit_and_score(model=nn_model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test))
    fair_accuracies_dt.append(fit_and_score(model=dt_model_fair, X_train=X_train_fair, y_train=y_train, X_test=X_test_fair, y_test=y_test))
    fair_accuracies_nn.append(fit_and_score(model=nn_model_fair, X_train=X_train_fair, y_train=y_train, X_test=X_test_fair, y_test=y_test))


# Perform paired t-test for Decision Tree
t_stat_dt, p_value_dt = ttest_rel(original_accuracies_dt, fair_accuracies_dt)
print(f"Decision Tree - t-statistic: {t_stat_dt:.4f}, p-value: {p_value_dt:.4f}")
# Perform paired t-test for Neural Network
t_stat_nn, p_value_nn = ttest_rel(original_accuracies_nn, fair_accuracies_nn)
print(f"Neural Network - t-statistic: {t_stat_nn:.4f}, p-value: {p_value_nn:.4f}")