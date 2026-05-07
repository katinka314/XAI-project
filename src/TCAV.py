# %%
import os
os.makedirs("src/DataScientist", exist_ok=True)
os.makedirs("src/Director", exist_ok=True)
os.makedirs("src/EndUser", exist_ok=True)

from data_transformation import (
    categorical_cols,
    create_train_test_split,
    load_adult_data,
    numeric_cols,
    split_X_y,
)

from model_factory import fit_and_score, model_dt, model_lr, model_MLP
# %%
#Load data
data = load_adult_data(data_path='../data/adult.data', nrows = None )
X, y = split_X_y(data)
X_train, X_test, y_train, y_test = create_train_test_split(X, y, test_size=0.2, random_state=42)
# %%
X_sex_train, X_sex_test = X_train.drop(columns=['sex']), X_test.drop(columns=['sex'])
male_indexes = X_train[X_train["sex"] == " Male"].index.tolist()
female_indexes = X_train[X_train["sex"] == " Female"].index.tolist()

X_male, y_male = X_sex_train.loc[male_indexes], y_train.loc[male_indexes]
X_female, y_female = X_sex_train.loc[female_indexes], y_train.loc[female_indexes]

X_race = X.copy()
X_race = X_race.drop(columns=['race']);
X_race_train, X_race_test, y_race_train, y_race_test = create_train_test_split(X_race, y, test_size=0.2, random_state=42)


X_age = X.copy()
X_age = X_age.drop(columns=['age']);
X_age_train, X_age_test, y_age_train, y_age_test = create_train_test_split(X_age, y, test_size=0.2, random_state=42)

# %% [markdown]
# Gender analyisis
# %%
categorical_cols_sex = categorical_cols[0:6] + [categorical_cols[7]]
# %%
dt_model = model_dt(categorical_cols_sex, numeric_cols)
lr_model = model_lr(categorical_cols_sex, numeric_cols)

dt_model.fit(X_sex_train, y_train);
lr_model.fit(X_sex_train, y_train);
# %%
predictions = dt_model.predict(X_female)

print(sum(predictions == " <=50K"))
print(sum(predictions != " >50K"))

predictions = dt_model.predict(X_male)

print(sum(predictions == " <=50K"))
print(sum(predictions != " >50K"))
# %% [markdown]
# TCAV — reusable implementation
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _directional_derivatives(main_model, X_transformed, v):
    clf = main_model.named_steps["clf"]
    if hasattr(clf, "coef_"):
        return np.full(len(X_transformed), np.dot(clf.coef_[0], v))
    else:
        eps = 1e-4
        p_plus = clf.predict_proba(X_transformed + eps * v)[:, 1]
        p_minus = clf.predict_proba(X_transformed - eps * v)[:, 1]
        return (p_plus - p_minus) / (2 * eps)


def compute_tcav(
    main_model,
    X_train,
    X_test,
    y_test,
    concept_col,
    pos_value,
    neg_value=None,
    concept_name=None,
    pos_label=None,
    n_random=50,
    random_state=42,
):
    from sklearn.linear_model import LogisticRegression

    concept_name = concept_name or concept_col
    pos_label = pos_label or pos_value.strip()

    # Columns the main model's preprocessor expects
    prep = main_model.named_steps["prep"]
    prep_cols = list(prep.feature_names_in_)

    # Split training rows into positive/negative concept groups
    pos_idx = X_train[X_train[concept_col] == pos_value].index
    if neg_value is None:
        neg_idx = X_train[X_train[concept_col] != pos_value].index
    else:
        neg_idx = X_train[X_train[concept_col] == neg_value].index

    # Transform concept groups through main model's preprocessor
    X_pos_t = prep.transform(X_train.loc[pos_idx, prep_cols])
    X_neg_t = prep.transform(X_train.loc[neg_idx, prep_cols])
    if hasattr(X_pos_t, "toarray"):
        X_pos_t, X_neg_t = X_pos_t.toarray(), X_neg_t.toarray()

    X_concept_t = np.vstack([X_pos_t, X_neg_t])
    y_concept = np.concatenate([np.ones(len(X_pos_t)), np.zeros(len(X_neg_t))])

    # Train CAV as a plain LR in the main model's feature space
    cav_clf = LogisticRegression(max_iter=5000, solver="saga", tol=1e-2)
    cav_clf.fit(X_concept_t, y_concept)
    cav = cav_clf.coef_[0]
    v = cav / np.linalg.norm(cav)

    # Transform test data
    X_transformed = prep.transform(X_test[prep_cols])
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    # TCAV scores and directional derivative per true class
    dds = _directional_derivatives(main_model, X_transformed, v)
    y_binary = (y_test.str.strip() == ">50K").values
    tcav_pos = float(np.mean(dds[y_binary] > 0))
    tcav_neg = float(np.mean(dds[~y_binary] > 0))
    dd_pos = float(np.mean(dds[y_binary]))
    dd_neg = float(np.mean(dds[~y_binary]))

    # Random CAV baseline
    rng = np.random.default_rng(random_state)
    rand_pos, rand_neg = [], []
    for _ in range(n_random):
        y_shuffled = rng.permutation(y_concept)
        rand_clf = LogisticRegression(max_iter=5000, solver="saga", tol=1e-2)
        rand_clf.fit(X_concept_t, y_shuffled)
        rand_cav = rand_clf.coef_[0]
        rand_v = rand_cav / np.linalg.norm(rand_cav)
        rand_dds = _directional_derivatives(main_model, X_transformed, rand_v)
        rand_pos.append(float(np.mean(rand_dds[y_binary] > 0)))
        rand_neg.append(float(np.mean(rand_dds[~y_binary] > 0)))

    # Plot
    model_name = type(main_model.named_steps["clf"]).__name__
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, rand_scores, tcav_val, dd_val, class_label in zip(
        axes,
        [rand_pos, rand_neg],
        [tcav_pos, tcav_neg],
        [dd_pos, dd_neg],
        [">50K", "<=50K"],
    ):
        ax.hist(rand_scores, bins=20, color="steelblue", alpha=0.7, label="Random CAVs")
        ax.axvline(tcav_val, color="crimson", linewidth=2,
                   label=f"{pos_label} CAV ({tcav_val:.3f})")
        ax.set_title(f"{class_label} class  |  directional derivative: {dd_val:.4f}")
        ax.set_xlabel("TCAV Score")
        ax.set_ylabel("Count")
        ax.legend()
    plt.suptitle(f"TCAV: {concept_name} | {model_name}", fontsize=13)
    plt.tight_layout()
    plt.show()

    print(f"[{model_name} | {concept_name}]")
    print(f"  >50K  — TCAV: {tcav_pos:.3f}  dd: {dd_pos:.4f}")
    print(f"  <=50K — TCAV: {tcav_neg:.3f}  dd: {dd_neg:.4f}")
    return {"tcav_pos": tcav_pos, "tcav_neg": tcav_neg, "dd_pos": dd_pos, "dd_neg": dd_neg}


# %%
# Fit a model with ALL sensitive columns removed
sensitive_cols = ["sex", "race", "native-country"]
adj_cat = [c for c in categorical_cols if c not in sensitive_cols]

lr_model_fair = model_lr(adj_cat, numeric_cols)
dt_model_fair = model_dt(adj_cat, numeric_cols)
mlp_model_fair = model_MLP(adj_cat, numeric_cols)
X_train_fair = X_train.drop(columns=sensitive_cols)
X_test_fair = X_test.drop(columns=sensitive_cols)
lr_model_fair.fit(X_train_fair, y_train)
dt_model_fair.fit(X_train_fair, y_train)
mlp_model_fair.fit(X_train_fair, y_train)

# %%
for model in [dt_model_fair, lr_model_fair, mlp_model_fair]:
    compute_tcav(model, X_train, X_test, y_test,
                 concept_col="sex", pos_value=" Male", neg_value=" Female",
                 concept_name="Gender", pos_label="Male")

    compute_tcav(model, X_train, X_test, y_test,
                 concept_col="race", pos_value=" White",
                 concept_name="Race", pos_label="White")

    compute_tcav(model, X_train, X_test, y_test,
                 concept_col="native-country", pos_value=" United-States",
                 concept_name="Country", pos_label="United-States")