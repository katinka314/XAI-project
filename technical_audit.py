import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from src.data_transformation import (
    categorical_cols,
    numeric_cols,
    create_train_test_split,
    load_adult_data,
    split_X_y,
)
from src.model_factory import fit_and_score, model_dt, model_lr

data = load_adult_data(data_path='data/adult.data', nrows=None)
X, y = split_X_y(data)
X_train, X_test, y_train, y_test = create_train_test_split(X, y, test_size=0.2, random_state=42)

dt_model = model_dt(categorical_cols, numeric_cols)
lr_model = model_lr(categorical_cols, numeric_cols)
fit_and_score(dt_model, X_train, y_train, X_test, y_test)
fit_and_score(lr_model, X_train, y_train, X_test, y_test)


def _preprocess_sample(prep, X, n_samples, random_state):
    sample = X.sample(n=min(n_samples, len(X)), random_state=random_state)
    X_t = prep.transform(sample)
    if hasattr(X_t, "toarray"):
        X_t = X_t.toarray()
    return sample, X_t


def _run_tsne(X_t, random_state):
    return TSNE(
        n_components=2,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
        perplexity=min(30, len(X_t) - 1),
    ).fit_transform(X_t)


def plot_tsne_2x2(model, X, y, model_name, n_samples=1000, random_state=42,
                  standardise=False, dir_path='src/DataScientist/', fname=None):
    prep, clf = model.named_steps["prep"], model.named_steps["clf"]
    sample, X_t = _preprocess_sample(prep, X, n_samples, random_state)

    X_t_tsne = StandardScaler().fit_transform(X_t) if standardise else X_t
    X_2d = _run_tsne(X_t_tsne, random_state)

    y_sample = y.loc[sample.index]
    prob_pos = clf.predict_proba(X_t)[:, 1]
    actual   = (y_sample.str.strip() == ">50K").astype(float).values
    edu      = sample['education-num'].values.astype(float)

    _marital_pretty = {
        "Divorced":              "Divorced",
        "Married-civ-spouse":    "Married",
        "Married-spouse-absent": "Spouse Absent",
        "Married-AF-spouse":     "Married (Military)",
        "Never-married":         "Never Married",
        "Separated":             "Separated",
        "Widowed":               "Widowed",
    }
    marital_cat = sample['marital-status'].str.strip().astype('category')
    marital_codes  = marital_cat.cat.codes.values
    marital_labels = [_marital_pretty.get(lbl, lbl) for lbl in marital_cat.cat.categories.tolist()]
    n_marital = len(marital_labels)

    panels = [
        {"c": prob_pos,      "title": "Predicted P(>50K)",   "cmap": "coolwarm", "vmin": 0.0, "vmax": 1.0},
        {"c": actual,        "title": "Actual label (>50K)", "cmap": "coolwarm", "vmin": 0.0, "vmax": 1.0},
        {"c": marital_codes, "title": "Marital Status",      "cmap": "tab10",    "vmin": -0.5, "vmax": n_marital - 0.5, "categorical_labels": marital_labels},
        {"c": edu,           "title": "Years of Education",  "cmap": "viridis",  "vmin": None, "vmax": None},
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, panel in zip(axes.flat, panels):
        kw = dict(vmin=panel["vmin"], vmax=panel["vmax"]) if panel["vmin"] is not None else {}
        sc = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=panel["c"], cmap=panel["cmap"], alpha=0.7, **kw)
        ax.set_title(panel["title"])
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")

        if "categorical_labels" in panel:
            cmap_obj = plt.get_cmap(panel["cmap"], n_marital)
            handles = [
                plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=cmap_obj(i), markersize=7, label=lbl)
                for i, lbl in enumerate(panel["categorical_labels"])
            ]
            ax.legend(handles=handles, fontsize=7, loc='best', framealpha=0.7)
        else:
            fig.colorbar(sc, ax=ax, shrink=0.8)

    scale_note = " (standardised for t-SNE)" if standardise else ""
    fig.suptitle(f"t-SNE — {model_name}{scale_note}", fontsize=14)
    plt.tight_layout()
    if fname is None:
        fname = f"tsne_{model_name.lower().replace(' ', '_')}_2x2"
    plt.savefig(f"{dir_path}{fname}.png", dpi=150)
    plt.show()


def plot_tsne_dt_unscaled(model, X, n_samples=1000, random_state=42,
                           dir_path='src/DataScientist/'):
    prep, clf = model.named_steps["prep"], model.named_steps["clf"]
    _, X_t = _preprocess_sample(prep, X, n_samples, random_state)
    X_2d = _run_tsne(X_t, random_state)

    leaf_ids = clf.apply(X_t)
    counts = clf.tree_.value[leaf_ids][:, 0, :]
    prob_pos = counts[:, 1] / counts.sum(axis=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=prob_pos, cmap="coolwarm",
                    alpha=0.7, vmin=0, vmax=1)
    fig.colorbar(sc, ax=ax, label="P(>50K) at leaf")
    ax.set_title("t-SNE — Decision Tree  |  Native feature space (unscaled)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(f"{dir_path}tsne_dt_unscaled.png", dpi=150)
    plt.show()


plot_tsne_dt_unscaled(dt_model, X_test)
plot_tsne_2x2(dt_model, X_test, y_test, model_name="Decision Tree",    standardise=True)
plot_tsne_2x2(lr_model, X_test, y_test, model_name="Logistic Regression", standardise=False)