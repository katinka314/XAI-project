# %%
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from data_transformation import (
    categorical_cols,
    create_train_test_split,
    load_adult_data,
    numeric_cols,
    split_X_y,
)
from model_factory import fit_and_score


# %%
def build_MLP(categorical_cols, numeric_cols):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            (
                "cat",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="infrequent_if_exist",
                    min_frequency=0.05,
                    max_categories=10,
                    sparse_output=False,
                ),
                categorical_cols,
            ),
        ]
    )

    return Pipeline(
        [
            ("prep", preprocessor),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=(25, 25),
                    max_iter=1000,
                    alpha=1e-5,
                    solver="sgd",
                    random_state=1,
                    tol=1e-4,
                ),
            ),
        ]
    )


def _run_tsne(X_t, random_state):
    return TSNE(
        n_components=2,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
        perplexity=min(30, len(X_t) - 1),
    ).fit_transform(X_t)


# %%
data = load_adult_data(data_path="../data/adult.data", nrows=None)
X, y = split_X_y(data)
X_train, X_test, y_train, y_test = create_train_test_split(
    X, y, test_size=0.2, random_state=42
)
# X_test: pd.DataFrame = X_test
X_test = X_test.reset_index()
mlp = build_MLP(categorical_cols, numeric_cols)
fit_and_score(mlp, X_train, y_train, X_test, y_test)
testset = X_test
# %%
# Manual forward pass through all layers of the MLP

# Extract fitted components from the pipeline
preprocessor = mlp["prep"]
mlp_clf = mlp["clf"]

# Preprocess the raw observation
X_preprocessed = preprocessor.transform(testset)

print("=== Manual Forward Pass ===")
print(f"Raw input shape: {testset.shape}")
print(f"After preprocessing shape: {X_preprocessed.shape}\n")


# Activation functions
def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)


# Get fitted weights and biases for each layer
coefs = mlp_clf.coefs_
intercepts = mlp_clf.intercepts_
out_activation = mlp_clf.out_activation_

print(f"Number of layers (incl. output): {len(coefs)}")
print(f"Output activation: {out_activation}")
for i, W in enumerate(coefs):
    print(f"  Layer {i + 1}: weights {W.shape}, bias {intercepts[i].shape}")
print()
logits = []
# Pass through each layer manually
h = X_preprocessed
for i, (W, b) in enumerate(zip(coefs, intercepts)):
    z = h @ W + b
    is_output = i == len(coefs) - 1

    if is_output:
        h = sigmoid(z)
        print(f"Layer {i + 1} (output, {out_activation}):")
        logits.append(z)
        print(f"  Logits: {z}")
        print(f"  After activation: {h}")
    else:
        h = relu(z)
        logits.append(z)
        print(f"Layer {i + 1} (hidden):")
        print(
            f"  After ReLU — min: {h.min():.4f}, max: {h.max():.4f}, "
            f"zeros: {(h == 0).sum()}/{h.size}"
        )
    print()

# Replicate sklearn's predict_proba format
manual_proba = np.hstack([1 - h, h])
mlp_classifications = mlp.predict(testset)
wrong_mask = mlp_classifications != y_test.values
pos_label = mlp.classes_[1]
fp_mask = wrong_mask & (mlp_classifications == pos_label)
fn_mask = wrong_mask & (mlp_classifications != pos_label)

print(f"Manual predict_proba:      {manual_proba}")
print(f"Sklearn predict_proba:      {mlp.predict_proba(testset)}")
print(f"Match: {np.allclose(manual_proba, mlp.predict_proba(testset))}")
# %%
layer1 = _run_tsne(logits[0], random_state=22)
layer2 = _run_tsne(logits[1], random_state=22)
layers = [layer1, layer2]
Path("tsne/Layer 1").mkdir(parents=True, exist_ok=True)
Path("tsne/Layer 2").mkdir(parents=True, exist_ok=True)
# %%

# Color settings
point_alpha = 0.6
# %%
onehotencoder = preprocessor["cat"]
layername = "Layer 1"
for layer in layers:
    vis_layer = layer

    # --- Categorical columns ---
    for col_idx, overall_category in enumerate(categorical_cols):
        _, ax = plt.subplots(figsize=(8, 5))
        encoder_categories = onehotencoder.categories_[col_idx]
        frequent_categories = set(
            c for c in encoder_categories if "infrequent" not in str(c)
        )

        total_obs = 0

        ax.set_title(f"t-SNE Visualization of Hidden {layername}\n{overall_category.replace('-', ' ').title()}")
        category_counts = X_test[overall_category].value_counts()
        for class_name in sorted(frequent_categories, key=lambda c: category_counts.get(c, 0), reverse=True):
            mask = X_test[overall_category] == class_name
            x = vis_layer[:, 0][mask]
            y = vis_layer[:, 1][mask]
            count = mask.sum()
            total_obs += count
            ax.scatter(x, y, label=str(class_name), alpha=point_alpha, edgecolors="w")

        infrequent_mask = ~X_test[overall_category].isin(frequent_categories)
        if infrequent_mask.any():
            x = vis_layer[:, 0][infrequent_mask]
            y = vis_layer[:, 1][infrequent_mask]
            total_obs += infrequent_mask.sum()
            ax.scatter(x, y, label="Other", alpha=point_alpha, edgecolors="w")

        print(f"{overall_category}: {total_obs}")
        ax.legend()
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        plt.tight_layout()
        plt.savefig(f"tsne/{layername}/{str(overall_category).strip()}")
        # plt.show()
        plt.close()

    # --- Numeric columns ---
    _LOG_SCALE_COLS = {"capital-gain", "capital-loss"}
    for col in numeric_cols:
        _, ax = plt.subplots(figsize=(8, 5))
        values = X_test[col].values
        if col in _LOG_SCALE_COLS:
            plot_values = np.clip(values, 1, None)
            norm = mcolors.LogNorm(vmin=1, vmax=max(plot_values.max(), 2))
            cbar_label = col.replace("-", " ").title() + " (log scale)"
        else:
            plot_values = values
            norm = None
            cbar_label = col.replace("-", " ").title()
        sc = ax.scatter(
            vis_layer[:, 0], vis_layer[:, 1],
            c=plot_values, cmap="viridis", alpha=point_alpha, edgecolors="w", norm=norm,
        )
        plt.colorbar(sc, ax=ax, label=cbar_label)
        ax.set_title(f"t-SNE Visualization of Hidden {layername}\n{col.replace('-', ' ').title()}")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        plt.tight_layout()
        plt.savefig(f"tsne/{layername}/{str(col).strip()}")
        # plt.show()
        plt.close()

    # --- Prediction probability (post-sigmoid, pre-threshold) ---
    _, ax = plt.subplots(figsize=(8, 5))
    proba = mlp.predict_proba(testset)[:, 1]
    sc = ax.scatter(
        vis_layer[:, 0], vis_layer[:, 1],
        c=proba, cmap="RdYlGn", vmin=0, vmax=1,
        alpha=point_alpha, edgecolors="w", linewidths=0.3, s=15,
    )
    plt.colorbar(sc, ax=ax, label="P(>50K)")
    ax.set_title(f"t-SNE Visualization of Hidden {layername}\nPrediction Probability")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(f"tsne/{layername}/prediction_probability")
    # plt.show()
    plt.close()

    # --- Misclassifications ---
    _, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        vis_layer[:, 0][~wrong_mask], vis_layer[:, 1][~wrong_mask],
        color='lightgrey', alpha=0.3, s=10, edgecolors='none', zorder=1,
    )
    ax.scatter(
        vis_layer[:, 0][fp_mask], vis_layer[:, 1][fp_mask],
        color='#e66101', alpha=0.8, s=20, edgecolors='w', linewidths=0.3,
        label=f'False Positive (n={fp_mask.sum()})', zorder=3,
    )
    ax.scatter(
        vis_layer[:, 0][fn_mask], vis_layer[:, 1][fn_mask],
        color='#5e3c99', alpha=0.8, s=20, edgecolors='w', linewidths=0.3,
        label=f'False Negative (n={fn_mask.sum()})', zorder=3,
    )
    ax.legend()
    ax.set_title(f"t-SNE Visualization of Hidden {layername}\nMisclassifications")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(f"tsne/{layername}/misclassifications")
    # plt.show()
    plt.close()

    # --- True labels ---
    _, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(f"t-SNE Visualization of Hidden {layername}\nTrue Label")
    for true_class in np.unique(y_test.values):
        mask = y_test.values == true_class
        ax.scatter(layer[:, 0][mask], layer[:, 1][mask],
                   label=str(true_class).strip(), alpha=point_alpha, edgecolors="w")
    ax.legend()
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(f"tsne/{layername}/true_labels")
    # plt.show()
    plt.close()

    # --- Classifications ---
    _, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(f"t-SNE Visualization of Hidden {layername}\nPredicted Label")
    total_obs = 0
    for prediction_class in np.unique(mlp_classifications):
        print(prediction_class)
        x = layer[:, 0][mlp_classifications == prediction_class]
        y = layer[:, 1][mlp_classifications == prediction_class]
        total_obs += len(x)
        ax.scatter(x, y, label=str(prediction_class).strip(), alpha=point_alpha, edgecolors="w")
    ax.legend()
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(f"tsne/{layername}/classifications")
    # plt.show()
    plt.close()

    layername = "Layer 2"