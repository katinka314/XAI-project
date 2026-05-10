# %%
from pathlib import Path

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
    for col_idx, overall_category in enumerate(categorical_cols):
        _, ax = plt.subplots(figsize=(8, 5))
        encoder_categories = onehotencoder.categories_[col_idx]
        frequent_categories = set(
            c for c in encoder_categories if "infrequent" not in str(c)
        )

        total_obs = 0

        ax.set_title(f"t-SNE Visualization of Hidden {layername}\n{overall_category.replace('-', ' ').title()}")
        for class_name in sorted(frequent_categories):
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
        plt.savefig(f"tsne/{layername}/{str(overall_category).strip()}")
        # plt.show()
        ax.clear()

    _, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(f"t-SNE Visualization of Hidden {layername}\n Classifications")
    total_obs = 0
    for prediction_class in np.unique(mlp_classifications):
        print(prediction_class)
        x = layer1[:, 0][mlp_classifications == prediction_class]
        y = layer1[:, 1][mlp_classifications == prediction_class]
        total_obs += len(x)
        ax.scatter(x, y, label=prediction_class, alpha=point_alpha, edgecolors="w")
    ax.legend()
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
        # plt.scatter(x, y)
    plt.savefig(f"tsne/{layername}/classifications")
    # plt.show()

    layername = "Layer 2"