# ============================================
# Logistic Regression – Advanced Model Analysis
# Concept-Based Explanations + Latent Space t-SNE
# ============================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from training_models import model_lr
from src.data_transformation import (
    categorical_cols,
    numeric_cols,
    create_train_test_split,
    load_adult_data,
    split_X_y,
)


df = load_adult_data()
X, y = split_X_y(df)
X_train, X_test, y_train, y_test = create_train_test_split(X, y)

lr = model_lr(categorical_cols, numeric_cols)
lr.fit(X_train, y_train)


import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

def plot_tsne_decision_boundary(prep, clf, X, y, n_samples=200, random_state=42):
    # Subsample
    n_samples = min(n_samples, len(X))
    sample = X.sample(n=n_samples, random_state=random_state)
    y_sample = y.loc[sample.index]

    # Transform features
    X_transformed = prep.transform(sample)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    tsne = TSNE(
        n_components=2,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
        perplexity=min(30, n_samples - 1)
    )

    X_tsne = tsne.fit_transform(X_transformed)

    # Plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_tsne[:, 0],
        X_tsne[:, 1],
        c=(y_sample.str.strip() == ">50K").astype(int),
        cmap="coolwarm",
        alpha=0.7
    )
    plt.colorbar(scatter, label="Class (>50K)")
    plt.title("t-SNE of Decision Function (Distance to Decision Boundary)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.show()


plot_tsne_decision_boundary(
    prep=lr.named_steps["prep"],
    clf=lr.named_steps["clf"],
    X=X_test,
    y=y_test
)




