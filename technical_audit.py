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

# --------------------------------------------
# Create output directories
# --------------------------------------------
os.makedirs("src/DataScientist", exist_ok=True)
os.makedirs("src/Director", exist_ok=True)
os.makedirs("src/EndUser", exist_ok=True)

# --------------------------------------------
# Load and split data
# --------------------------------------------
df = load_adult_data()
X, y = split_X_y(df)
X_train, X_test, y_train, y_test = create_train_test_split(X, y)

# --------------------------------------------
# Train Logistic Regression
# --------------------------------------------
lr_model = model_lr(categorical_cols, numeric_cols)
lr_model.fit(X_train, y_train)

# --------------------------------------------
# ========= CONCEPT-BASED EXPLANATION =========
# --------------------------------------------

# Define human-interpretable concepts
concepts = {
    "Demographic": ["age", "sex", "race"],
    "Education": ["education", "education-num"],
    "Work": ["hours-per-week", "occupation", "workclass"],
    "Capital": ["capital-gain", "capital-loss"],
}

# Extract trained components
prep = lr_model.named_steps["prep"]
clf = lr_model.named_steps["clf"]

# Get feature names after preprocessing
feature_names = prep.get_feature_names_out()

# Create coefficient series
coef_df = pd.Series(
    clf.coef_[0],
    index=feature_names
)

# Compute concept influence
concept_influence = {}

for concept, raw_features in concepts.items():
    matching_features = [
        f for f in coef_df.index
        if any(rf in f for rf in raw_features)
    ]
    concept_influence[concept] = coef_df[matching_features].abs().sum()

concept_influence = (
    pd.Series(concept_influence)
    .sort_values(ascending=False)
)

print("\nConcept influence (logistic regression):")
print(concept_influence)

# Save for audit / reporting
concept_influence.to_csv("src/DataScientist/lr_concept_influence.csv")

# --------------------------------------------
# ========= LATENT SPACE VISUALIZATION =========
# --------------------------------------------

# Subsample for t-SNE
sample = X_test.sample(n=2000, random_state=42)
sample_idx = sample.index
X_sample = X_test.loc[sample_idx]
y_sample = y_test.loc[sample_idx]

# Transform features
X_sample_transformed = prep.transform(X_sample)
if hasattr(X_sample_transformed, "toarray"):
    X_sample_transformed = X_sample_transformed.toarray()

# Decision function = latent representation
logits = clf.decision_function(X_sample_transformed)

# Apply t-SNE
tsne = TSNE(
    n_components=2,
    random_state=42,
    learning_rate="auto",
    init="pca",
    perplexity=30
)

X_tsne = tsne.fit_transform(logits.reshape(-1, 1))

# Encode labels for coloring
y_colors = (y_sample == ">50K").astype(int)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    c=y_colors,
    cmap="coolwarm",
    s=12,
    alpha=0.7
)
plt.title("t-SNE of Logistic Regression Latent Space")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.colorbar(label="Income class")
plt.tight_layout()

# Save figure
plt.savefig("src/DataScientist/lr_tsne.png")
plt.close()

print("\nSaved outputs:")
print("- src/DataScientist/lr_concept_influence.csv")
print("- src/DataScientist/lr_tsne.png")