import re
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
import shap
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

plt.rcParams.update({'font.size': 13, 'axes.titlesize': 14, 'axes.labelsize': 13})

dir_path = "src/DataScientist/"


def _clean_feature_name(name):
    """Strip sklearn pipeline prefixes/suffixes into a readable label."""
    name = re.sub(r'^(cat|num)__', '', name)
    name = name.replace('_infrequent_sklearn', ' (rare)')
    name = name.replace('_', ' ').strip()
    return name


def _clean_lime_label(label):
    """Remove LIME condition suffix, keeping only the feature name."""
    # Range: "37.00 < Feature Name <= 48.00"
    m = re.match(r'^[\-\d\.]+ [<>]=? (.+?) [<>]=? [\-\d\.]+$', label)
    if m:
        return m.group(1).strip()
    # Simple: "Feature Name <= 0.00" or "Feature Name > 12.00"
    m = re.match(r'^(.+?)\s*[<>!]=?\s*[\-\d\.]+$', label)
    if m:
        return m.group(1).strip()
    return label


def _pretty_lr_feature_name(name):
    """Map engineered Adult-feature names to concise, audience-friendly labels."""
    name = _clean_feature_name(name)
    name = re.sub(r'\s+', ' ', name).strip()
    name_norm = name.replace(' (rare)', ' rare')
    pretty = {
        'marital-status Married-civ-spouse': 'Married',
        'marital-status Never-married': 'Never Married',
        'relationship Not-in-family': 'Not in Family',
        'relationship Own-child': 'Own Child',
        'relationship Unmarried': 'Unmarried',
        'relationship rare': 'Other Relationship',
        'occupation Exec-managerial': 'Executive/Managerial',
        'occupation Prof-specialty': 'Professional Occupation',
        'occupation Machine-op-inspct': 'Machine Operator/Inspector',
        'occupation Other-service': 'Other Service',
        'workclass Self-emp-not-inc': 'Self-Employed (Non-inc)',
        'workclass rare': 'Other Work Type',
        'education-num': 'Years of Education',
        'education HS-grad': 'High School Graduate',
        'education Masters': "Master's Degree",
        'education rare': 'Other Education',
        'capital-gain': 'Capital Gain',
        'native-country rare': 'Other Country',
        'sex Male': 'Male',
        'race rare': 'Other Race',
    }
    if name in pretty:
        return pretty[name]
    if name_norm in pretty:
        return pretty[name_norm]

    # Fallback: remove category prefix and prettify token text.
    parts = name.split(' ', 1)
    if len(parts) == 2 and parts[0] in {'marital-status', 'relationship', 'occupation', 'workclass', 'education', 'native-country', 'sex', 'race'}:
        fallback = parts[1]
    else:
        fallback = name
    fallback = fallback.replace('-', ' ').strip().title()
    return fallback


def plot_decision_tree(model, output_path='tree.png'):
    feature_names = model.named_steps['prep'].get_feature_names_out()
    class_names = [str(c) for c in model.named_steps['clf'].classes_]

    plt.figure(figsize=(24, 12), dpi=180)
    from sklearn.tree import plot_tree

    plot_tree(
        model.named_steps['clf'],
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=7,
    )
    plt.tight_layout()
    plt.savefig(dir_path + output_path)
    plt.show()


def plot_logistic_coefficients(model, top_k=20, output_path='logreg.png', dir_path='src/DataScientist/'):
    feature_names = model.named_steps['prep'].get_feature_names_out()
    coef = model.named_steps['clf'].coef_[0]
    top_k = max(2, top_k)
    n_each_side = max(1, top_k // 2)

    pos_idx = np.argsort(coef)[-n_each_side:]
    neg_idx = np.argsort(coef)[:n_each_side]
    selected_idx = np.concatenate([neg_idx, pos_idx])

    selected_labels = [_pretty_lr_feature_name(feature_names[i]) for i in selected_idx]
    selected_values = coef[selected_idx]
    order = np.argsort(selected_values)
    selected_values = selected_values[order]
    selected_labels = [selected_labels[i] for i in order]

    colors = ['#d95f02' if v < 0 else '#1b9e77' for v in selected_values]

    fig, ax = plt.subplots(figsize=(12, 8), dpi=220)
    bars = ax.barh(selected_labels, selected_values, color=colors, edgecolor='none')
    ax.axvline(0, color='#444444', linewidth=1.0)
    ax.set_xlabel('Logistic Coefficient (impact on log-odds of >50K)')
    ax.set_title(
        f'Logistic Regression: Top {len(selected_values)} Positive and Negative Drivers',
        pad=10,
    )
    ax.grid(axis='x', linestyle='--', alpha=0.25)
    ax.set_axisbelow(True)

    max_abs = np.max(np.abs(selected_values))
    pad = max_abs * 0.04 if max_abs > 0 else 0.01
    inside_pad = max_abs * 0.02 if max_abs > 0 else 0.005
    for bar, value in zip(bars, selected_values):
        y = bar.get_y() + bar.get_height() / 2
        # For very short bars, place labels outside to keep them readable.
        if abs(value) < max_abs * 0.12:
            if value >= 0:
                ax.text(
                    value + pad * 0.6,
                    y,
                    f'{value:.3f}',
                    va='center',
                    ha='left',
                    fontsize=10,
                    color='#333333',
                    fontweight='bold',
                )
            else:
                ax.text(
                    value - pad * 0.6,
                    y,
                    f'{value:.3f}',
                    va='center',
                    ha='right',
                    fontsize=10,
                    color='#333333',
                    fontweight='bold',
                )
        elif value >= 0:
            ax.text(
                value - inside_pad,
                y,
                f'{value:.3f}',
                va='center',
                ha='right',
                fontsize=10,
                color='white',
                fontweight='bold',
            )
        else:
            ax.text(
                value + inside_pad,
                y,
                f'{value:.3f}',
                va='center',
                ha='left',
                fontsize=10,
                color='white',
                fontweight='bold',
            )

    fig.text(
        0.5,
        0.01,
        'Green features increase predicted high-income odds; orange features decrease them.',
        ha='center',
        va='bottom',
        fontsize=10,
        color='#555555',
    )
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(dir_path + output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def plot_shap_summary(
    model, X_train, model_name, dir_path='src/DataScientist/', features=None,
    pretty_names=None, log_scale=True):
    pretty_names = pretty_names or {}
    X_train_t = model.named_steps['prep'].transform(X_train)
    if hasattr(X_train_t, 'toarray'):
        X_train_t = X_train_t.toarray()

    feature_names = model.named_steps['prep'].get_feature_names_out()
    
    pretty_feature_names = [
        pretty_names.get(name, _clean_feature_name(name)) for name in feature_names
    ]

    explainer = shap.Explainer(model.named_steps['clf'].predict_proba, X_train_t)
    shap_values = explainer(X_train_t)
    safe_name = model_name.lower().replace(' ', '_')

    plt.figure(figsize=(12, 8), dpi=180)
    if shap_values.values.ndim == 3:
        class_idx = 1 if shap_values.values.shape[2] > 1 else 0
        shap.summary_plot(
            shap_values.values[:, :, class_idx],
            X_train_t,
            feature_names=pretty_feature_names,
            show=False,
            max_display=features,
            use_log_scale=log_scale
        )
    else:
        shap.summary_plot(
            shap_values,
            X_train_t,
            feature_names=pretty_feature_names,
            show=False,
            max_display=features,
            use_log_scale=log_scale
        )
    plt.title(f'SHAP Summary Plot for {model_name}')
    plt.tight_layout()
    plt.savefig(f'{dir_path}shap_{safe_name}.png')
    plt.show()


def plot_lime_explanation(model, X_train, X_test, model_name, instance_idx=0, num_features=10,
                          dir_path='src/DataScientist/', features_hidden=None, pretty_names=None):
    features_hidden = features_hidden or []
    pretty_names = pretty_names or {}

    X_train_t = model.named_steps['prep'].transform(X_train)
    X_test_t = model.named_steps['prep'].transform(X_test)
    if hasattr(X_train_t, 'toarray'):
        X_train_t = X_train_t.toarray()
    if hasattr(X_test_t, 'toarray'):
        X_test_t = X_test_t.toarray()

    feature_names = model.named_steps['prep'].get_feature_names_out()
    pretty_feature_names = [
        pretty_names.get(name, _clean_feature_name(name)) for name in feature_names
    ]

    explainer = LimeTabularExplainer(
        X_train_t,
        feature_names=pretty_feature_names,
        class_names=[str(c) for c in model.named_steps['clf'].classes_],
        mode='classification',
        discretize_continuous=True,
        random_state=42
    )

    exp = explainer.explain_instance(
        X_test_t[instance_idx],
        model.named_steps['clf'].predict_proba,
        num_features=num_features,
    )

    exp_list = exp.as_list()
    # Filter hidden features, then strip LIME conditions from labels
    exp_list_filtered = [
        (_clean_lime_label(f), v) for f, v in exp_list
        if not any(h in f for h in features_hidden)
    ]

    labels = [f for f, v in exp_list_filtered]
    values = [v for f, v in exp_list_filtered]
    colors = ['green' if v > 0 else 'red' for v in values]

    safe_name = model_name.lower().replace(' ', '_')
    fig, ax = plt.subplots(figsize=(11, max(5, len(labels) * 0.55)))
    ax.barh(labels, values, color=colors)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title(f'LIME Explanation — {model_name}', pad=10)
    ax.set_xlabel('LIME weight  (green = increases >50K prediction, red = decreases)')
    plt.tight_layout()
    fig.savefig(f'{dir_path}lime_{safe_name}.png', dpi=180, bbox_inches='tight')
    plt.show()
    plt.close(fig)


# Business labels per cell: [actual <=50K row, actual >50K row] x [pred <=50K, pred >50K]
_CM_BUSINESS_LABELS = [
    ["Correct Denial",    "Financial Loss"],
    ["Missed Profits",    "Correct Approval"],
]

def plot_confusion_matrix(model, X_test, y_test, model_name, dir_path='src/DataScientist/', plot=True, vmax=None):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    total = cm.sum()

    if plot:
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues,
                       vmin=0, vmax=vmax or cm.max())
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Number of samples')
        ax.set_title(f'{model_name} — Confusion Matrix')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Predicted  ≤50K', 'Predicted  >50K'], fontsize=12)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Actual  ≤50K', 'Actual  >50K'], fontsize=12)

        thresh = (vmax if vmax else cm.max()) / 2
        for i in range(2):
            for j in range(2):
                count = cm[i, j]
                pct = count / total * 100
                dark = count > thresh
                fg = 'white' if dark else 'black'
                fg_sub = 'white' if dark else '#555555'

                biz = _CM_BUSINESS_LABELS[i][j]
                ax.text(j, i - 0.22, biz, ha='center', va='center',
                        color=fg_sub, fontsize=9, style='italic')
                ax.text(j, i + 0.07, f'{count:,}', ha='center', va='center',
                        color=fg, fontsize=17, fontweight='bold')
                ax.text(j, i + 0.28, f'({pct:.1f}%)', ha='center', va='center',
                        color=fg, fontsize=10)

        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        plt.tight_layout()
        safe_name = model_name.lower().replace(' ', '_')
        plt.savefig(f"{dir_path}confusion_matrix_{safe_name}.png", dpi=150)
        plt.show()

    return {"TP": cm[1, 1], "FP": cm[0, 1], "FN": cm[1, 0], "TN": cm[0, 0]}


def plot_fairness(model, X_test, Y_test, category, model_name,
                  X_test_original, dir_path='src/DataScientist/'):
    predictions = model.predict(X_test)
    predictions_int = (predictions == ' >50K').astype(int)

    df_test = X_test_original.copy()
    df_test['prediction'] = predictions_int
    df_test['actual'] = (Y_test == ' >50K').astype(int)

    fairness = df_test.groupby(category).agg(
        actual_rate=('actual', 'mean'),
        predicted_rate=('prediction', 'mean')
    ).rename(columns={'actual_rate': 'Actual Rate', 'predicted_rate': 'Predicted Rate'})
    fairness['Gap'] = fairness['Predicted Rate'] - fairness['Actual Rate']
    fairness.index = fairness.index.str.strip()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9))

    # Top panel: actual vs predicted rates
    x = np.arange(len(fairness))
    width = 0.35
    ax1.bar(x - width / 2, fairness['Actual Rate'], width, label='Actual Rate', color='steelblue')
    ax1.bar(x + width / 2, fairness['Predicted Rate'], width, label='Predicted Rate', color='salmon')
    ax1.set_xticks(x)
    ax1.set_xticklabels(fairness.index, rotation=30, ha='right')
    ax1.set_ylabel('Rate of >50K income')
    ax1.set_ylim(0, min(1.0, fairness[['Actual Rate', 'Predicted Rate']].max().max() * 1.3))
    ax1.set_title(f'Actual vs Predicted Positive Rate by {category.title()}')
    ax1.legend()

    # Bottom panel: gap
    gap_colors = ['salmon' if g < 0 else 'steelblue' for g in fairness['Gap']]
    ax2.bar(fairness.index, fairness['Gap'], color=gap_colors)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_ylabel('Predicted Rate − Actual Rate')
    ax2.set_xlabel(category.title())
    ax2.set_title('Fairness Gap  (positive = over-predicts high income, negative = under-predicts)')
    plt.setp(ax2.get_xticklabels(), rotation=30, ha='right')

    fig.suptitle(f'Model Bias Analysis — {model_name}', fontsize=15)
    plt.tight_layout()
    plt.savefig(f'{dir_path}fairness_{category}_{model_name}.png', bbox_inches='tight')
    plt.show()


def plot_class_distribution(y, dir_path='src/DataScientist/'):
    counts = y.value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([c.strip() for c in counts.index], counts.values, color='steelblue')
    ax.bar_label(bars, fmt='%,.0f', padding=3)
    ax.set_xlabel('Income Class')
    ax.set_ylabel('Count')
    ax.set_title('Class Distribution in Training Data')
    plt.tight_layout()
    plt.savefig(f'{dir_path}class_distribution.png')
    plt.show()


def plot_roc_curve(models_dict, X_test, y_test, dir_path='src/DataScientist/'):
    from sklearn.metrics import roc_curve, auc
    plt.figure(figsize=(10, 6))
    for name, model in models_dict.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve((y_test == ' >50K').astype(int), y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'{dir_path}roc_curve.png')
    plt.show()


def plot_feature_distribution(X_train, categorical, numerical, dir_path='src/DataScientist/',
                               numerical_pretty=None):
    numerical_pretty = numerical_pretty or {}

    # Numerical distributions — each feature has its own y-scale (units differ)
    cols = 3
    rows = (len(numerical) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()
    for i, col in enumerate(numerical):
        label = numerical_pretty.get(col, col)
        axes[i].hist(X_train[col], bins=30, color='steelblue', edgecolor='black')
        axes[i].set_title(f'Distribution of {label}')
        axes[i].set_xlabel(label)
        axes[i].set_ylabel('Count')
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig(f'{dir_path}numerical_distributions.png')
    plt.show()

    # Categorical distributions — shared y-axis so frequencies are comparable
    cols = 3
    rows = (len(categorical) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()
    cat_counts = [X_train[col].value_counts().head(10) for col in categorical]
    shared_ymax = max(c.max() for c in cat_counts) * 1.1
    for i, (col, counts) in enumerate(zip(categorical, cat_counts)):
        axes[i].bar(range(len(counts)), counts.values, color='steelblue')
        axes[i].set_xticks(range(len(counts)))
        axes[i].set_xticklabels(counts.index.str.strip(), rotation=40, ha='right', fontsize=9)
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_ylabel('Count')
        axes[i].set_ylim(0, shared_ymax)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig(f'{dir_path}categorical_distributions.png')
    plt.show()


def plot_bias_report(model, X_test, y_test, X_test_original, category, model_name, dir_path='src/DataScientist/'):
    rows = []
    for group in sorted(X_test_original[category].unique()):
        mask = X_test_original[category] == group
        X_g = X_test[mask]
        y_g = y_test[mask]
        y_pred = model.predict(X_g)
        rep = classification_report(y_g, y_pred, output_dict=True, zero_division=0)
        rows.append({
            'Group': group.strip(),
            'N': len(y_g),
            'Accuracy': rep['accuracy'],
            'Precision': rep[' >50K']['precision'],
            'Recall': rep[' >50K']['recall'],
            'F1': rep[' >50K']['f1-score'],
        })

    df = pd.DataFrame(rows).set_index('Group')
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']

    x = range(len(df))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    for i, metric in enumerate(metrics):
        offset = [xi + i * width for xi in x]
        ax.bar(offset, df[metric], width, label=metric)

    ax.set_xticks([xi + width * (len(metrics) - 1) / 2 for xi in x])
    ax.set_xticklabels([f'{g}\n(N={df.loc[g, "N"]})' for g in df.index], rotation=30, ha='right')
    ax.set_ylim(0, 1.1)
    ax.set_title(f'Per-group metrics by {category} – {model_name}')
    ax.set_ylabel('Score')
    ax.legend()
    plt.tight_layout()
    safe_name = model_name.lower().replace(' ', '_')
    plt.savefig(f'{dir_path}bias_report_{category}_{safe_name}.png', dpi=150)
    plt.show()


def plot_business_error_summary(model_dict, X_test, y_test, dir_path='src/DataScientist/'):
    model_stats = {}
    for model_name, model in model_dict.items():
        model_stats[model_name] = plot_confusion_matrix(model, X_test, y_test, model_name, dir_path=dir_path, plot=False)

    labels = ['Approved\nGood Applicant', 'Approved\nBad Applicant', 'Denied\nGood Applicant', 'Denied\nBad Applicant']
    keys = ['TP', 'FP', 'FN', 'TN']
    n_models = len(model_dict)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    for i, (model_name, stats) in enumerate(model_stats.items()):
        offsets = [j + i * width for j in range(len(keys))]
        ax.bar(offsets, [stats[k] for k in keys], width=width, label=model_name)

    tick_positions = [j + width * (n_models - 1) / 2 for j in range(len(keys))]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(labels)
    ax.set_title('Business Error Summary')
    ax.set_ylabel('Count')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{dir_path}business_error_summary.png', dpi=150)
    plt.show()
