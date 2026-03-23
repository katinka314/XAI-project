import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
import shap
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

dir_path = "src/DataScientist/"

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


def plot_logistic_coefficients(model, top_k=20, output_path='logreg.png'):
    feature_names = model.named_steps['prep'].get_feature_names_out()
    coef = model.named_steps['clf'].coef_[0]
    top_idx = abs(coef).argsort()[-top_k:]

    plt.figure(figsize=(12, 8), dpi=180)
    plt.barh(feature_names[top_idx], coef[top_idx])
    plt.xlabel('Coefficient')
    plt.title(f'Top {top_k} Logistic Regression Coefficients')
    plt.tight_layout()
    plt.savefig(dir_path + output_path)
    plt.show()


def plot_shap_summary(
    model, X_train, model_name, dir_path = 'src/DataScientist/', features = None, pretty_names = None, log_scale = True):
    pretty_names = pretty_names or {}
    X_train_t = model.named_steps['prep'].transform(X_train)
    if hasattr(X_train_t, 'toarray'):
        X_train_t = X_train_t.toarray()
    
    
    feature_names = model.named_steps['prep'].get_feature_names_out()
    pretty_feature_names = [pretty_names.get(name, name) for name in feature_names]

    explainer = shap.Explainer(model.named_steps['clf'], X_train_t)
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
            max_display = features,
            use_log_scale = log_scale
        )
    else:
        shap.summary_plot(
            shap_values,
            X_train_t,
            feature_names=pretty_feature_names,
            show=False,
            max_display = features,
            use_log_scale = log_scale
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
    pretty_feature_names = [pretty_names.get(name, name) for name in feature_names]
    
    explainer = LimeTabularExplainer(
        X_train_t,
        feature_names=pretty_feature_names,
        class_names=[str(c) for c in model.named_steps['clf'].classes_],
        mode='classification',
        discretize_continuous=True,
        random_state = 42
    )
    
    exp = explainer.explain_instance(
        X_test_t[instance_idx],
        model.named_steps['clf'].predict_proba,
        num_features=num_features,
    )
    
    
    exp_list = exp.as_list()
    exp_list_filtered = [(f, v) for f, v in exp_list
                         if not any(h in f for h in features_hidden)]
    
    features = [f for f, v in exp_list_filtered]
    values = [v for f, v in exp_list_filtered]
    colors = ['green' if v > 0 else 'red' for v in values]
    
    safe_name = model_name.lower().replace(' ', '_')
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(features, values, color=colors)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title(f'LIME Explanation for {model_name}')
    plt.tight_layout()
    fig.savefig(f'{dir_path}lime_{safe_name}.png', dpi=180)
    plt.show()
    plt.close(fig)


def plot_confusion_matrix(model, X_test, y_test, model_name, dir_path = 'src/DataScientist/', plot = True):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    if plot:
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'{model_name} Confusion Matrix')
        plt.colorbar()
        plt.xticks([0, 1], ['<=50K', '>50K'])
        plt.yticks([0, 1], ['<=50K', '>50K'])
        
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j, i, str(cm[i, j]),
                    ha='center',
                    va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black'
                )
        
        plt.xlabel('Predicted')
        plt.ylabel('True')
        safe_name = model_name.lower().replace(' ', '_')
        plt.savefig(f"{dir_path}confusion_matrix_{safe_name}.png")
        plt.show()
        
    #return TP, FP, FN, TN dict
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
    ).rename(columns={
        'actual_rate': 'Actual Rate',
        'predicted_rate': 'Predicted Rate'
    })
    
    fairness['gap'] = fairness['Predicted Rate'] - fairness['Actual Rate']

    colors = ['salmon' if x < 0 else 'steelblue' for x in fairness['gap']]

    plt.figure(figsize=(10, 6))
    fairness['gap'].plot(kind='bar', color=colors)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.title(f'Model Bias by {category} ({model_name})')
    plt.ylabel('Predicted - Actual Rate')
    plt.suptitle("Positive = model over predicts high income, Negative = model under-predicts")
    plt.tight_layout()
    plt.savefig(f'{dir_path}fairness_{category}_{model_name}.png')
    plt.show()


def plot_class_distribution(y, dir_path='src/DataScientist/'):
    clases = y.unique()
    counts = y.value_counts()
    total = counts.sum()
    pct = counts / total * 100
    
    plt.figure(figsize=(10, 6))
    plt.bar(clases, pct)
    plt.xlabel('Class')
    plt.ylabel('Percentage')
    plt.title('Class Distribution')
    plt.xticks(rotation=45)
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
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'{dir_path}roc_curve.png')
    plt.show()

def plot_feature_distribution(X_train, categorical, numerical, dir_path='src/DataScientist/'):
    
    num_cols = len(numerical)
    cols = 3
    rows = (num_cols + cols - 1) // cols
    plt.figure(figsize=(cols * 5, rows * 4))
    for i, col in enumerate(numerical):
        plt.subplot(rows, cols, i + 1)
        plt.hist(X_train[col], bins=30, color='steelblue', edgecolor='black')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'{dir_path}numerical_distributions.png')
    plt.show()
    
    
    
    num_cols = len(categorical)
    cols = 3
    rows = (num_cols + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()
    for i, col in enumerate(categorical):
        counts = X_train[col].value_counts().head(10)
        axes[i].bar(range(len(counts)), counts.values, color='steelblue')
        axes[i].set_xticks(range(len(counts)))
        axes[i].set_xticklabels(counts.index.str.strip(), rotation=40, ha='right', fontsize=9)
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_ylabel('Frequency')
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

    labels = ['Approved Good Applicant', 'Approved Bad Applicant', 'Denied Good Applicant', 'Denied Bad Applicant']
    keys = ['TP', 'FP', 'FN', 'TN']
    n_models = len(model_dict)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    for i, (model_name, stats) in enumerate(model_stats.items()):
        offsets = [j + i * width for j in range(len(keys))]
        ax.bar(offsets, [stats[k] for k in keys], width=width, label=model_name)

    tick_positions = [j + width * (n_models - 1) / 2 for j in range(len(keys))]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_title('Business Error Summary')
    ax.set_ylabel('Count')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{dir_path}business_error_summary.png', dpi=150)
    plt.show()
    
def plot_confusion_matrix_pct(model, X_test, y_test, model_name, dir_path='src/DataScientist/'):
    cfm_data = plot_confusion_matrix(model, X_test, y_test, "","", False)
    TP, FP, FN, TN = cfm_data['TP'], cfm_data['FP'], cfm_data['FN'], cfm_data['TN']
    cm = np.array([[TN, FP], [FN, TP]])
    cm = cm / cm.sum()
    #round
    cm = cm.round(2)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{model_name} Confusion Matrix')
    plt.colorbar()
    plt.xticks([0, 1], ['<=50K', '>50K'])
    plt.yticks([0, 1], ['<=50K', '>50K'])
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, str(cm[i, j]),
                ha='center',
                va='center',
                color='white' if cm[i, j] > cm.max() / 2 else 'black'
            )
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    safe_name = model_name.lower().replace(' ', '_')
    plt.savefig(f"{dir_path}confusion_matrix_{safe_name}.png")
    plt.show()