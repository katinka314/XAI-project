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


def plot_confusion_matrix(model, X_test, y_test, model_name, dir_path = 'src/DataScientist/'):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
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
    plt.savefig(f"{dir_path}confusion_matrix{model_name}.png")
    plt.show()

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

    fairness['gap'].plot(kind='bar', color=colors)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.title(f'Model Bias by {category} ({model_name})')
    plt.ylabel('Predicted - Actual Rate')
    plt.tight_layout()
    plt.savefig(f'{dir_path}fairness_{category}_{model_name}.png')
    plt.show()