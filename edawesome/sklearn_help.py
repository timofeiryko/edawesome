"""
Function for visual evaluation of sklearn models.
"""

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, f1_score, auc, RocCurveDisplay

def plot_roc_curve(fpr, tpr, title=None):

    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0,1], [0,1], 'k--')

    plt.axis([0,1,0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)

    plt.show()

def pretty_classification_report(estimator, X, y, class_names, how='cv'):
    
    y_pred = estimator.predict(X)

    if how == 'cv':
        f_score = cross_val_score(estimator, X, y, cv=5, scoring='f1').mean()
    elif how == 'test':
        f_score = f1_score(y, y_pred)
    else:
        raise ValueError('how must be either "cv" or "test"')   
    
    cm = confusion_matrix(y, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    # heatmap of confusion matrix
    sns.heatmap(cm_df, annot=True, cbar=None, cmap="Blues", fmt='d')
    if how == 'cv':
        plt.title(f'Cross-validated f-score: {f_score:.3f}')
    elif how == 'test':
        plt.title(f'Test f-score: {f_score:.3f}')
    plt.ylabel('True Class'), plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.show()

    if how == 'cv':
        roc_auc = cross_val_score(estimator, X, y, cv=5, scoring='roc_auc').mean()
    elif how == 'test':
        roc_auc = roc_auc_score(y, y_pred)
    else:
        raise ValueError('how must be either "cv" or "test"')

    # roc graph
    fpr, tpr, thresholds = roc_curve(y, y_pred)
    roc_auc = auc(fpr, tpr)
    assert roc_auc == roc_auc_score(y, y_pred)

    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='').plot()
    plt.title(f'ROC curve: {roc_auc:.3f}')

    plt.show()




def plot_predicted_actual(Y_test, name, model=None, X_test=None, Y_pred=None):

    if (Y_pred is None) and (model is not None) and (X_test is not None):
        Y_pred = model.predict(X_test)
    
    if Y_pred is None:
        raise ValueError('Y_pred or model with X_test must be provided')

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(Y_test, Y_pred, alpha=0.3)
    ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--', lw=2)

    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')

    ax.set_title(f'{name}: predicted vs measured')

    plt.show()