import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    average_precision_score,
    roc_auc_score,
    confusion_matrix
)

from sklearn.calibration import CalibratedClassifierCV, calibration_curve




def pr_auc(y_true, y_proba):

    return average_precision_score(y_true, y_proba)

def print_validation_scores(model,  X_train, y_train, splits=5):
    tscv = TimeSeriesSplit(n_splits=splits)
    cv_scores = {}

    fold_scores = {'precision':[], 'recall':[], 'f1':[], 'pr_auc':[], 'accuracy': []}

    for tr_idx, va_idx in tscv.split(X_train):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]

        y_tr = y_train[tr_idx].ravel()
        y_va = y_train[va_idx].ravel()

        model.fit(X_tr, y_tr)

        proba = model.predict_proba(X_va)[:, 1]
        preds = model.predict(X_va)

        fold_scores['accuracy'].append(accuracy_score(y_va, preds))
        fold_scores['pr_auc'].append(
            average_precision_score(y_va, proba.reshape(-1, 1))
        )
        fold_scores['precision'].append(
            precision_recall_fscore_support(y_va, preds, average='binary')[0]
        )
        fold_scores['recall'].append(
            precision_recall_fscore_support(y_va, preds, average='binary')[1]
        )
        fold_scores['f1'].append(
            precision_recall_fscore_support(y_va, preds, average='binary')[2]
        )

    means = [{k: np.mean(v)} for k, v in fold_scores.items()]
    stds = [{k: np.std(v)} for k, v in fold_scores.items()]

    print("============ Metrics means:")
    print(means)
    print("============ Metrics STDs:")
    print(stds)

def print_test_score(test_proba, y_test, t=0.7):
    test_pred = (test_proba >= t).astype(int)
    acc = accuracy_score(y_test, test_pred)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, test_pred, average='binary', zero_division=0
    )

    roc = roc_auc_score(y_test, test_proba)
    prauc = average_precision_score(y_test, test_proba)

    cm = confusion_matrix(y_test, test_pred)
    print(f"\n=== Test Metrics === at threshold {t}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"ROC-AUC: {roc:.4f}")
    print(f"PR-AUC: {prauc:.4f}")
    print("Confusion Matrix:\n", sns.heatmap(cm, 
                annot=True, 
                fmt='d',  
                cmap='Blues', 
                linewidths=0.5, 
                square=True, 
                annot_kws={"size": 12}))

def calibrate_and_plot(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    method="sigmoid",
    n_splits=5,
    n_bins=10,
    strategy="quantile",
    plot=True
):
    """
    Calibrate a probabilistic classifier using time-series CV
    and optionally plot reliability curves.

    Returns:
        calibrated_model
        proba_uncal (np.array)
        proba_cal (np.array)
        calibration_data (dict)
    """


    base_model = model
    base_model.fit(X_train, y_train)

    proba_uncal = base_model.predict_proba(X_test)[:, 1]

    tscv = TimeSeriesSplit(n_splits=n_splits)

    calibrated_model = CalibratedClassifierCV(
        estimator=base_model,
        method=method,
        cv=tscv
    )

    calibrated_model.fit(X_train, y_train)
    proba_cal = calibrated_model.predict_proba(X_test)[:, 1]


    emp_uncal, pred_uncal = calibration_curve(
        y_test, proba_uncal, n_bins=n_bins, strategy=strategy
    )

    emp_cal, pred_cal = calibration_curve(
        y_test, proba_cal, n_bins=n_bins, strategy=strategy
    )


    if plot:
        plt.figure(figsize=(7, 7))
        plt.plot([0, 1], [0, 1], "k--", linewidth=2, label="Perfect calibration")

        plt.plot(pred_uncal, emp_uncal, "o-", label="Uncalibrated model")
        plt.plot(pred_cal, emp_cal, "s-", label="Calibrated model")

        plt.xlabel("Predicted probability")
        plt.ylabel("Empirical probability")
        plt.title("Calibration (Reliability) Curve")
        plt.legend()
        plt.grid(True)
        plt.show()

    calibration_data = {
        "uncalibrated": {"pred": pred_uncal, "emp": emp_uncal},
        "calibrated": {"pred": pred_cal, "emp": emp_cal},
    }

    return calibrated_model, proba_uncal, proba_cal, calibration_data