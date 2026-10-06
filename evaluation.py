from __future__ import annotations
import time
from typing import Any
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

def _binary_y(y: Any, *, require_both: bool=False) -> np.ndarray:
    arr = np.asarray(y)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError('y must be a nonempty one-dimensional array')
    if not np.all(np.isin(arr, [0, 1])):
        raise ValueError('Binary labels must be explicitly encoded as 0 and 1')
    arr = arr.astype(np.int64)
    if require_both and np.unique(arr).size != 2:
        raise ValueError('Both classes are required for training and fitness validation')
    return arr

def _finite_matrix(X: Any) -> np.ndarray:
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2 or not arr.shape[0] or (not arr.shape[1]):
        raise ValueError('X must be a nonempty samples-by-features matrix')
    arr = arr.copy()
    arr[~np.isfinite(arr)] = np.nan
    return arr

def make_classifier(name: str, seed: int=42) -> BaseEstimator:
    if name != 'lr':
        raise ValueError('The classifier is lr')
    return LogisticRegression(C=1.0, class_weight='balanced', solver='lbfgs', max_iter=2000, random_state=seed, n_jobs=1)

def fit_predict(Xtrain: Any, ytrain: Any, Xtest: Any, name: str, seed: int=42) -> tuple[np.ndarray, Pipeline]:
    train, test = (_finite_matrix(Xtrain), _finite_matrix(Xtest))
    y = _binary_y(ytrain, require_both=True)
    if len(y) != len(train) or train.shape[1] != test.shape[1]:
        raise ValueError('Training labels or train/test feature dimensions mismatch')
    classifier = make_classifier(name, seed)
    pipeline = Pipeline([('finite', FunctionTransformer(_finite_matrix, validate=False)), ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)), ('scaler', StandardScaler()), ('classifier', classifier)])
    pipeline.fit(train, y)
    classes = pipeline.named_steps['classifier'].classes_
    positive_column = int(np.flatnonzero(classes == 1)[0])
    probability = pipeline.predict_proba(test)[:, positive_column]
    if not np.all(np.isfinite(probability)):
        raise RuntimeError('Classifier produced nonfinite probabilities')
    return (np.asarray(probability, dtype=float), pipeline)

def metrics_binary(y: Any, p: Any) -> dict[str, float | None]:
    labels = _binary_y(y)
    probability = np.asarray(p, dtype=float)
    if probability.shape != labels.shape:
        raise ValueError('Probability and label shapes mismatch')
    if not np.all(np.isfinite(probability)) or np.any((probability < 0) | (probability > 1)):
        raise ValueError('Probabilities must be finite and within [0, 1]')
    pred = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    return {'auc': float(roc_auc_score(labels, probability)) if np.unique(labels).size == 2 else None, 'acc': float(accuracy_score(labels, pred)), 'balanced_accuracy': float(balanced_accuracy_score(labels, pred)), 'f1': float(f1_score(labels, pred, zero_division=0)), 'precision': float(precision_score(labels, pred, zero_division=0)), 'recall': float(recall_score(labels, pred, zero_division=0)), 'specificity': float(tn / (tn + fp)) if tn + fp else None, 'mcc': float(matthews_corrcoef(labels, pred)), 'logloss': float(log_loss(labels, probability, labels=[0, 1]))}

def _groups_array(groups: Any, n: int) -> np.ndarray | None:
    if groups is None:
        return None
    arr = np.asarray(groups)
    if arr.ndim != 1 or len(arr) != n:
        raise ValueError('groups must have one value per sample')
    if any((value is None or str(value).strip().lower() in {'', 'nan', 'none'} for value in arr)):
        raise ValueError('Missing group identifiers must be resolved before splitting')
    arr = arr.astype(str)
    return arr if np.unique(arr).size < n else None

def make_outer_splits(y: Any, groups: Any=None, n_splits: int=5, seed: int=42) -> list[tuple[np.ndarray, np.ndarray]]:
    labels = _binary_y(y, require_both=True)
    _groups_array(groups, len(labels))
    try:
        from splitting import make_donor_aware_splits
    except ImportError:
        from splitting import make_donor_aware_splits
    return make_donor_aware_splits(labels, groups, n_splits, seed)

class BudgetExceededError(RuntimeError):
    pass
