from __future__ import annotations
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

def make_donor_aware_splits(y, groups=None, n_splits=5, seed=42):
    labels = np.asarray(y)
    if labels.ndim != 1 or set(np.unique(labels)) != {0, 1}:
        raise ValueError('Expected one-dimensional 0/1 labels with both classes')
    if int(n_splits) != n_splits or n_splits < 2:
        raise ValueError('n_splits must be an integer >= 2')
    if min(np.bincount(labels.astype(int))) < n_splits:
        raise ValueError('Each class needs at least n_splits samples')
    if groups is None:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(splitter.split(np.zeros(len(labels)), labels))
        group = None
    else:
        group = np.asarray(groups)
        if group.ndim != 1 or len(group) != len(labels):
            raise ValueError('Groups must have one value per sample')
        if group.dtype.kind in 'fc' and (not np.isfinite(group).all()):
            raise ValueError('Missing/nonfinite groups are not allowed')
        if group.dtype.kind in 'US' and np.any(np.char.str_len(group.astype(str)) == 0):
            raise ValueError('Empty group identifiers are not allowed')
        unique_groups, inverse = np.unique(group, return_inverse=True)
        if len(unique_groups) < n_splits:
            raise ValueError('Not enough independent groups for requested folds')
        group_labels = [np.unique(labels[inverse == k]) for k in range(len(unique_groups))]
        if all((len(values) == 1 for values in group_labels)):
            donor_labels = np.array([values[0] for values in group_labels], dtype=int)
            if min(np.bincount(donor_labels)) < n_splits:
                raise ValueError('Each class needs at least n_splits independent donors')
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            splits = []
            for train_donors, test_donors in splitter.split(unique_groups, donor_labels):
                train = np.flatnonzero(np.isin(inverse, train_donors))
                test = np.flatnonzero(np.isin(inverse, test_donors))
                splits.append((train, test))
        else:
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            splits = list(splitter.split(np.zeros(len(labels)), labels, group))
    coverage = np.zeros(len(labels), dtype=int)
    for train, test in splits:
        if set(labels[train]) != {0, 1} or set(labels[test]) != {0, 1}:
            raise ValueError('Split cannot provide both classes; requested rule is infeasible')
        if np.intersect1d(train, test).size or len(train) + len(test) != len(labels):
            raise RuntimeError('Invalid row partition')
        if group is not None and np.intersect1d(group[train], group[test]).size:
            raise RuntimeError('Train/test donor overlap')
        coverage[test] += 1
    if not np.all(coverage == 1):
        raise RuntimeError('Each sample must occur in exactly one test fold')
    return [(np.sort(train), np.sort(test)) for train, test in splits]
