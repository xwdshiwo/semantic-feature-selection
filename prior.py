from __future__ import annotations
import numpy as np
import time
VERSION = 'guidance-4.0.1'

def f_scores(X, y):
    a, b = (X[y == 0], X[y == 1])
    delta = a.mean(0) - b.mean(0)
    within = ((a - a.mean(0)) ** 2).sum(0) + ((b - b.mean(0)) ** 2).sum(0)
    f = len(a) * len(b) / len(X) * delta ** 2 / np.maximum(within / max(1, len(X) - 2), 1e-12)
    return np.nan_to_num(f, nan=0, posinf=0, neginf=0)

def normalize(x):
    x = np.maximum(np.asarray(x, float), 0)
    return x / max(float(x.max()), 1e-12)

def build_guidance(X, y, groups, feature_ids, seed=20260925, bootstraps=16, pool_size=512):
    started = time.perf_counter()
    X = np.asarray(X, dtype=float).copy()
    X[~np.isfinite(X)] = np.nan
    with np.errstate(all='ignore'):
        med = np.nanmedian(X, axis=0)
    med = np.nan_to_num(med)
    bad = np.where(np.isnan(X))
    X[bad] = med[bad[1]]
    y, groups = (np.asarray(y), np.asarray(groups).astype(str))
    p = X.shape[1]
    raw = np.log1p(f_scores(X, y))
    scores = normalize(raw)
    ranking = np.argsort(-raw, kind='stable')
    base_seconds = time.perf_counter() - started
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    group_rows = {g: np.flatnonzero(groups == g) for g in ug}
    boot = []
    frequencies = np.zeros(p)
    for _ in range(int(bootstraps)):
        for attempt in range(100):
            rows = np.concatenate([group_rows[g] for g in rng.choice(ug, len(ug), replace=True)])
            if np.unique(y[rows]).size == 2:
                break
        else:
            raise ValueError('Unable to obtain a two-class group bootstrap')
        s = normalize(np.log1p(f_scores(X[rows], y[rows])))
        boot.append(s)
        frequencies[np.argsort(-s, kind='stable')[:min(pool_size, p)]] += 1
    boot = np.asarray(boot)
    stability = frequencies / bootstraps
    reliability = np.clip(boot.mean(0) / (boot.mean(0) + boot.std(0) + 1e-08), 0, 1)
    mixed = scores * (0.5 + 0.5 * stability) * (0.5 + 0.5 * reliability)
    mixed_ranking = np.argsort(-mixed, kind='stable')
    return dict(scores=scores, stability_scores=stability, reliability_scores=reliability, mixed_scores=mixed, ranking=mixed_ranking, anova_ranking=ranking, active_indices=mixed_ranking[:min(pool_size, p)], feature_ids=np.asarray(feature_ids).astype(str), bootstrap_scores=boot.astype(np.float32), version=np.asarray(VERSION), seed=np.asarray(seed), bootstraps=np.asarray(bootstraps), pool_size=np.asarray(pool_size), base_score_seconds=np.asarray(base_seconds), total_guidance_seconds=np.asarray(time.perf_counter() - started))
