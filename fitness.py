from __future__ import annotations
import csv
import json
import time
from pathlib import Path
from typing import Any
import numpy as np
try:
    from .evaluation import BudgetExceededError, _binary_y, fit_predict, make_classifier, make_outer_splits, metrics_binary
except ImportError:
    from evaluation import BudgetExceededError, _binary_y, fit_predict, make_classifier, make_outer_splits, metrics_binary

def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and (not np.isfinite(value)):
        return None
    if isinstance(value, Path):
        return str(value)
    return value

def _dump_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(_jsonable(obj), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(dict.fromkeys((k for row in rows for k in row)))
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            encoded = {}
            for key, value in row.items():
                value = _jsonable(value)
                encoded[key] = json.dumps(value, ensure_ascii=False, allow_nan=False) if isinstance(value, (list, dict)) else value
            writer.writerow(encoded)

class CrossValFitnessEvaluator:
    INVALID_SCORE = -1000000000000.0
    _RESERVED = {'score', 'core_score', 'penalty_term', 'auc', 'acc', 'balanced_accuracy', 'BA', 'f1', 'precision', 'recall', 'specificity', 'mcc', 'logloss', 'pooled_auc', 'n_features', 'fe', 'status', 'cache_hit', 'candidate_id', 'request_id', 'mask_index', 'fold_scores', 'fold_aucs', 'fold_metrics', 'best_score', 'best_auc', 'n_fits', 'n_successful_fits', 'unique_candidates', 'elapsed', 'eval_elapsed', 'first_fe', 'selected_feature_indices', 'added_feature_indices', 'removed_feature_indices', 'parent_resolved'}

    def __init__(self, X: Any, y: Any, groups: Any, feature_ids: Any, classifier: str='lr', seed: int=42, budget: int=100, penalty: float=0.02, inner_folds: int=3, output_dir: str | Path | None=None, sample_ids: Any=None, max_features: int | None=None):
        self.X = np.asarray(X, dtype=np.float64)
        self.y = _binary_y(y, require_both=True)
        if self.X.ndim != 2 or not self.X.shape[1] or len(self.X) != len(self.y):
            raise ValueError('X must be samples-by-features and agree with y')
        self.n_features = self.X.shape[1]
        ids = np.asarray(feature_ids)
        if ids.shape != (self.n_features,):
            raise ValueError('feature_ids must have one ID per original feature column')
        self.feature_ids = ids.astype(str)
        if np.unique(self.feature_ids).size != self.n_features:
            raise ValueError('feature_ids must be unique; resolve duplicate assay rows first')
        if any((not x.strip() for x in self.feature_ids)):
            raise ValueError('feature_ids cannot be empty')
        self.sample_ids = np.arange(len(self.y)).astype(str) if sample_ids is None else np.asarray(sample_ids).astype(str)
        if self.sample_ids.shape != self.y.shape or np.unique(self.sample_ids).size != len(self.y):
            raise ValueError('sample_ids must be unique and have one ID per sample')
        self.groups = None if groups is None else np.asarray(groups).astype(str)
        self.splits = make_outer_splits(self.y, groups, n_splits=inner_folds, seed=seed)
        self.inner_folds = len(self.splits)
        self.fold_assignment = np.full(len(self.y), -1, dtype=np.int16)
        for fold, (_, valid) in enumerate(self.splits):
            if np.any(self.fold_assignment[valid] != -1):
                raise ValueError('Inner validation samples must appear exactly once')
            self.fold_assignment[valid] = fold
        if np.any(self.fold_assignment < 0):
            raise ValueError('Inner validation folds must cover every supplied sample')
        if isinstance(budget, bool) or int(budget) != budget or budget < 1:
            raise ValueError('budget must be a positive integer')
        if not np.isfinite(penalty) or penalty < 0:
            raise ValueError('penalty must be finite and nonnegative')
        if max_features is not None and (int(max_features) != max_features or not 1 <= max_features <= self.n_features):
            raise ValueError('max_features must be an integer between 1 and original p')
        make_classifier(classifier, seed)
        self.classifier, self.seed = (classifier, int(seed))
        self.budget, self.penalty = (int(budget), float(penalty))
        self.max_features = max_features
        self.output_dir = None if output_dir is None else Path(output_dir)
        self.fe = self.n_fits = self.n_successful_fits = self.n_unique = self.cache_hits = 0
        self.best_score: float | None = None
        self.best_auc: float | None = None
        self.history: list[dict[str, Any]] = []
        self._cache: dict[bytes, dict[str, Any]] = {}
        self._masks: list[np.ndarray] = []
        self._probabilities: list[np.ndarray] = []
        self._candidates: list[dict[str, Any]] = []
        self._candidate_indices: dict[str, int] = {}
        self._context: dict[str, Any] = {}
        self._started = time.perf_counter()

    @property
    def remaining(self) -> int:
        return self.budget - self.fe

    @property
    def unique_candidates(self) -> int:
        return self.n_unique

    @property
    def model_fits(self) -> int:
        return self.n_fits

    def set_context(self, **metadata: Any) -> None:
        overlap = self._RESERVED.intersection(metadata)
        if overlap:
            raise ValueError(f'Reserved evaluation metadata fields: {sorted(overlap)}')
        self._context.update(_jsonable(metadata))

    def clear_context(self) -> None:
        self._context.clear()

    def _record(self, result: dict[str, Any], start: float, cache_hit: bool, metadata: dict[str, Any]) -> dict[str, Any]:
        if result['status'] == 'ok':
            self.best_score = max(self.best_score if self.best_score is not None else self.INVALID_SCORE, float(result['score']))
            self.best_auc = max(self.best_auc if self.best_auc is not None else 0.0, float(result['auc']))
        row = {**metadata, 'core_score': None, 'penalty_term': None, **result, 'fe': self.fe, 'request_id': f'r{self.fe:06d}', 'cache_hit': cache_hit, 'elapsed': time.perf_counter() - self._started, 'eval_elapsed': time.perf_counter() - start, 'best_score': self.best_score, 'best_auc': self.best_auc, 'n_fits': self.n_fits, 'n_successful_fits': self.n_successful_fits, 'unique_candidates': self.n_unique}
        parent_id = metadata.get('parent_id')
        candidate_id = result.get('candidate_id')
        row['parent_resolved'] = bool(parent_id in self._candidate_indices)
        if row['parent_resolved'] and candidate_id in self._candidate_indices:
            parent = self._masks[self._candidate_indices[parent_id]]
            current = self._masks[self._candidate_indices[candidate_id]]
            parent = np.unpackbits(parent, count=self.n_features, bitorder='little').astype(bool)
            current = np.unpackbits(current, count=self.n_features, bitorder='little').astype(bool)
            row['added_feature_indices'] = np.flatnonzero(current & ~parent).tolist()
            row['removed_feature_indices'] = np.flatnonzero(parent & ~current).tolist()
        else:
            row['added_feature_indices'] = None
            row['removed_feature_indices'] = None
        self.history.append(row)
        return row.copy()

    def __call__(self, mask: Any, **metadata: Any) -> dict[str, Any]:
        if self.remaining <= 0:
            raise BudgetExceededError(f'FE budget exhausted ({self.fe}/{self.budget})')
        overlap = self._RESERVED.intersection(metadata)
        if overlap:
            raise ValueError(f'Reserved evaluation metadata fields: {sorted(overlap)}')
        metadata = {**self._context, **_jsonable(metadata)}
        start = time.perf_counter()
        self.fe += 1
        selection = np.asarray(mask)
        if selection.shape != (self.n_features,) or not np.all(np.isin(selection, [0, 1])):
            self._record({'score': self.INVALID_SCORE, 'auc': None, 'balanced_accuracy': None, 'BA': None, 'n_features': None, 'candidate_id': None, 'status': 'invalid_mask'}, start, False, metadata)
            raise ValueError('mask must be a length-original-p boolean or binary vector')
        selection = selection.astype(bool)
        packed = np.packbits(selection, bitorder='little')
        key = packed.tobytes()
        if key in self._cache:
            self.cache_hits += 1
            return self._record(self._cache[key], start, True, metadata)
        index = self.n_unique
        self.n_unique += 1
        candidate_id = f'c{self.n_unique:06d}'
        self._candidate_indices[candidate_id] = index
        self._masks.append(packed)
        count = int(selection.sum())
        oof = np.full(len(self.y), np.nan, dtype=np.float64)
        self._probabilities.append(oof)
        common = {'candidate_id': candidate_id, 'mask_index': index, 'first_fe': self.fe, 'n_features': count}
        fold_metrics: list[dict[str, Any]] = []
        if count == 0 or (self.max_features is not None and count > self.max_features):
            result = {**common, 'score': self.INVALID_SCORE, 'auc': None, 'acc': None, 'balanced_accuracy': None, 'BA': None, 'fold_scores': [], 'fold_aucs': [], 'fold_metrics': [], 'status': 'invalid_empty' if count == 0 else 'invalid_max_features'}
        else:
            error = None
            for fold, (train, valid) in enumerate(self.splits):
                self.n_fits += 1
                try:
                    p, _ = fit_predict(self.X[np.ix_(train, selection)], self.y[train], self.X[np.ix_(valid, selection)], self.classifier, self.seed + fold)
                    self.n_successful_fits += 1
                    oof[valid] = p
                    values = metrics_binary(self.y[valid], p)
                    fold_metrics.append({'fold': fold, 'n_train': len(train), 'n_valid': len(valid), **values})
                except Exception as exc:
                    error = f'{type(exc).__name__}: {exc}'
                    break
            if error is not None:
                result = {**common, 'score': self.INVALID_SCORE, 'auc': None, 'balanced_accuracy': None, 'BA': None, 'fold_metrics': fold_metrics, 'fold_scores': [], 'fold_aucs': [], 'status': 'fit_error', 'error': error}
            else:
                means = {name: float(np.mean([fold[name] for fold in fold_metrics])) for name in metrics_binary(self.y, oof)}
                core = 0.8 * means['auc'] + 0.2 * means['balanced_accuracy']
                penalty = self.penalty * np.log1p(count) / np.log1p(self.n_features)
                fold_scores = [0.8 * item['auc'] + 0.2 * item['balanced_accuracy'] - penalty for item in fold_metrics]
                result = {**common, **means, 'BA': means['balanced_accuracy'], 'score': float(core - penalty), 'core_score': float(core), 'penalty_term': float(penalty), 'fold_metrics': fold_metrics, 'fold_scores': [float(v) for v in fold_scores], 'fold_aucs': [item['auc'] for item in fold_metrics], 'pooled_auc': metrics_binary(self.y, oof)['auc'], 'status': 'ok'}
        self._cache[key] = result.copy()
        self._candidates.append({**metadata, **result, 'selected_feature_indices': np.flatnonzero(selection).tolist()})
        row = self._record(result, start, False, metadata)
        if result['status'] == 'fit_error':
            raise RuntimeError(f"Fitness model failed at FE {self.fe}: {result['error']}")
        return row

    def evaluate(self, mask: Any, **metadata: Any) -> dict[str, Any]:
        return self(mask, **metadata)

    def save_artifacts(self, output_dir: str | Path | None=None) -> dict[str, str]:
        directory = Path(output_dir) if output_dir is not None else self.output_dir
        if directory is None:
            raise ValueError('Provide output_dir at construction or save_artifacts')
        directory.mkdir(parents=True, exist_ok=True)
        masks = np.stack(self._masks) if self._masks else np.empty((0, (self.n_features + 7) // 8), dtype=np.uint8)
        probabilities = np.stack(self._probabilities) if self._probabilities else np.empty((0, len(self.y)), dtype=np.float64)
        candidate_ids = np.asarray([item['candidate_id'] for item in self._candidates], dtype=str)
        np.savez_compressed(directory / 'candidate_masks.npz', candidate_ids=candidate_ids, packed_masks=masks, feature_ids=self.feature_ids, n_features=np.asarray(self.n_features), bitorder=np.asarray('little'))
        np.savez_compressed(directory / 'candidate_probabilities.npz', candidate_ids=candidate_ids, p_positive=probabilities.astype(np.float32), p_positive_float64=probabilities, y_true=self.y, sample_ids=self.sample_ids, fold_assignment=self.fold_assignment, groups=np.asarray([] if self.groups is None else self.groups, dtype=str))
        _write_csv(directory / 'curve.csv', self.history)
        _write_csv(directory / 'candidate_manifest.csv', self._candidates)
        with (directory / 'candidate_fold_metrics.jsonl').open('w', encoding='utf-8') as handle:
            for item in self._candidates:
                handle.write(json.dumps(_jsonable({key: item.get(key) for key in ('candidate_id', 'first_fe', 'status', 'fold_metrics')}), ensure_ascii=False, allow_nan=False) + '\n')
        _dump_json(directory / 'inner_splits.json', {'seed': self.seed, 'n_splits': self.inner_folds, 'sample_ids': self.sample_ids, 'y_true': self.y, 'groups': self.groups, 'splits': [{'fold': fold, 'train_indices': train, 'validation_indices': valid} for fold, (train, valid) in enumerate(self.splits)]})
        _dump_json(directory / 'evaluation_summary.json', {'classifier': self.classifier, 'seed': self.seed, 'budget': self.budget, 'fe': self.fe, 'n_fits': self.n_fits, 'n_successful_fits': self.n_successful_fits, 'n_unique': self.n_unique, 'cache_hits': self.cache_hits, 'penalty': self.penalty, 'n_original_features': self.n_features, 'inner_folds': self.inner_folds, 'best_score': self.best_score, 'best_auc': self.best_auc, 'aggregation': 'unweighted mean of inner-fold metrics', 'scope': 'adaptive inner optimization only; unbiased whole-procedure evaluation requires untouched outer tests'})
        names = ['curve.csv', 'candidate_manifest.csv', 'candidate_masks.npz', 'candidate_probabilities.npz', 'candidate_fold_metrics.jsonl', 'inner_splits.json', 'evaluation_summary.json']
        return {name: str(directory / name) for name in names}
