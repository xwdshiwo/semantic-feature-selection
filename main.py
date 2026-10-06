import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '1')

import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from evaluation import fit_predict, make_outer_splits, metrics_binary
from fitness import CrossValFitnessEvaluator
from prior import build_guidance
from search import run_search


def encode(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def save_json(path, value):
    path.write_text(json.dumps(value, default=encode, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--knowledge', type=Path)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    with np.load(args.data, allow_pickle=False) as data:
        X, y = data['X'], data['y']
        features = data['feature_ids'].astype(str)
        samples = data['sample_ids'].astype(str)
        groups = data['groups'].astype(str) if 'groups' in data else samples.copy()
        fold_ids = data['outer_fold'] if 'outer_fold' in data else None
    if X.shape != (len(y), len(features)) or len(samples) != len(y) or len(groups) != len(y):
        raise ValueError('Input dimensions do not match')
    if len(set(samples)) != len(samples) or len(set(features)) != len(features):
        raise ValueError('Sample IDs and feature IDs must be unique')
    edges = json.loads(args.knowledge.read_text()) if args.knowledge else []
    if any(type(e['feature_index']) is not int or not 0 <= e['feature_index'] < len(features) for e in edges):
        raise ValueError('Knowledge indices must address the original feature axis')
    splits = make_outer_splits(y, groups, cfg['outer_folds'], cfg['split_seed']) if fold_ids is None else [
        (np.flatnonzero(fold_ids != f), np.flatnonzero(fold_ids == f)) for f in range(cfg['outer_folds'])]
    if fold_ids is not None and (len(fold_ids) != len(y) or set(fold_ids) != set(range(cfg['outer_folds']))):
        raise ValueError('outer_fold must cover all configured folds')
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / 'config.json', cfg)
    summaries = []
    for fold, (train, test) in enumerate(splits):
        if set(groups[train]) & set(groups[test]) or set(y[train]) != {0, 1} or set(y[test]) != {0, 1}:
            raise ValueError('Invalid group or class partition')
        out = args.output / f'fold_{fold}'
        out.mkdir()
        started = time.perf_counter()
        inner_seed = cfg['split_seed'] + fold * 97
        guide = build_guidance(X[train], y[train], groups[train], features,
                               seed=inner_seed, bootstraps=cfg['bootstraps'], pool_size=cfg['pool_size'])
        evaluator = CrossValFitnessEvaluator(X[train], y[train], groups[train], features,
            classifier='lr', seed=inner_seed, budget=cfg['budget'], penalty=cfg['penalty'],
            inner_folds=cfg['inner_folds'], sample_ids=samples[train])
        context = dict(log_dir=str(out / 'agent_records'), dataset=args.data.stem,
                       disease=cfg['disease'], symbols=dict(enumerate(features)), train_X=X[train],
                       evidence_scope=cfg['evidence_scope'])
        result = run_search('M32', evaluator, len(features), cfg['search_seed'], cfg['budget'],
                            guide, cfg['search'], knowledge_edges=edges, council_context=context)
        mask = result['mask']
        probability, _ = fit_predict(X[train][:, mask], y[train], X[test][:, mask], 'lr', inner_seed)
        summary = dict(fold=fold, **metrics_binary(y[test], probability),
                       n_features=int(mask.sum()), seconds=time.perf_counter()-started,
                       fe=evaluator.fe)
        summaries.append(summary)
        pd.DataFrame({'sample_id': samples[test], 'label': y[test],
                      'probability': probability, 'prediction': (probability >= .5).astype(int)}).to_csv(out / 'predictions.csv', index=False)
        pd.DataFrame({'feature_index': np.flatnonzero(mask), 'feature_name': features[mask]}).to_csv(out / 'selected_features.csv', index=False)
        pd.DataFrame(result['trace']).to_csv(out / 'evolution.csv', index=False)
        save_json(out / 'metrics.json', summary)
        save_json(out / 'split.json', dict(train=train, test=test))
        save_json(out / 'agent_events.json', result.get('injection_events', []))
    pd.DataFrame(summaries).to_csv(args.output / 'metrics.csv', index=False)


if __name__ == '__main__':
    main()
