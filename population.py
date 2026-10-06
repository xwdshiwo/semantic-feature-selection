from __future__ import annotations
import math
import numpy as np
VERSION = 'population-5.0.0'
METHODS = ['population_static_v5', 'population_feedback_v5']
ANCHORS = (1, 2, 3, 5, 8, 12, 20, 30, 50)
NICHE_NAMES = ('small_1_5', 'medium_6_20', 'large_over_20')

def _niche(k):
    return 0 if k <= 5 else 1 if k <= 20 else 2

def _key(record):
    return (float(record['score']), -int(record['n_features']))

def _core(record):
    values = np.asarray(record['fold_scores'], dtype=float)
    if values.ndim != 1 or not values.size or (not np.isfinite(values).all()):
        raise ValueError('A finite vector of internal fold_scores is required')
    penalty = float(record.get('penalty_term', 0.0))
    if not math.isfinite(penalty):
        raise ValueError('Nonfinite penalty_term')
    return values + penalty

def _record_summary(record):
    return dict(candidate_id=record['candidate_id'], indices=np.flatnonzero(record['mask']).tolist(), score=float(record['score']), auc=float(record.get('auc', np.nan)), n_features=int(record['n_features']), fold_scores=np.asarray(record['fold_scores']).tolist(), penalty_term=float(record.get('penalty_term', 0.0)), first_search_call=int(record['search_call']))

def run_search(method, evaluator, n_features, seed, budget, guidance, config=None):
    if method not in METHODS:
        raise ValueError(f'Unknown population method {method}')
    p, budget = (int(n_features), int(budget))
    if p < 1 or budget < 1:
        raise ValueError('Positive dimensions and budget required')
    start_remaining = int(evaluator.remaining)
    limit = min(budget, start_remaining)
    if limit < 1:
        raise ValueError('No evaluator budget remains')
    cfg = dict(config or {})
    capacity = int(cfg.get('niche_capacity', 6))
    if not 1 <= capacity <= 6:
        raise ValueError('niche_capacity must be between 1 and 6')
    eta = float(cfg.get('feedback_eta', 0.2))
    beta = float(cfg.get('feedback_beta', math.log(3.0)))
    reward_floor = float(cfg.get('reward_scale_floor', 0.02))
    exploration = float(cfg.get('outside_exploration_probability', 0.1))
    guard_weight = float(cfg.get('guard_weight', 0.5))
    guard_floor = float(cfg.get('guard_mean_floor', 0.002))
    guard_worst = float(cfg.get('guard_worst_loss', 0.01))
    if not (0 <= eta <= 1 and 0 <= beta <= 5 and (reward_floor > 0) and (0 <= exploration <= 1)):
        raise ValueError('Invalid feedback or exploration parameters')
    if min(guard_weight, guard_floor, guard_worst) < 0:
        raise ValueError('Negative guard setting')
    ranking = np.asarray(guidance['ranking'], dtype=int).reshape(-1)
    if len(ranking) != p or not np.array_equal(np.sort(ranking), np.arange(p)):
        raise ValueError('ranking must be a permutation of all original feature indices')
    prior = np.asarray(guidance['mixed_scores'], dtype=float).reshape(-1).copy()
    if len(prior) != p or not np.isfinite(prior).all() or (prior < 0).any():
        raise ValueError('mixed_scores must be finite, nonnegative and aligned to all features')
    prior /= max(float(prior.max()), 1e-12)
    base_weights = 0.05 + prior
    active = np.asarray(guidance.get('active_indices', ranking[:min(512, p)]), dtype=int).reshape(-1)
    if not active.size or len(np.unique(active)) != len(active) or ((active < 0) | (active >= p)).any():
        raise ValueError('Invalid original-axis active_indices')
    active_flag = np.zeros(p, dtype=bool)
    active_flag[active] = True
    rng = np.random.default_rng(seed)
    credit = np.zeros(p, dtype=float)
    feedback = method == 'population_feedback_v5'
    archive = [[], [], []]
    trace, reward_events, archive_events = ([], [], [])
    calls = 0
    raw_best = None
    anchor_records = []
    initialized_calls = 0

    def remaining():
        return max(0, min(limit - calls, int(evaluator.remaining)))

    def weights():
        return base_weights * np.exp(beta * credit)

    def choose(indices, count=1, inverse=False):
        indices = np.asarray(indices, dtype=int)
        w = weights()[indices]
        if inverse:
            w = 1.0 / w
        w /= w.sum()
        return rng.choice(indices, size=count, replace=False, p=w)

    def evaluate(mask, parent=None, second_parent=None, phase='evolution', operator='swap', generation=0, **metadata):
        nonlocal calls, raw_best
        if not remaining():
            raise RuntimeError('FE budget exceeded')
        mask = np.asarray(mask, dtype=bool).copy()
        if mask.shape != (p,) or not mask.any():
            raise ValueError('Nonempty original-axis mask required')
        parent_id = parent['candidate_id'] if parent is not None else None
        second_id = second_parent['candidate_id'] if second_parent is not None else None
        row = dict(evaluator(mask.copy(), parent_id=parent_id, parent_candidate_id=parent_id, other_parent_ids=[] if second_id is None else [second_id], second_parent_id=second_id, generation=int(generation), phase=phase, operator=operator, algorithm=method, **metadata))
        calls += 1
        if row.get('status', 'ok') != 'ok' or not np.isfinite(float(row['score'])):
            raise ValueError(f"Invalid fitness candidate: {row.get('status')} {row.get('error', '')}")
        row.update(mask=mask, n_features=int(mask.sum()), search_call=calls)
        row.setdefault('candidate_id', f'c{calls:06d}')
        _core(row)
        if raw_best is None or _key(row) > _key(raw_best):
            raw_best = row
        return row

    def insert(record, eligible=True, reason='guarded_parent_improvement'):
        ni = _niche(record['n_features'])
        if not eligible:
            return (False, None, 'parent_or_guard_failed')
        members = archive[ni]
        if any((np.array_equal(record['mask'], r['mask']) for r in members)):
            return (False, None, 'duplicate_archive_mask')
        removed = None
        if len(members) < capacity:
            members.append(record)
        else:
            worst = min(range(len(members)), key=lambda i: _key(members[i]))
            if _key(record) <= _key(members[worst]):
                return (False, None, 'destination_niche_not_improved')
            removed = members[worst]['candidate_id']
            members[worst] = record
        archive_events.append(dict(fe=int(record.get('fe', calls)), candidate_id=record['candidate_id'], niche=NICHE_NAMES[ni], removed_candidate_id=removed, reason=reason, members_after=[r['candidate_id'] for r in members]))
        return (True, removed, reason)

    def current_best():
        return max((r for members in archive for r in members), key=_key)

    def log(record, accepted, reason, phase, operator, parent=None, second_parent=None, generation=0, **extra):
        selected = current_best()
        trace.append(dict(fe=int(record.get('fe', calls)), call=calls, candidate_id=record['candidate_id'], phase=phase, operator=operator, generation=generation, parent_id=parent['candidate_id'] if parent else None, parent_candidate_id=parent['candidate_id'] if parent else None, second_parent_id=second_parent['candidate_id'] if second_parent else None, score=float(record['score']), auc=float(record.get('auc', np.nan)), n_features=int(record['n_features']), accepted=bool(accepted), acceptance_reason=reason, child_niche=NICHE_NAMES[_niche(record['n_features'])], best_score=float(selected['score']), best_n_features=int(selected['n_features']), raw_best_score=float(raw_best['score']), niche_sizes=[len(m) for m in archive], **extra))
    for k in ANCHORS:
        if k > p or not remaining():
            continue
        mask = np.zeros(p, dtype=bool)
        mask[ranking[:k]] = True
        row = evaluate(mask, phase='initialization', operator='rank_anchor', candidate_k=k)
        accepted, _, reason = insert(row, reason='initial_anchor')
        anchor_records.append(_record_summary(row))
        log(row, accepted, reason, 'initialization', 'rank_anchor')
    grids = [[k for k in ANCHORS if k <= len(active) and _niche(k) == n] for n in range(3)]
    available = [n for n in range(3) if grids[n]]
    target = min(limit, max(calls, capacity * len(available)))
    schedule_index = 0
    while calls < target and remaining():
        ni = available[schedule_index % len(available)]
        schedule_index += 1
        k = int(rng.choice(grids[ni]))
        mask = np.zeros(p, dtype=bool)
        mask[choose(active, k)] = True
        row = evaluate(mask, phase='initialization', operator='weighted_multiscale', candidate_k=k)
        accepted, _, reason = insert(row, reason='initial_weighted_panel')
        log(row, accepted, reason, 'initialization', 'weighted_multiscale')
    initialized_calls = calls
    if not any(archive):
        raise RuntimeError('No feasible initialized archive member')

    def addition_options(mask):
        inside = np.flatnonzero(active_flag & ~mask)
        outside = np.flatnonzero(~active_flag & ~mask)
        take_outside = bool(outside.size and (rng.random() < exploration or not inside.size))
        if take_outside:
            return (outside, True)
        return (inside, False)
    next_niche = 0
    generation = 0
    while remaining():
        for _ in range(3):
            parent_niche = next_niche
            next_niche = (next_niche + 1) % 3
            if archive[parent_niche]:
                break
        members = archive[parent_niche]
        parent = members[int(rng.integers(len(members)))]
        mask = parent['mask'].copy()
        selected = np.flatnonzero(mask)
        second = None
        op = str(rng.choice(['add', 'drop', 'swap', 'crossover'], p=[0.2, 0.2, 0.4, 0.2]))
        outside_used = False
        if op == 'crossover':
            mates = [r for bucket in archive for r in bucket if r['candidate_id'] != parent['candidate_id']]
            if mates:
                second = mates[int(rng.integers(len(mates)))]
                mask = np.where(rng.random(p) < 0.5, parent['mask'], second['mask'])
                if not mask.any():
                    union = np.flatnonzero(parent['mask'] | second['mask'])
                    mask[choose(union)[0]] = True
                    op += ':nonempty_repair'
            else:
                op = 'swap'
        if op in ('add', 'drop', 'swap'):
            options, outside_used = addition_options(mask)
            if op == 'drop' and len(selected) == 1:
                op = 'swap' if len(options) else 'noop'
            if op in ('add', 'swap') and (not len(options)):
                op = 'drop' if len(selected) > 1 else 'noop'
            if op == 'drop':
                mask[choose(selected, inverse=True)[0]] = False
                outside_used = False
            elif op == 'add':
                mask[choose(options)[0]] = True
            elif op == 'swap':
                mask[choose(selected, inverse=True)[0]] = False
                mask[choose(options)[0]] = True
            if op == 'noop':
                outside_used = False
        generation += 1
        row = evaluate(mask, parent=parent, second_parent=second, phase='evolution', operator=op, generation=generation, parent_niche=NICHE_NAMES[parent_niche], outside_exploration=outside_used, active_feature_count=int(len(active)))
        delta = _core(row) - _core(parent)
        mean_delta, std_delta = (float(delta.mean()), float(delta.std(ddof=0)))
        min_delta = float(delta.min())
        same_core = bool(np.allclose(_core(row), _core(parent), rtol=0.0, atol=1e-10))
        small_same = same_core and row['n_features'] < parent['n_features']
        guard = mean_delta > max(guard_floor, guard_weight * std_delta) and min_delta >= -guard_worst or small_same
        improves_parent = float(row['score']) > float(parent['score']) + 1e-12
        accepted, removed, reason = insert(row, bool(improves_parent and guard))
        reward = float(np.tanh(mean_delta / (reward_floor + std_delta)))
        added = np.flatnonzero(mask & ~parent['mask'])
        dropped = np.flatnonzero(parent['mask'] & ~mask)
        touched = np.concatenate([added, dropped])
        before = credit[touched].copy()
        targets = np.concatenate([np.full(len(added), reward), np.full(len(dropped), -reward)])
        if feedback and len(touched):
            credit[touched] = np.clip((1.0 - eta) * before + eta * targets, -1.0, 1.0)
        after = credit[touched].copy()
        reward_events.append(dict(fe=int(row.get('fe', calls)), candidate_id=row['candidate_id'], parent_id=parent['candidate_id'], second_parent_id=second['candidate_id'] if second else None, core_fold_deltas=delta.tolist(), core_mean_delta=mean_delta, core_std_delta=std_delta, bounded_reward=reward, added_indices=added.tolist(), removed_indices=dropped.tolist(), updated_indices=touched.tolist(), credit_targets=targets.tolist(), credit_before=before.tolist(), credit_after=after.tolist(), proposal_multiplier_before=np.exp(beta * before).tolist(), proposal_multiplier_after=np.exp(beta * after).tolist(), applied=bool(feedback and len(touched) and (eta > 0)), candidate_accepted=bool(accepted)))
        log(row, accepted, reason, 'evolution', op, parent, second, generation, parent_niche=NICHE_NAMES[parent_niche], outside_exploration=outside_used, guard_passed=bool(guard), score_improved_parent=bool(improves_parent), core_mean_delta=mean_delta, core_std_delta=std_delta, core_min_delta=min_delta, bounded_reward=reward, removed_archive_candidate_id=removed)
    final = current_best()
    resolved = dict(cfg, niche_capacity=capacity, initialization_sizes=list(ANCHORS), initial_candidate_requests=initialized_calls, feedback_eta=eta, feedback_beta=beta, reward_scale_floor=reward_floor, outside_exploration_probability=exploration, guard_weight=guard_weight, guard_mean_floor=guard_floor, guard_worst_loss=guard_worst, operation_probabilities=dict(add=0.2, drop=0.2, swap=0.4, crossover=0.2), active_feature_count=int(len(active)), feedback_enabled=feedback, final_choice='max_common_score_among_accepted_niche_archive_members')
    return dict(method=method, mask=final['mask'].copy(), best_mask=final['mask'].copy(), best_result=final, raw_best_result=_record_summary(raw_best), score=float(final['score']), auc=float(final.get('auc', np.nan)), n_features=int(final['n_features']), seed=int(seed), budget=budget, evaluation_calls=calls, fe_used=start_remaining - int(evaluator.remaining), trace=trace, method_trace=trace, implementation_version=VERSION, implementation_kind='prospective_population_candidate', method_parameters=resolved, archives={NICHE_NAMES[i]: [_record_summary(r) for r in members] for i, members in enumerate(archive)}, anchor_records=anchor_records, archive_events=archive_events, reward_events=reward_events, final_feature_credit=credit.tolist(), final_proposal_weights=weights().tolist(), accepted_after_initialization=sum((r['accepted'] for r in trace if r['phase'] == 'evolution')))
