from __future__ import annotations
import math
import numpy as np
from agents import Council, validate
VERSION = 'semantic-search-1.0'
METHODS = ['M32']
ANCHORS = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 30, 50, 100, 200, 500)
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

def run_search(method, evaluator, n_features, seed, budget, guidance, config=None, knowledge_edges=None, council_context=None):
    if method not in METHODS:
        raise ValueError(f'Unknown population method {method}')
    if not knowledge_edges:
        from population import run_search as base_search
        out = base_search('population_static_v5', evaluator, n_features, seed, budget, guidance, config)
        out.update(method=method, implementation_version=VERSION, knowledge_events=[], knowledge_final_alpha=0.0, knowledge_edge_count=len(knowledge_edges or []), fallback='data_driven_population')
        return out
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
    exploration = float(cfg.get('outside_exploration_probability', 0.1))
    if not 0 <= exploration <= 1:
        raise ValueError('Invalid exploration probability')
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
    ctx = dict(council_context or {})
    if 'log_dir' not in ctx:
        raise ValueError('Council context/log directory required')
    council = ctx.get('council') or Council(ctx['log_dir'], cfg.get('llm_model'))
    modules = {}
    memberships = {}
    for e in knowledge_edges:
        i = int(e['feature_index'])
        m = str(e['module'])
        if not 0 <= i < p:
            raise ValueError('Knowledge feature index out of bounds')
        modules.setdefault(m, set()).add(i)
        memberships.setdefault(i, []).append(m)
    kw = np.zeros(p)
    for m, ids in modules.items():
        for i in ids:
            kw[i] += 1.0 / math.sqrt(len(ids) * len(memberships[i]))
    kw /= max(float(kw.max()), 1e-12)
    knowledge_strength = 0.25
    injection_events = []
    pending_parents = []
    next_checkpoint = 50
    shrink_at = None
    rank_position = np.empty(p, int)
    rank_position[ranking] = np.arange(p)
    symbols = ctx.get('symbols', {})
    archive = [[], [], []]
    trace, reward_events, archive_events = ([], [], [])
    calls = 0
    raw_best = None
    anchor_records = []
    initialized_calls = 0

    def remaining():
        return max(0, min(limit - calls, int(evaluator.remaining)))

    def weights():
        return base_weights * (1.0 + knowledge_strength * kw)

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

    def council_inject():
        nonlocal knowledge_strength, shrink_at
        parents = [max(bucket, key=_key) for bucket in archive if bucket]
        parent_map = {x['candidate_id']: x for x in parents}
        parent_sets = {k: np.flatnonzero(v['mask']).tolist() for k, v in parent_map.items()}
        focus = set((int(i) for i in ranking[:int(cfg.get('context_data_top', 40))]))
        knowledge_rank = np.argsort(-(prior * (0.1 + kw)))
        focus.update((int(i) for i in knowledge_rank[:32]))
        for ids in parent_sets.values():
            focus.update(ids)
        for parent in parents:
            represented = {m for i in np.flatnonzero(parent['mask']) for m in memberships.get(int(i), [])}
            related = {i for m in represented for i in modules[m]}
            focus.update(sorted(related, key=lambda i: rank_position[i])[:16])
        candidates = [dict(feature_index=i, gene_symbol=symbols.get(str(i), symbols.get(i, '')), data_rank=int(rank_position[i]) + 1, training_score=float(prior[i]), modules=memberships.get(i, [])[:4]) for i in sorted(focus, key=lambda i: rank_position[i])]
        redundant = []
        if ctx.get('train_X') is not None:
            for parent in parents:
                ids = np.flatnonzero(parent['mask'])
                if len(ids) < 2:
                    continue
                arr = np.asarray(ctx['train_X'][:, ids], float)
                with np.errstate(invalid='ignore', divide='ignore'):
                    corr = np.corrcoef(arr, rowvar=False)
                for a, b in zip(*np.where(np.triu(np.nan_to_num(np.abs(corr)), 1) > 0.85)):
                    redundant.append(dict(left=int(ids[a]), right=int(ids[b]), abs_correlation=float(abs(corr[a, b]))))
        redundant = sorted(redundant, key=lambda x: -x['abs_correlation'])[:12]
        payload = dict(dataset=ctx.get('dataset', 'development'), disease_context=ctx.get('disease', ''), fe=calls, budget=limit, knowledge_strength=knowledge_strength, candidates=candidates, redundant_pairs=redundant, parents=[dict(parent_id=x['candidate_id'], features=parent_sets[x['candidate_id']], inner_score=float(x['score']), inner_auc=float(x['auc'])) for x in parents], evidence_scope=ctx.get('evidence_scope', 'Hallmark membership; disease-specific interpretation is an unverified proposal hypothesis'))
        event = dict(trigger_fe=calls, weight_before=knowledge_strength, proposals=[], rejected=[])
        try:
            proposed = council.propose(payload)
            legal, rejected = validate(proposed, parent_sets, focus, 4)
            event['rejected'] = rejected
        except Exception as exc:
            event.update(error_type=type(exc).__name__, error='Council unavailable or malformed response; no unlogged heuristic substitute')
            injection_events.append(event)
            return
        improving = 0
        admitted = 0
        for edit in legal:
            if not remaining():
                break
            parent = parent_map[edit['parent_id']]
            mask = np.zeros(p, bool)
            mask[edit['indices']] = True
            row = evaluate(mask, parent=parent, phase='injection', operator='council_edit', generation=generation)
            ni = _niche(row['n_features'])
            bucket = archive[ni]
            duplicate = any((np.array_equal(mask, x['mask']) for x in bucket))
            accepted = False
            if not duplicate:
                if len(bucket) < capacity + 2:
                    bucket.append(row)
                    accepted = True
                else:
                    worst = min(range(len(bucket)), key=lambda j: _key(bucket[j]))
                    if _key(row) > _key(bucket[worst]):
                        bucket[worst] = row
                        accepted = True
            improving += int(_key(row) > _key(parent))
            if accepted:
                admitted += 1
                pending_parents.append(row)
            log(row, accepted, 'temporary_expansion' if accepted else 'duplicate_or_capacity', 'injection', 'council_edit', parent, generation=generation, injection_reason=edit.get('rationale', ''), knowledge_strength=knowledge_strength)
            event['proposals'].append(dict(edit=edit, candidate_id=row['candidate_id'], accepted=accepted, score=float(row['score']), parent_score=float(parent['score']), cache_hit=bool(row.get('cache_hit', False))))
        fraction = improving / max(1, len(event['proposals']))
        knowledge_strength = float(np.clip(0.8 * knowledge_strength + 0.2 * (0.1 + 0.5 * fraction), 0.1, 0.6))
        shrink_at = calls + int(cfg.get('probation_fe', 24))
        event.update(weight_after=knowledge_strength, admitted=admitted, shrink_at=shrink_at)
        injection_events.append(event)
    next_niche = 0
    generation = 0
    while remaining():
        if shrink_at is not None and calls >= shrink_at:
            for bucket in archive:
                bucket.sort(key=_key, reverse=True)
                del bucket[capacity:]
            pending_parents[:] = [x for x in pending_parents if any((x is y for bucket in archive for y in bucket))]
            archive_events.append(dict(fe=calls, reason='competition_contraction', niche_sizes=[len(b) for b in archive]))
            shrink_at = None
        if calls >= next_checkpoint and next_checkpoint <= 150 and (remaining() >= 4):
            council_inject()
            next_checkpoint += 50
            if not remaining():
                break
        for _ in range(3):
            parent_niche = next_niche
            next_niche = (next_niche + 1) % 3
            if archive[parent_niche]:
                break
        members = archive[parent_niche]
        if pending_parents:
            parent = pending_parents.pop(0)
            parent_niche = _niche(parent['n_features'])
        else:
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
        accepted, removed, reason = insert(row, True, reason='niche_fitness_competition')
        log(row, accepted, reason, 'evolution', op, parent, second, generation,
            parent_niche=NICHE_NAMES[parent_niche], outside_exploration=outside_used,
            removed_archive_candidate_id=removed)
    final = current_best()
    resolved = dict(cfg, niche_capacity=capacity, max_niche_capacity=capacity + 2, initialization_sizes=list(ANCHORS), feedback_enabled=False, injection_fe=[50, 100, 150], max_injected_per_event=4, probation_fe=int(cfg.get('probation_fe', 24)), knowledge_weight_initial=0.25, knowledge_weight_bounds=[0.1, 0.6], classifier_policy='unchanged common evaluator', final_choice='max_common_score_among_archive_members')
    return dict(method=method, mask=final['mask'].copy(), best_mask=final['mask'].copy(), best_result=final, raw_best_result=_record_summary(raw_best), score=float(final['score']), auc=float(final.get('auc', np.nan)), n_features=int(final['n_features']), seed=int(seed), budget=budget, evaluation_calls=calls, fe_used=start_remaining - int(evaluator.remaining), trace=trace, method_trace=trace, implementation_version=VERSION, implementation_kind='semantic_population_search', method_parameters=resolved, archives={NICHE_NAMES[i]: [_record_summary(r) for r in members] for i, members in enumerate(archive)}, anchor_records=anchor_records, archive_events=archive_events, knowledge_events=[], knowledge_final_alpha=knowledge_strength, knowledge_edge_count=len(knowledge_edges), injection_events=injection_events, council_calls=council.events, reward_events=reward_events, final_proposal_weights=weights().tolist(), accepted_after_initialization=sum((r['accepted'] for r in trace if r['phase'] == 'evolution')))
