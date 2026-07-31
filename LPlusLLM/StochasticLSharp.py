import json

try:
    from .config import LLMRunConfig
    from .SamplingBasedObservationTree import SamplingBudgetExceeded, StochasticObservationTree
except ImportError:  # pragma: no cover - convenience for standalone test execution
    from config import LLMRunConfig
    from SamplingBasedObservationTree import SamplingBudgetExceeded, StochasticObservationTree

from aalpy.learning_algs.stochastic.DifferenceChecker import (
    AdvancedHoeffdingChecker,
    ChiSquareChecker,
    DifferenceChecker,
    HoeffdingChecker,
)

from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import product
from math import ceil, log, sqrt
import os
from random import Random
from time import perf_counter


class CountingSUL:
    def __init__(self, sul, name='sul'):
        self.sul = sul
        self.name = name
        self.stats = {
            'episodes': 0,
            'steps': 0,
            'pre_calls': 0,
            'post_calls': 0,
        }
        self._sampling_budget_limit = None
        self._sampling_budget_unit = 'episodes'

    def set_sampling_budget_limit(self, limit, unit='episodes'):
        if limit is None:
            self.clear_sampling_budget_limit()
            return
        unit = unit or 'episodes'
        if unit not in ('episodes', 'steps'):
            raise ValueError("sampling budget unit must be either 'episodes' or 'steps'.")
        self._sampling_budget_limit = int(limit)
        self._sampling_budget_unit = unit

    def clear_sampling_budget_limit(self):
        self._sampling_budget_limit = None

    def _budget_value(self):
        if self._sampling_budget_unit == 'steps':
            return int(self.stats.get('steps', 0) or 0)
        return int(self.stats.get('episodes', 0) or 0)

    def _check_sampling_budget(self):
        if self._sampling_budget_limit is None:
            return
        current = self._budget_value()
        if current >= int(self._sampling_budget_limit):
            raise SamplingBudgetExceeded(
                current_samples=current,
                limit=int(self._sampling_budget_limit),
                unit=self._sampling_budget_unit,
            )

    def pre(self):
        if self._sampling_budget_unit == 'episodes':
            self._check_sampling_budget()
        self.stats['episodes'] += 1
        self.stats['pre_calls'] += 1
        return self.sul.pre()

    def step(self, inp):
        if self._sampling_budget_unit == 'steps':
            self._check_sampling_budget()
        self.stats['steps'] += 1
        return self.sul.step(inp)

    def post(self):
        self.stats['post_calls'] += 1
        return self.sul.post()

    def snapshot_stats(self):
        return dict(self.stats)

    def reset_stats(self):
        for key in self.stats:
            self.stats[key] = 0

    def __getattr__(self, name):
        return getattr(self.sul, name)


def _default_debug_logger(payload, line):
    print(line)


def _make_debug_log_fn(enabled=False, logger=None):
    if not enabled:
        return lambda *args, **kwargs: None

    actual_logger = logger or _default_debug_logger

    def _log(event, message=None, component='learner', **data):
        payload = {
            'ts': datetime.now().strftime('%H:%M:%S'),
            'component': component,
            'event': event,
        }
        payload.update(data)
        line = f"[{payload['ts']}] [{component}] {event}"
        if message:
            line += f": {message}"
        if data:
            ordered = ', '.join(f"{k}={payload[k]!r}" for k in data.keys())
            line += f" | {ordered}"
        actual_logger(payload, line)

    return _log


def _snapshot_stats(obj, attr_name='stats'):
    if obj is None:
        return None
    if hasattr(obj, 'snapshot_stats'):
        try:
            snap = obj.snapshot_stats()
            return dict(snap) if snap is not None else None
        except TypeError:
            pass
    stats = getattr(obj, attr_name, None)
    if isinstance(stats, dict):
        return dict(stats)
    return None


def _stats_delta(after, before):
    if after is None:
        return None
    if before is None:
        return dict(after)
    keys = set(after.keys()).union(before.keys())
    return {k: after.get(k, 0) - before.get(k, 0) for k in keys}


def _accumulate_stats(target, delta):
    if delta is None:
        return target
    if target is None:
        return dict(delta)
    for key, value in delta.items():
        target[key] = target.get(key, 0) + value
    return target


def _stats_value(stats, *keys):
    stats = stats or {}
    for key in keys:
        value = stats.get(key)
        if value is not None:
            return int(value or 0)
    return 0


QUERY_STRATEGIES = ('symbolic', 'llm')
MODEL_STRATEGIES = ('symbolic', 'llm')


def _coerce_query_strategy(query_strategy):
    query_strategy = str(query_strategy or 'symbolic').lower().strip()
    if query_strategy not in QUERY_STRATEGIES:
        raise ValueError(
            f"Unsupported query_strategy {query_strategy!r}. "
            f"Choose one of: {', '.join(QUERY_STRATEGIES)}."
        )
    return query_strategy


def _coerce_model_strategy(model_strategy):
    model_strategy = str(model_strategy or 'symbolic').lower().strip()
    if model_strategy not in MODEL_STRATEGIES:
        raise ValueError(
            f"Unsupported model_strategy {model_strategy!r}. "
            f"Choose one of: {', '.join(MODEL_STRATEGIES)}."
        )
    return model_strategy


def _load_input_sequences_json(path, alphabet=None):
    """Load LLMQuerier's final learner-facing input_sequences.json."""
    if path is None:
        return []
    with open(path, 'r', encoding='utf-8') as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        raw_sequences = payload.get('input_sequences', [])
    elif isinstance(payload, list):
        raw_sequences = payload
    else:
        raise ValueError(f'Unsupported input sequence payload in {path!r}.')

    if not isinstance(raw_sequences, list):
        raise ValueError(f'input_sequences must be a list in {path!r}.')

    alphabet_set = set(alphabet or ()) if alphabet is not None else None
    sequences = []
    for raw in raw_sequences:
        if isinstance(raw, str):
            seq = (raw,)
        elif isinstance(raw, (list, tuple)):
            seq = tuple(raw)
        else:
            raise ValueError(f'Invalid input sequence entry {raw!r} in {path!r}.')
        if not seq:
            continue
        if alphabet_set is not None:
            invalid = [inp for inp in seq if inp not in alphabet_set]
            if invalid:
                raise ValueError(
                    f'Input sequence {seq!r} from {path!r} contains symbols outside alphabet: {invalid!r}.'
                )
        sequences.append(seq)
    return sequences


def _make_default_llm_querier(
    *,
    llm_config: LLMRunConfig,
):
    try:
        from .LLMQuerier import LLMQuerier, LLMQuerierConfig
        from .LLMQuerierPromptBuilder import LLMQuerierPromptBuilder, QuerierPromptConstraints
    except ImportError:  # pragma: no cover - standalone execution
        from LLMQuerier import LLMQuerier, LLMQuerierConfig
        from LLMQuerierPromptBuilder import LLMQuerierPromptBuilder, QuerierPromptConstraints

    prompt_builder = LLMQuerierPromptBuilder(
        constraints=QuerierPromptConstraints(
            max_query_length=llm_config.max_query_length,
            remaining_sampling_budget=llm_config.sampling_budget_per_round,
            top_k=llm_config.max_selected_queries,
            allowed_purposes=('complete_branch', 'distinguish_frontier', 'refine_hypothesis'),
            query_mode='rank_only',
            include_full_nodes=bool(llm_config.include_full_nodes),
        )
    )
    return LLMQuerier(
        prompt_builder=prompt_builder,
        config=LLMQuerierConfig(
            max_selected_queries=llm_config.max_selected_queries,
            per_query_default_samples=1,
            complete_branch_use_missing_as_samples=False,
            budget_unit='episodes',
            save_artifacts=True,
            artifact_mode='minimal',
            artifact_dir=llm_config.artifact_dir,
            allow_fallback_symbolic=False,
        ),
    )


def _make_default_llm_modeler(
    *,
    llm_config: LLMRunConfig,
    budget_finalization=False,
):
    try:
        from .LLMModeler import LLMModeler, LLMModelerConfig
        from .LLMModelerPromptBuilder import LLMModelerPromptBuilder, ModelerPromptConstraints
    except ImportError:  # pragma: no cover - standalone execution
        from LLMModeler import LLMModeler, LLMModelerConfig
        from LLMModelerPromptBuilder import LLMModelerPromptBuilder, ModelerPromptConstraints

    return LLMModeler(
        prompt_builder=LLMModelerPromptBuilder(
            constraints=ModelerPromptConstraints(
                include_full_nodes=True if budget_finalization else bool(llm_config.include_full_nodes),
                preserve_basis=not budget_finalization,
                strict_apartness=bool(budget_finalization),
            ),
        ),
        config=LLMModelerConfig(
            max_retries=llm_config.modeler_max_retries,
            save_artifacts=True,
            artifact_mode='minimal',
            artifact_dir=llm_config.modeler_artifact_dir,
            raise_on_failure=False,
        ),
    )


def _is_recoverable_symbolic_model_error(exc):
    message = str(exc)
    recoverable_markers = (
        'Observation tree is not adequate',
        'Unresolved frontier candidates',
        'not identified yet',
        'multiple basis candidates',
        'Multiple basis candidates',
        'Frontier successor has no basis candidate',
        'has no compatible basis candidate',
    )
    return any(marker in message for marker in recoverable_markers)


def _identified_frontier_merge_resolutions(modeler_context):
    resolutions = []
    for item in (modeler_context or {}).get('identified_frontier_basis_map', []) or []:
        try:
            frontier_id = int(item.get('frontier_node_id'))
            basis_id = int(item.get('basis_node_id'))
        except (AttributeError, TypeError, ValueError):
            continue
        resolutions.append({
            'frontier_node_id': frontier_id,
            'action': 'merge',
            'basis_node_id': basis_id,
            'evidence_node_ids': [frontier_id, basis_id],
            'evidence': ['symbolic identified frontier-basis mapping frozen for candidate construction'],
            'risks': ['symbolic_identified_mapping'],
        })
    return resolutions


def _llm_frontier_suffix_depth():
    raw_value = os.environ.get('LLM_FRONTIER_SUFFIX_DEPTH')
    if raw_value in (None, ''):
        return None
    value = int(raw_value)
    if value < 0:
        raise ValueError('LLM_FRONTIER_SUFFIX_DEPTH must be non-negative.')
    return value


def _llm_snapshot_depth(llm_config):
    if _llm_frontier_suffix_depth() is not None:
        return None
    return getattr(llm_config, 'snapshot_max_depth', None)


def _freeze_identified_frontier_merges(tree, modeler_context):
    resolutions = _identified_frontier_merge_resolutions(modeler_context)
    if resolutions:
        tree.apply_frontier_resolutions(resolutions)
    return resolutions


def _run_llm_modeler(
    *,
    llm_modeler,
    tree,
    learning_round,
    state_cls,
    machine_cls,
    learner_stats,
    llm_config,
    log,
):
    if hasattr(tree, 'update_frontier'):
        tree.update_frontier()

    snapshot_max_depth = _llm_snapshot_depth(llm_config)
    snapshot_max_nodes = getattr(llm_config, 'snapshot_max_nodes', None)
    frontier_suffix_depth = _llm_frontier_suffix_depth()
    tree_snapshot = tree.serialize_for_llm(
        max_depth=snapshot_max_depth,
        max_nodes=snapshot_max_nodes,
        frontier_suffix_depth=frontier_suffix_depth,
    )
    modeler_context = _build_frontier_resolution_context(tree)

    used_llm_modeler = bool(modeler_context.get('unresolved_frontier_constraints'))
    if not used_llm_modeler:
        symbolic_resolutions = _freeze_identified_frontier_merges(tree, modeler_context)
        hypothesis = tree.construct_hypothesis(state_cls=state_cls, machine_cls=machine_cls)
        result = type('ResolvedResult', (), {
            'accepted': True,
            'hypothesis': hypothesis,
            'parsed_candidate': {
                'frontier_merges': [],
                'symbolic_frontier_resolutions': symbolic_resolutions,
            },
            'validation_report': type('Report', (), {'accepted': True, 'errors': []})(),
            'error': None,
            'artifacts': {},
        })()
    else:
        result = llm_modeler.synthesize_frontier_resolutions(
            tree=tree,
            tree_snapshot=tree_snapshot,
            modeler_context=modeler_context,
            state_cls=state_cls,
            machine_cls=machine_cls,
            validate=False,
        )

    if used_llm_modeler:
        learner_stats['nr_llm_modeler_calls'] = learner_stats.get('nr_llm_modeler_calls', 0) + 1
        learner_stats['nr_llm_modeler_accepted'] = learner_stats.get('nr_llm_modeler_accepted', 0) + int(bool(result.accepted))
        learner_stats['nr_llm_modeler_rejected'] = learner_stats.get('nr_llm_modeler_rejected', 0) + int(not result.accepted)
        learner_stats['nr_llm_modeler_validation_failures'] = (
            learner_stats.get('nr_llm_modeler_validation_failures', 0)
            + int(not result.validation_report.accepted)
        )

    log(
        'llm_modeler_finished',
        component='learner',
        learning_round=learning_round,
        llm_invoked=used_llm_modeler,
        accepted=result.accepted,
        error=result.error,
        validation_errors=result.validation_report.errors,
        artifacts=result.artifacts,
    )

    if not result.accepted or result.hypothesis is None:
        raise RuntimeError(
            'LLMModeler failed to produce accepted frontier resolutions. '
            f'Errors: {result.validation_report.errors}. '
            f'Artifacts: {result.artifacts}'
        )

    return llm_modeler, result, result.hypothesis


def _run_llm_budget_finalizer(
    *,
    llm_modeler,
    tree,
    learning_round,
    state_cls,
    machine_cls,
    learner_stats,
    llm_config,
    log,
):
    if hasattr(tree, 'update_frontier'):
        tree.update_frontier()

    if llm_modeler is None:
        llm_modeler = _make_default_llm_modeler(
            llm_config=llm_config,
            budget_finalization=True,
        )
    else:
        constraints = getattr(getattr(llm_modeler, 'prompt_builder', None), 'constraints', None)
        if constraints is not None:
            if hasattr(constraints, 'include_full_nodes'):
                constraints.include_full_nodes = True
            if hasattr(constraints, 'preserve_basis'):
                constraints.preserve_basis = False
            if hasattr(constraints, 'strict_apartness'):
                constraints.strict_apartness = True

    snapshot_max_depth = _llm_snapshot_depth(llm_config)
    snapshot_max_nodes = getattr(llm_config, 'snapshot_max_nodes', None)
    frontier_suffix_depth = _llm_frontier_suffix_depth()
    tree_snapshot = tree.serialize_for_llm(
        max_depth=snapshot_max_depth,
        max_nodes=snapshot_max_nodes,
        frontier_suffix_depth=frontier_suffix_depth,
    )
    serialized_node_ids = {
        node.get('id')
        for node in (tree_snapshot.get('nodes') or [])
        if isinstance(node, dict) and node.get('id') is not None
    }
    apart_node_pairs = [
        pair for pair in _collect_apart_node_pairs_for_modeler(tree)
        if pair.get('node_id_1') in serialized_node_ids and pair.get('node_id_2') in serialized_node_ids
    ]
    modeler_context = _build_frontier_resolution_context(tree)
    modeler_context.setdefault('meta', {})
    modeler_context['meta'].update({
        'root_id': getattr(getattr(tree, 'root', None), 'id', None),
        'alphabet': list(getattr(tree, 'alphabet', []) or []),
        'budget_finalization': True,
    })
    modeler_context['apart_node_pairs'] = apart_node_pairs
    modeler_context['budget_finalization'] = {
        'mode': 'frozen_tree_state_merge',
        'sampling_allowed': False,
        'llm_output': 'frontier_merges_only',
        'symbolic_default_for_unmerged_unresolved_frontier': 'chaos',
        'symbolic_construction': 'construct_hypothesis_from_resolved_frontier',
        'validation': 'legal_symbols_only_no_observation_counterexample',
    }

    used_llm_modeler = bool(modeler_context.get('unresolved_frontier_constraints'))
    if not used_llm_modeler:
        symbolic_resolutions = _freeze_identified_frontier_merges(tree, modeler_context)
        hypothesis = tree.construct_hypothesis(state_cls=state_cls, machine_cls=machine_cls)
        result = type('ResolvedResult', (), {
            'accepted': True,
            'hypothesis': hypothesis,
            'parsed_candidate': {
                'frontier_merges': [],
                'symbolic_frontier_resolutions': symbolic_resolutions,
            },
            'validation_report': type('Report', (), {
                'accepted': True,
                'errors': [],
                'warnings': [],
                'checked_constraints': ['symbolic_hypothesis_construction'],
            })(),
            'error': None,
            'artifacts': {},
        })()
    else:
        result = llm_modeler.synthesize_frontier_resolutions(
            tree=tree,
            tree_snapshot=tree_snapshot,
            modeler_context=modeler_context,
            state_cls=state_cls,
            machine_cls=machine_cls,
            validate=True,
        )

    if used_llm_modeler:
        learner_stats['nr_llm_modeler_calls'] = learner_stats.get('nr_llm_modeler_calls', 0) + 1
        learner_stats['nr_llm_modeler_accepted'] = learner_stats.get('nr_llm_modeler_accepted', 0) + int(bool(result.accepted))
        learner_stats['nr_llm_modeler_rejected'] = learner_stats.get('nr_llm_modeler_rejected', 0) + int(not result.accepted)
        learner_stats['nr_llm_modeler_validation_failures'] = (
            learner_stats.get('nr_llm_modeler_validation_failures', 0)
            + int(not result.validation_report.accepted)
        )

    log(
        'llm_budget_finalizer_finished',
        component='learner',
        learning_round=learning_round,
        llm_invoked=used_llm_modeler,
        accepted=result.accepted,
        error=result.error,
        validation_errors=result.validation_report.errors,
        validation_warnings=result.validation_report.warnings,
        frontier_merge_count=len((getattr(result, 'parsed_candidate', None) or {}).get('frontier_merges', []) or []),
        symbolic_frontier_resolution_count=len((getattr(result, 'parsed_candidate', None) or {}).get('symbolic_frontier_resolutions', []) or []),
        artifacts=result.artifacts,
    )

    if not result.accepted or result.hypothesis is None:
        raise RuntimeError(
            'LLMModeler failed to produce accepted budget-final merge resolutions. '
            f'Errors: {result.validation_report.errors}. '
            f'Warnings: {result.validation_report.warnings}. '
            f'Artifacts: {result.artifacts}'
        )

    return llm_modeler, result, result.hypothesis


def _collect_apart_node_pairs_for_modeler(tree):
    pairs = []
    seen = set()
    witness_cache = getattr(tree, 'witness_cache', {}) or {}
    for key, witness in witness_cache.items():
        if not witness or not isinstance(key, tuple) or len(key) < 2:
            continue
        left, right = key[0], key[1]
        if not isinstance(left, int) or not isinstance(right, int):
            continue
        ordered = tuple(sorted((left, right)))
        if ordered in seen:
            continue
        seen.add(ordered)
        pairs.append({
            'node_id_1': ordered[0],
            'node_id_2': ordered[1],
        })
    return pairs


def _build_frontier_resolution_context(tree):
    apart_pairs = _collect_apart_node_pairs_for_modeler(tree)
    apart_set = {
        tuple(sorted((int(item['node_id_1']), int(item['node_id_2']))))
        for item in apart_pairs
        if item.get('node_id_1') is not None and item.get('node_id_2') is not None
    }
    identified = []
    unresolved = []
    for frontier_node, candidates in sorted(
        getattr(tree, 'frontier_to_basis_dict', {}).items(),
        key=lambda item: item[0].id,
    ):
        candidate_ids = sorted(candidate.id for candidate in candidates)
        if len(candidate_ids) == 1 and not tree._should_use_chaos_for_frontier(frontier_node):
            identified.append({
                'frontier_node_id': frontier_node.id,
                'basis_node_id': candidate_ids[0],
            })
            continue
        forbidden = sorted(
            candidate_id
            for candidate_id in candidate_ids
            if tuple(sorted((frontier_node.id, candidate_id))) in apart_set
        )
        unresolved.append({
            'frontier_node_id': frontier_node.id,
            'candidate_basis_ids': candidate_ids,
            'forbidden_basis_ids': forbidden,
            'apart_pairs': [
                list(pair)
                for pair in sorted(apart_set)
                if frontier_node.id in pair
            ],
        })
    return {
        'meta': {
            'root_id': getattr(getattr(tree, 'root', None), 'id', None),
            'alphabet': list(getattr(tree, 'alphabet', []) or []),
        },
        'identified_frontier_basis_map': identified,
        'unresolved_frontier_constraints': unresolved,
        'apart_node_pairs': apart_pairs,
    }


def _save_llm_serialization_payloads(tree, tree_snapshot, querier_context):
    """Persist the exact in-memory LLM snapshot/context used for this round."""
    serialized_tree_path = None
    querier_context_path = None

    if hasattr(tree, '_save_json_document') and hasattr(tree, '_default_serialized_tree_path'):
        serialized_tree_path = tree._save_json_document(
            tree_snapshot,
            tree._default_serialized_tree_path(),
        )
    elif hasattr(tree, 'save_serialized_for_llm'):
        serialized_tree_path = tree.save_serialized_for_llm()

    if hasattr(tree, '_save_json_document') and hasattr(tree, '_default_querier_context_path'):
        querier_context_path = tree._save_json_document(
            querier_context,
            tree._default_querier_context_path(),
        )
    elif hasattr(tree, 'save_querier_context_for_llm'):
        querier_context_path = tree.save_querier_context_for_llm()

    return serialized_tree_path, querier_context_path


def _run_llm_querier_from_input_sequences_file(
    *,
    llm_querier,
    tree,
    sul,
    alphabet,
    learning_round,
    llm_config: LLMRunConfig,
    phase_sul_stats,
    learner_stats,
    log,
):
    """
    Run LLMQuerier once, then execute the produced input_sequences.json.

    The LLMQuerier is used only as a query planner. The learner intentionally
    reads the final file artifact and performs sampling itself so the integration
    point is explicit and easy to replace.
    """
    if hasattr(tree, 'update_frontier'):
        tree.update_frontier()

    tree_snapshot = tree.serialize_for_llm(
        max_depth=_llm_snapshot_depth(llm_config),
        max_nodes=llm_config.snapshot_max_nodes,
        frontier_suffix_depth=_llm_frontier_suffix_depth(),
    )
    querier_context = tree.serialize_querier_context_for_llm(
        max_depth=_llm_snapshot_depth(llm_config),
        max_nodes=llm_config.snapshot_max_nodes,
        frontier_suffix_depth=_llm_frontier_suffix_depth(),
    )

    serialized_tree_path, querier_context_path = _save_llm_serialization_payloads(
        tree,
        tree_snapshot,
        querier_context,
    )

    prompt_package = llm_querier.build_prompt_package(
        tree_snapshot,
        querier_context,
        remaining_sampling_budget=llm_config.sampling_budget_per_round,
        save_prompt=True,
    )

    if not getattr(prompt_package, 'candidate_pool', None):
        learner_stats['nr_llm_querier_skipped_empty_pool'] = learner_stats.get('nr_llm_querier_skipped_empty_pool', 0) + 1
        log(
            'llm_querier_skipped',
            component='learner',
            learning_round=learning_round,
            reason='empty_candidate_pool',
            serialized_tree_path=str(serialized_tree_path) if serialized_tree_path is not None else None,
            querier_context_path=str(querier_context_path) if querier_context_path is not None else None,
        )
        return llm_querier, None, []

    # Planning call only. Do not let LLMQuerier execute traces directly; the
    # learner consumes the file artifact below.
    result = llm_querier.plan_next_queries(
        tree_snapshot=tree_snapshot,
        querier_context=querier_context,
        remaining_sampling_budget=llm_config.sampling_budget_per_round,
        prompt_package=prompt_package,
    )

    input_sequences_path = (result.artifacts or {}).get('input_sequences')
    if not input_sequences_path:
        raise RuntimeError('LLMQuerier did not produce an input_sequences artifact.')

    input_sequences = _load_input_sequences_json(input_sequences_path, alphabet=alphabet)
    if hasattr(tree, 'set_guided_query_sequences'):
        tree.set_guided_query_sequences(
            input_sequences,
            max_query_length=llm_config.max_query_length,
        )
    sul_before = _snapshot_stats(sul)
    if hasattr(tree, 'execute_input_sequences'):
        executed_traces = tree.execute_input_sequences(
            input_sequences,
            num_samples=llm_config.samples_per_sequence,
        )
    else:
        executed_traces = []
        for inputs in input_sequences:
            for _ in range(max(1, int(llm_config.samples_per_sequence or 1))):
                executed_traces.append(tree._sample_query_once(tuple(inputs)))

    phase_sul_stats['llm_querier'] = _accumulate_stats(
        phase_sul_stats.get('llm_querier'),
        _stats_delta(_snapshot_stats(sul), sul_before),
    )

    learner_stats['nr_llm_querier_calls'] = learner_stats.get('nr_llm_querier_calls', 0) + 1
    learner_stats['nr_llm_input_sequences'] = learner_stats.get('nr_llm_input_sequences', 0) + len(input_sequences)
    learner_stats['nr_llm_executed_traces'] = learner_stats.get('nr_llm_executed_traces', 0) + len(executed_traces)

    log(
        'llm_querier_finished',
        component='learner',
        learning_round=learning_round,
        input_sequences=[tuple(seq) for seq in input_sequences],
        executed_traces=len(executed_traces),
        input_sequences_path=input_sequences_path,
        serialized_tree_path=str(serialized_tree_path) if serialized_tree_path is not None else None,
        querier_context_path=str(querier_context_path) if querier_context_path is not None else None,
    )
    return llm_querier, result, executed_traces


@dataclass
class StochasticLSharpInternalConfig:
    """Low-level sampling / apartness knobs kept out of the public API."""

    apartness_n_c: int | None = None
    completion_max_attempts: int = 300
    identification_budget: int | None = None
    eq_distribution_alpha: float = 0.01
    eq_probability_tolerance: float = 0.05
    eq_distribution_min_total: int | None = None
    store_failed_prefixes: bool = True
    completion_rollout_len: int = 1
    chaos_reach_threshold: float = 0.01
    chaos_adaptive: bool = False
    chaos_max_reach_threshold: float | None = None
    chaos_growth_factor: float = 2.0
    chaos_stagnation_rounds: int = 2
    apartness_max_depth: int | None = None
    apartness_max_nodes_visited: int | None = None
    apartness_max_queue_size: int | None = None

    def resolved(self, n_c: int):
        resolved_n_c = max(0, int(n_c or 0))
        if self.apartness_n_c is None:
            self.apartness_n_c = resolved_n_c
        else:
            self.apartness_n_c = max(0, int(self.apartness_n_c))
        self.completion_max_attempts = None if self.completion_max_attempts is None else int(self.completion_max_attempts)
        self.identification_budget = None if self.identification_budget is None else max(1, int(self.identification_budget))
        self.eq_distribution_alpha = float(self.eq_distribution_alpha or 0.01)
        if not (0 < self.eq_distribution_alpha < 1):
            raise ValueError('eq_distribution_alpha must lie in (0, 1).')
        self.eq_probability_tolerance = max(0.0, float(self.eq_probability_tolerance or 0.0))
        self.eq_distribution_min_total = (
            None
            if self.eq_distribution_min_total is None
            else max(0, int(self.eq_distribution_min_total))
        )
        self.store_failed_prefixes = bool(self.store_failed_prefixes)
        self.completion_rollout_len = max(1, int(self.completion_rollout_len or 1))
        self.chaos_reach_threshold = max(0.0, float(self.chaos_reach_threshold or 0.0))
        self.chaos_adaptive = bool(self.chaos_adaptive)
        if self.chaos_max_reach_threshold is None:
            self.chaos_max_reach_threshold = self.chaos_reach_threshold
        else:
            self.chaos_max_reach_threshold = max(self.chaos_reach_threshold, float(self.chaos_max_reach_threshold or 0.0))
        self.chaos_growth_factor = max(1.0, float(self.chaos_growth_factor or 1.0))
        self.chaos_stagnation_rounds = max(1, int(self.chaos_stagnation_rounds or 1))
        if self.apartness_max_depth is not None:
            self.apartness_max_depth = int(self.apartness_max_depth)
        if self.apartness_max_nodes_visited is not None:
            self.apartness_max_nodes_visited = int(self.apartness_max_nodes_visited)
        if self.apartness_max_queue_size is not None:
            self.apartness_max_queue_size = int(self.apartness_max_queue_size)
        return self


def _coerce_internal_config(internal_config=None, *, n_c=0, legacy_kwargs=None):
    legacy_kwargs = dict(legacy_kwargs or {})
    legacy_internal_keys = (
        'apartness_n_c',
        'completion_max_attempts',
        'identification_budget',
        'eq_distribution_alpha',
        'eq_probability_tolerance',
        'eq_distribution_min_total',
        'store_failed_prefixes',
        'completion_rollout_len',
        'chaos_reach_threshold',
        'chaos_adaptive',
        'chaos_max_reach_threshold',
        'chaos_growth_factor',
        'chaos_stagnation_rounds',
        'apartness_max_depth',
        'apartness_max_nodes_visited',
        'apartness_max_queue_size',
    )
    legacy_overrides = {
        key: legacy_kwargs.pop(key)
        for key in legacy_internal_keys
        if key in legacy_kwargs
    }

    if internal_config is None:
        config = StochasticLSharpInternalConfig()
    elif isinstance(internal_config, StochasticLSharpInternalConfig):
        config = StochasticLSharpInternalConfig(**asdict(internal_config))
    elif isinstance(internal_config, dict):
        config = StochasticLSharpInternalConfig(**internal_config)
    else:
        raise TypeError('internal_config must be None, a dict, or StochasticLSharpInternalConfig.')

    for key, value in legacy_overrides.items():
        setattr(config, key, value)

    config.resolved(n_c=n_c)
    return config, legacy_kwargs


def _resolve_identification_budget(n_resample, internal_config):
    resample_hits = max(1, int(n_resample or 1))
    configured = getattr(internal_config, 'identification_budget', None)
    if configured is not None:
        return max(1, int(configured))
    chaos_threshold = float(getattr(internal_config, 'chaos_reach_threshold', 0.0) or 0.0)
    if chaos_threshold > 0.0:
        return max(resample_hits, int(ceil(resample_hits / chaos_threshold)))
    return max(
        resample_hits,
        int(getattr(internal_config, 'completion_max_attempts', None) or resample_hits),
    )


def _resolve_resample_hits(n_resample, n_c):
    if n_resample is None:
        return 5
    return max(1, int(n_resample or 1))


def _resolve_eq_distribution_min_total(n_resample, internal_config):
    configured = getattr(internal_config, 'eq_distribution_min_total', None)
    if configured is not None:
        return max(0, int(configured))
    resample_hits = max(1, int(n_resample or 1))
    return max(20, 2 * resample_hits)


class StatisticalEqOracle:
    """
    Statistical equivalence oracle for stochastic Mealy hypotheses.

    This variant is tree-aware:
      - it can reuse a previously found io-trace counterexample
      - it can inspect the current observation tree for local support/distribution
        mismatches against the hypothesis
      - during random/exhaustive fallback sampling, sampled episodes are
        inserted into the attached learning tree immediately
    """

    def __init__(
        self,
        alphabet,
        sul,
        num_samples=200,
        max_cex_len=5,
        min_cex_len=1,
        alpha=0.05,
        input_sequences=None,
        random_search=False,
        random_trace_count=150,
        random_trace_len=None,
        random_seed=None,
        unique_random_sequences=True,
        eq_distribution_alpha=0.01,
        eq_probability_tolerance=0.05,
        eq_distribution_min_total=None,
        insert_samples_into_tree=True,
        debug=False,
        logger=None,
    ):
        if alphabet is None:
            raise ValueError('alphabet must not be None.')
        if sul is None:
            raise ValueError('sul must not be None.')
        if num_samples <= 0:
            raise ValueError('num_samples must be positive.')
        if min_cex_len <= 0:
            raise ValueError('min_cex_len must be positive.')
        if max_cex_len < min_cex_len:
            raise ValueError('max_cex_len must be >= min_cex_len.')
        if not (0 < alpha < 1):
            raise ValueError('alpha must lie in (0, 1).')
        if not (0 < float(eq_distribution_alpha) < 1):
            raise ValueError('eq_distribution_alpha must lie in (0, 1).')

        self.alphabet = tuple(alphabet)
        self.sul = sul
        self.num_samples = int(num_samples)
        self.max_cex_len = int(max_cex_len)
        self.min_cex_len = int(min_cex_len)
        self.alpha = float(alpha)
        self.eq_distribution_alpha = float(eq_distribution_alpha)
        self.eq_probability_tolerance = max(0.0, float(eq_probability_tolerance or 0.0))
        self.eq_distribution_min_total = (
            None
            if eq_distribution_min_total is None
            else max(0, int(eq_distribution_min_total))
        )
        self.eq_distribution_refinement_hits = None
        self.eq_distribution_identification_budget = None
        # EQ samples are real learner-side observations. They must update the
        # observation tree regardless of query/model strategy; the parameter is
        # kept only for backward-compatible call sites.
        self.insert_samples_into_tree = True
        self.input_sequences = tuple(tuple(seq) for seq in input_sequences) if input_sequences is not None else None
        self.random_search = bool(random_search)
        self.random_trace_count = int(random_trace_count)
        self.random_trace_len = int(random_trace_len) if random_trace_len is not None else self.max_cex_len
        self.unique_random_sequences = bool(unique_random_sequences)
        self._random = Random(random_seed)
        if self.random_trace_count <= 0:
            raise ValueError('random_trace_count must be positive.')
        if self.random_trace_len <= 0:
            raise ValueError('random_trace_len must be positive.')
        self.last_result = None
        self.last_cex = None
        self.debug = bool(debug)
        self._logger = logger
        self.tree = None
        self.diff_checker = None
        self.stats = {
            'find_cex_calls': 0,
            'candidate_sequences_checked': 0,
            'sampled_traces': 0,
            'sampled_steps': 0,
            'counterexamples_found': 0,
            'random_sequences_generated': 0,
            'tree_cex_found': 0,
            'tree_nodes_checked': 0,
            'tree_distribution_cells_under_sampled': 0,
            'find_cex_time_sec': 0.0,
            'last_find_cex_time_sec': 0.0,
        }

    def set_debug(self, enabled=True, logger=None):
        self.debug = bool(enabled)
        if logger is not None:
            self._logger = logger
        return self

    def attach_tree(self, tree, diff_checker=None):
        self.tree = tree
        if diff_checker is not None:
            self.diff_checker = diff_checker
        return self

    def configure_distribution_ce(
        self,
        *,
        alpha=None,
        probability_tolerance=None,
        min_total=None,
        refinement_hits=None,
        identification_budget=None,
    ):
        if alpha is not None:
            alpha = float(alpha)
            if not (0 < alpha < 1):
                raise ValueError('eq_distribution_alpha must lie in (0, 1).')
            self.eq_distribution_alpha = alpha
        if probability_tolerance is not None:
            self.eq_probability_tolerance = max(0.0, float(probability_tolerance or 0.0))
        if min_total is not None:
            self.eq_distribution_min_total = max(0, int(min_total))
        if refinement_hits is not None:
            self.eq_distribution_refinement_hits = max(1, int(refinement_hits or 1))
        if identification_budget is not None:
            self.eq_distribution_identification_budget = max(1, int(identification_budget or 1))
        return self

    def _log(self, event, **data):
        if not self.debug:
            return
        log_fn = _make_debug_log_fn(True, self._logger)
        log_fn(event, component='eq_oracle', **data)

    def _candidate_count_estimate(self):
        if self.input_sequences is not None:
            return len(self.input_sequences)
        if self.random_search:
            if not self.unique_random_sequences:
                return self.random_trace_count
            return min(self.random_trace_count, len(self.alphabet) ** self._effective_random_trace_len())
        total = 0
        for length in range(self.min_cex_len, self.max_cex_len + 1):
            total += len(self.alphabet) ** length
        return total

    @staticmethod
    def _short_inputs_repr(inputs, limit=8):
        inputs = tuple(inputs)
        if len(inputs) <= limit:
            return inputs
        return inputs[:limit] + ('...',)

    def _effective_random_trace_len(self):
        # In random EQ, eq_max_cex_len is the maximum searchable counterexample
        # length. Candidate query sequences must therefore never exceed it.
        return min(self.random_trace_len, self.max_cex_len)

    def _io_trace_within_cex_limit(self, io_trace):
        if not io_trace:
            return True
        return (len(io_trace) // 2) <= self.max_cex_len

    def _inputs_within_cex_limit(self, inputs):
        if inputs is None:
            return True
        return len(tuple(inputs)) <= self.max_cex_len

    def _should_log_candidate(self, candidate_index, total_candidates):
        if candidate_index <= 3:
            return True
        if total_candidates is not None and candidate_index == total_candidates:
            return True
        return candidate_index % 10 == 0

    def _draw_random_input_sequences(self):
        trace_len = self._effective_random_trace_len()
        count = self.random_trace_count

        if not self.unique_random_sequences:
            sequences = [
                tuple(self._random.choice(self.alphabet) for _ in range(trace_len))
                for _ in range(count)
            ]
            self.stats['random_sequences_generated'] += len(sequences)
            return tuple(sequences)

        max_unique = len(self.alphabet) ** trace_len
        target = min(count, max_unique)
        seen = set()
        sequences = []
        while len(sequences) < target:
            seq = tuple(self._random.choice(self.alphabet) for _ in range(trace_len))
            if seq in seen:
                continue
            seen.add(seq)
            sequences.append(seq)

        self.stats['random_sequences_generated'] += len(sequences)
        return tuple(sequences)

    def _candidate_input_sequences(self):
        if self.input_sequences is not None:
            for seq in self.input_sequences:
                yield seq
            return

        if self.random_search:
            for seq in self._draw_random_input_sequences():
                yield seq
            return

        for length in range(self.min_cex_len, self.max_cex_len + 1):
            for seq in product(self.alphabet, repeat=length):
                yield seq

    def _sample_one_io_trace(self, inputs, *, insert_into_tree=None):
        observed = []
        self.sul.pre()
        try:
            for inp in inputs:
                out = self.sul.step(inp)
                observed.extend([inp, out])
        finally:
            self.sul.post()
        io_trace = tuple(observed)
        self.stats['sampled_traces'] += 1
        self.stats['sampled_steps'] += len(inputs)
        if self.tree is not None and io_trace:
            self.tree.insert_trace(io_trace)
        return io_trace

    def _sample_sul_distribution(self, inputs):
        counts = defaultdict(int)
        sampled_io_traces = []
        log_progress = self.debug and self.num_samples >= 20
        progress_step = max(10, self.num_samples // 5) if self.num_samples > 0 else 10

        for sample_idx in range(1, self.num_samples + 1):
            io_trace = self._sample_one_io_trace(inputs)
            counts[tuple(io_trace[1::2])] += 1
            sampled_io_traces.append(io_trace)
            if log_progress and (sample_idx == 1 or sample_idx == self.num_samples or sample_idx % progress_step == 0):
                self._log(
                    'sampling_progress',
                    inputs=self._short_inputs_repr(inputs),
                    sample_idx=sample_idx,
                    num_samples=self.num_samples,
                    distinct_outputs=len(counts),
                )
        return dict(counts), sampled_io_traces

    def _sampled_trace_counterexample(self, hypothesis, sampled_io_traces):
        for io_trace in sampled_io_traces:
            if not io_trace:
                continue
            if self._get_hypothesis_successor_from_io_trace(hypothesis, io_trace) is not None:
                continue
            already_observed = (
                self.tree is not None
                and hasattr(self.tree, 'get_successor')
                and self.tree.get_successor(tuple(io_trace)) is not None
            )
            return {
                'io_trace': tuple(io_trace),
                'source': 'sampled_trace',
                'already_observed': already_observed,
                'diff_kind': 'unsupported_output',
            }
        return None

    def _hypothesis_output_distribution(self, hyp_state, inp):
        output_prob = defaultdict(float)
        output_to_state = {}
        for target_state, output, probability in getattr(hyp_state, 'transitions', {}).get(inp, []):
            if probability <= 0:
                continue
            output_prob[output] += probability
            output_to_state.setdefault(output, target_state)
        return dict(output_prob), output_to_state

    def _empirical_vs_hypothesis_different(self, observed_counts, hypothesis_probs):
        total = sum(observed_counts.values()) if observed_counts else 0
        if total <= 0:
            return False, {
                'threshold': None,
                'max_gap': 0.0,
                'witness_output': None,
            }

        support = set(observed_counts.keys()).union(hypothesis_probs.keys())
        nr_events = max(1, len(support))
        alpha = float(getattr(self, 'eq_distribution_alpha', self.alpha) or self.alpha)
        hoeffding_threshold = sqrt(log(2 * nr_events / alpha) / (2 * total))
        tolerance = max(0.0, float(getattr(self, 'eq_probability_tolerance', 0.0) or 0.0))
        threshold = max(hoeffding_threshold, tolerance)

        best = None
        for out in support:
            empirical = observed_counts.get(out, 0) / total
            expected = hypothesis_probs.get(out, 0.0)
            gap = abs(empirical - expected)
            item = {
                'output': out,
                'empirical_probability': empirical,
                'hypothesis_probability': expected,
                'observed_count': observed_counts.get(out, 0),
                'gap': gap,
            }
            if best is None:
                best = item
                continue
            # Prefer an observed output for the concrete I/O counterexample
            # when two outputs explain the same probability gap.
            best_key = (
                best['gap'],
                int(best['observed_count'] > 0),
                best['observed_count'],
                repr(best['output']),
            )
            item_key = (
                item['gap'],
                int(item['observed_count'] > 0),
                item['observed_count'],
                repr(item['output']),
            )
            if item_key > best_key:
                best = item

        if best is None:
            return False, {
                'threshold': threshold,
                'hoeffding_threshold': hoeffding_threshold,
                'probability_tolerance': tolerance,
                'max_gap': 0.0,
                'witness_output': None,
            }

        return best['gap'] > threshold, {
            'threshold': threshold,
            'hoeffding_threshold': hoeffding_threshold,
            'probability_tolerance': tolerance,
            'max_gap': best['gap'],
            'witness_output': best['output'],
            'empirical_probability': best['empirical_probability'],
            'hypothesis_probability': best['hypothesis_probability'],
            'observed_count': best['observed_count'],
            'total_observed': total,
        }

    def _get_hypothesis_successor_from_io_trace(self, hypothesis, io_trace):
        current_state = getattr(hypothesis, 'initial_state', None)
        if current_state is None:
            return None
        for inp, out in zip(io_trace[0::2], io_trace[1::2]):
            matched = None
            for target_state, output, probability in getattr(current_state, 'transitions', {}).get(inp, []):
                if output == out and probability > 0:
                    matched = target_state
                    break
            if matched is None:
                return None
            current_state = matched
        return current_state

    def _reuse_last_cex_if_valid(self, hypothesis):
        if not self.last_cex:
            return None
        io_trace = None
        if isinstance(self.last_cex, dict) and 'io_trace' in self.last_cex:
            io_trace = tuple(self.last_cex['io_trace'])
        if not io_trace:
            return None
        if not self._io_trace_within_cex_limit(io_trace):
            return None
        if self._get_hypothesis_successor_from_io_trace(hypothesis, io_trace) is None:
            return self.last_cex
        return None

    def _tree_eq_completeness_threshold(self):
        if self.tree is None:
            return 0
        threshold = getattr(self.tree, '_active_n_c', 0)
        try:
            threshold = int(threshold or 0)
        except (TypeError, ValueError):
            threshold = 0
        return max(0, threshold)

    def _tree_node_complete_for_eq(self, tree_node, threshold):
        if tree_node is None or self.tree is None:
            return False
        if threshold <= 0:
            return True
        for inp in self.alphabet:
            if not self.tree.is_complete(tree_node, inp, threshold):
                return False
        return True

    def _eq_distribution_min_total(self):
        threshold = getattr(self, 'eq_distribution_min_total', None)
        if threshold is None:
            return 0
        try:
            return max(0, int(threshold))
        except (TypeError, ValueError):
            return 0

    def _distribution_cell_complete_for_eq(self, total):
        min_total = self._eq_distribution_min_total()
        return min_total <= 0 or int(total or 0) >= min_total

    def _tree_based_counterexample(self, hypothesis, *, reason='tree_scan', candidate_index=None, inputs=None):
        if self.tree is None or self.diff_checker is None:
            return None

        root = getattr(self.tree, 'root', None)
        hyp_root = getattr(hypothesis, 'initial_state', None)
        if root is None or hyp_root is None:
            return None

        completion_threshold = self._tree_eq_completeness_threshold()
        max_searchable_len = self.max_cex_len
        if completion_threshold > 0 and not self._tree_node_complete_for_eq(root, completion_threshold):
            self._log(
                'tree_scan_skipped',
                reason=reason,
                candidate_index=candidate_index,
                inputs=self._short_inputs_repr(inputs) if inputs is not None else None,
                completion_threshold=completion_threshold,
                max_searchable_len=max_searchable_len,
                skip_reason='root_not_complete',
            )
            return None

        scan_t0 = perf_counter()
        tree_nodes_before = self.stats['tree_nodes_checked']
        self._log(
            'tree_scan_start',
            reason=reason,
            candidate_index=candidate_index,
            inputs=self._short_inputs_repr(inputs) if inputs is not None else None,
            completion_threshold=completion_threshold,
            max_searchable_len=max_searchable_len,
        )

        queue = deque([(root, hyp_root, tuple())])
        visited = set()

        while queue:
            tree_node, hyp_state, io_trace = queue.popleft()
            if tree_node is None or hyp_state is None:
                continue
            state_pair = (getattr(tree_node, 'id', id(tree_node)), id(hyp_state))
            if state_pair in visited:
                continue
            visited.add(state_pair)

            if completion_threshold > 0 and not self._tree_node_complete_for_eq(tree_node, completion_threshold):
                continue

            current_len_steps = len(io_trace) // 2
            if current_len_steps >= max_searchable_len:
                continue

            self.stats['tree_nodes_checked'] += 1

            for inp in self.alphabet:
                if completion_threshold > 0 and not self.tree.is_complete(tree_node, inp, completion_threshold):
                    continue

                c_tree = self.tree.get_output_frequencies(tree_node, inp)
                if not c_tree:
                    continue
                cell_total = sum(c_tree.values())
                c_hyp, output_to_state = self._hypothesis_output_distribution(hyp_state, inp)

                unsupported = [out for out in c_tree.keys() if c_hyp.get(out, 0.0) <= 0.0]
                if unsupported:
                    out = max(unsupported, key=lambda o: (c_tree.get(o, 0), repr(o)))
                    result = {
                        'io_trace': io_trace + (inp, out),
                        'source': 'observation_tree_scan',
                        'already_observed': True,
                        'diff_kind': 'unsupported_output',
                        'prefix_io': io_trace,
                        'input': inp,
                        'output': out,
                    }
                    self._log(
                        'tree_scan_counterexample',
                        reason=reason,
                        candidate_index=candidate_index,
                        inputs=self._short_inputs_repr(inputs) if inputs is not None else None,
                        witness_io_trace=result['io_trace'],
                        nodes_checked_delta=self.stats['tree_nodes_checked'] - tree_nodes_before,
                        max_searchable_len=max_searchable_len,
                        elapsed_sec=round(perf_counter() - scan_t0, 6),
                        diff_kind='unsupported_output',
                    )
                    return result

                is_diff, diagnostics = self._empirical_vs_hypothesis_different(c_tree, c_hyp)
                if c_hyp and is_diff:
                    if not self._distribution_cell_complete_for_eq(cell_total):
                        self.stats['tree_distribution_cells_under_sampled'] += 1
                        self._log(
                            'tree_scan_distribution_cell_incomplete',
                            reason=reason,
                            candidate_index=candidate_index,
                            inputs=self._short_inputs_repr(inputs) if inputs is not None else None,
                            witness_io_trace=io_trace + (inp, diagnostics.get('witness_output')),
                            node_id=getattr(tree_node, 'id', None),
                            input=inp,
                            total_observed=cell_total,
                            required_total=self._eq_distribution_min_total(),
                            max_gap=round(diagnostics.get('max_gap', 0.0), 6),
                            threshold=round(diagnostics.get('threshold', 0.0), 6),
                            hoeffding_threshold=round(diagnostics.get('hoeffding_threshold', 0.0), 6),
                            probability_tolerance=round(diagnostics.get('probability_tolerance', 0.0), 6),
                        )
                        continue
                    out = diagnostics.get('witness_output')
                    if out not in c_tree:
                        out = max(c_tree.keys(), key=lambda o: (c_tree.get(o, 0), repr(o)))
                    result = {
                        'io_trace': io_trace + (inp, out),
                        'source': 'observation_tree_scan',
                        'already_observed': True,
                        'diff_kind': 'distribution_mismatch',
                        'prefix_io': io_trace,
                        'input': inp,
                        'output': out,
                        'diagnostics': dict(diagnostics),
                    }
                    self._log(
                        'tree_scan_counterexample',
                        reason=reason,
                        candidate_index=candidate_index,
                        inputs=self._short_inputs_repr(inputs) if inputs is not None else None,
                        witness_io_trace=result['io_trace'],
                        nodes_checked_delta=self.stats['tree_nodes_checked'] - tree_nodes_before,
                        max_searchable_len=max_searchable_len,
                        elapsed_sec=round(perf_counter() - scan_t0, 6),
                        diff_kind='distribution_mismatch',
                        max_gap=round(diagnostics.get('max_gap', 0.0), 6),
                        threshold=round(diagnostics.get('threshold', 0.0), 6),
                        hoeffding_threshold=round(diagnostics.get('hoeffding_threshold', 0.0), 6),
                        probability_tolerance=round(diagnostics.get('probability_tolerance', 0.0), 6),
                        empirical_probability=round(diagnostics.get('empirical_probability', 0.0), 6),
                        hypothesis_probability=round(diagnostics.get('hypothesis_probability', 0.0), 6),
                        observed_count=diagnostics.get('observed_count'),
                        total_observed=diagnostics.get('total_observed'),
                    )
                    return result

                shared = sorted(set(c_tree.keys()).intersection(c_hyp.keys()), key=lambda o: (c_tree.get(o, 0), repr(o)), reverse=True)
                for out in shared:
                    succ_tree = tree_node.get_successor(inp, out)
                    succ_hyp = output_to_state.get(out)
                    if succ_tree is None or succ_hyp is None:
                        continue
                    if completion_threshold > 0 and not self._tree_node_complete_for_eq(succ_tree, completion_threshold):
                        continue
                    next_len_steps = current_len_steps + 1
                    if next_len_steps >= max_searchable_len:
                        continue
                    queue.append((succ_tree, succ_hyp, io_trace + (inp, out)))
        self._log(
            'tree_scan_done',
            reason=reason,
            candidate_index=candidate_index,
            inputs=self._short_inputs_repr(inputs) if inputs is not None else None,
            max_searchable_len=max_searchable_len,
            nodes_checked_delta=self.stats['tree_nodes_checked'] - tree_nodes_before,
            elapsed_sec=round(perf_counter() - scan_t0, 6),
            counterexample=None,
        )
        return None

    def find_cex(self, hypothesis):
        t0 = perf_counter()
        self.stats['find_cex_calls'] += 1
        self.last_result = None
        total_candidates = self._candidate_count_estimate()
        self._log(
            'find_cex_start',
            call_index=self.stats['find_cex_calls'],
            num_samples=self.num_samples,
            min_cex_len=self.min_cex_len,
            max_cex_len=self.max_cex_len,
            random_search=self.random_search,
            random_trace_count=self.random_trace_count if self.random_search else None,
            random_trace_len=self._effective_random_trace_len() if self.random_search else None,
            configured_random_trace_len=self.random_trace_len if self.random_search else None,
            total_candidates=total_candidates,
        )
        try:
            reusable = self._reuse_last_cex_if_valid(hypothesis)
            if reusable is not None:
                self.last_result = {
                    'counterexample': reusable,
                    'source': 'last_cex_reuse',
                }
                self._log('find_cex_reused', counterexample=reusable)
                return reusable

            tree_cex = self._tree_based_counterexample(hypothesis, reason='initial_tree_scan')
            if tree_cex is not None:
                self.last_cex = tree_cex
                self.last_result = {
                    'counterexample': tree_cex,
                    'source': 'tree_scan',
                }
                self.stats['counterexamples_found'] += 1
                self.stats['tree_cex_found'] += 1
                self._log('find_cex_counterexample', source='initial_tree_scan', counterexample=tree_cex)
                return tree_cex

            for candidate_index, inputs in enumerate(self._candidate_input_sequences(), start=1):
                self.stats['candidate_sequences_checked'] += 1
                if self._should_log_candidate(candidate_index, total_candidates):
                    self._log(
                        'candidate_start',
                        candidate_index=candidate_index,
                        total_candidates=total_candidates,
                        inputs=self._short_inputs_repr(inputs),
                    )

                sul_counts, sampled_io_traces = self._sample_sul_distribution(inputs)

                if self._should_log_candidate(candidate_index, total_candidates):
                    self._log(
                        'candidate_sampled',
                        candidate_index=candidate_index,
                        total_candidates=total_candidates,
                        inputs=self._short_inputs_repr(inputs),
                        sampled_traces=len(sampled_io_traces),
                        distinct_output_traces=len(sul_counts),
                    )

                sampled_cex = self._sampled_trace_counterexample(hypothesis, sampled_io_traces)
                if sampled_cex is not None:
                    self.last_cex = sampled_cex
                    self.last_result = {
                        'inputs': tuple(inputs),
                        'counterexample': sampled_cex,
                        'source': 'sampled_trace',
                        'sampled_io_traces': sampled_io_traces,
                    }
                    self.stats['counterexamples_found'] += 1
                    self._log(
                        'find_cex_counterexample',
                        source='sampled_trace',
                        candidate_index=candidate_index,
                        total_candidates=total_candidates,
                        inputs=self._short_inputs_repr(inputs),
                        counterexample=sampled_cex,
                    )
                    return sampled_cex

                tree_cex = self._tree_based_counterexample(
                    hypothesis,
                    reason='tree_after_sampling',
                    candidate_index=candidate_index,
                    inputs=inputs,
                )
                if tree_cex is not None:
                    self.last_cex = tree_cex
                    self.last_result = {
                        'inputs': tuple(inputs),
                        'counterexample': tree_cex,
                        'source': 'tree_after_sampling',
                        'sampled_io_traces': sampled_io_traces,
                    }
                    self.stats['counterexamples_found'] += 1
                    self.stats['tree_cex_found'] += 1
                    self._log(
                        'find_cex_counterexample',
                        source='tree_after_sampling',
                        candidate_index=candidate_index,
                        total_candidates=total_candidates,
                        inputs=self._short_inputs_repr(inputs),
                        counterexample=tree_cex,
                    )
                    return tree_cex

                if self._should_log_candidate(candidate_index, total_candidates):
                    self._log(
                        'candidate_checked',
                        candidate_index=candidate_index,
                        total_candidates=total_candidates,
                        inputs=self._short_inputs_repr(inputs),
                        sampled_traces=len(sampled_io_traces),
                        distinct_output_traces=len(sul_counts),
                    )

            self._log(
                'find_cex_exhausted',
                total_candidates=total_candidates,
                checked_candidates=self.stats['candidate_sequences_checked'],
                counterexample=None,
            )
            return None
        finally:
            elapsed = perf_counter() - t0
            self.stats['find_cex_time_sec'] += elapsed
            self.stats['last_find_cex_time_sec'] = elapsed
            self._log(
                'find_cex_done',
                elapsed_sec=round(elapsed, 6),
                checked_candidates=self.stats['candidate_sequences_checked'],
                sampled_traces=self.stats['sampled_traces'],
                sampled_steps=self.stats['sampled_steps'],
                counterexamples_found=self.stats['counterexamples_found'],
            )

    def find_counterexample(self, hypothesis):
        return self.find_cex(hypothesis)

    def equivalence_query(self, hypothesis):
        return self.find_cex(hypothesis)

def _execute_input_sequence(sul, inputs):
    if sul is None:
        raise ValueError('EQ oracle returned an input-only counterexample, but no SUL was provided.')

    observed = []
    sul.pre()
    try:
        for inp in inputs:
            out = sul.step(inp)
            observed.extend([inp, out])
    finally:
        sul.post()
    return tuple(observed)


def _normalize_counterexample(cex, sul=None):
    if cex is None:
        return None

    if isinstance(cex, dict):
        if 'io_trace' in cex:
            return tuple(cex['io_trace'])
        if 'inputs' in cex and 'outputs' in cex:
            inputs = tuple(cex['inputs'])
            outputs = tuple(cex['outputs'])
            if len(inputs) != len(outputs):
                raise ValueError('Counterexample inputs/outputs length mismatch.')
            io_trace = []
            for i, o in zip(inputs, outputs):
                io_trace.extend([i, o])
            return tuple(io_trace)
        if 'inputs' in cex:
            return _execute_input_sequence(sul, tuple(cex['inputs']))
        raise ValueError('Unsupported counterexample dict format.')

    if isinstance(cex, (tuple, list)):
        if len(cex) == 2 and all(isinstance(x, (tuple, list)) for x in cex):
            inputs = tuple(cex[0])
            outputs = tuple(cex[1])
            if len(inputs) != len(outputs):
                raise ValueError('Counterexample inputs/outputs length mismatch.')
            io_trace = []
            for i, o in zip(inputs, outputs):
                io_trace.extend([i, o])
            return tuple(io_trace)
        raise ValueError(
            'Ambiguous counterexample tuple/list format. '
            "Use {'io_trace': ...}, {'inputs': ...}, or (inputs, outputs)."
        )

    raise TypeError(f'Unsupported counterexample type: {type(cex)}')


def _find_counterexample(eq_oracle, hypothesis):
    if hasattr(eq_oracle, 'find_cex'):
        return eq_oracle.find_cex(hypothesis)
    if hasattr(eq_oracle, 'find_counterexample'):
        return eq_oracle.find_counterexample(hypothesis)
    if hasattr(eq_oracle, 'equivalence_query'):
        return eq_oracle.equivalence_query(hypothesis)
    raise AttributeError(
        'EQ oracle must implement one of: find_cex, find_counterexample, equivalence_query.'
    )


def _counterexample_already_in_tree(tree, io_trace):
    return tree is not None and io_trace and tree.get_successor(io_trace) is not None


def _hypothesis_state_path_for_io_trace(hypothesis, io_trace):
    current_state = getattr(hypothesis, 'initial_state', None)
    states = [current_state] if current_state is not None else []
    if current_state is None:
        return states, 0, False

    matched_steps = 0
    for inp, out in zip(io_trace[0::2], io_trace[1::2]):
        matched = None
        for target_state, output, probability in getattr(current_state, 'transitions', {}).get(inp, []):
            if output == out and probability > 0:
                matched = target_state
                break
        if matched is None:
            return states, matched_steps, False
        current_state = matched
        states.append(current_state)
        matched_steps += 1
    return states, matched_steps, True


def _shortest_unsupported_hypothesis_prefix(hypothesis, io_trace):
    _, matched_steps, supported = _hypothesis_state_path_for_io_trace(hypothesis, io_trace)
    if supported:
        return io_trace
    return io_trace[: 2 * (matched_steps + 1)]


def _process_counterexample(
    *,
    tree,
    sul,
    hypothesis,
    cex,
    cex_processing,
):
    io_trace = _normalize_counterexample(cex, sul=sul)
    if io_trace is None or len(io_trace) == 0:
        raise ValueError('Received an empty counterexample from EQ oracle.')

    original_len = len(io_trace) // 2
    cex_info = dict(cex) if isinstance(cex, dict) else {}
    already_inserted = _counterexample_already_in_tree(tree, io_trace)
    diagnostics = {
        'strategy': cex_processing,
        'original_len': original_len,
        'processed_len': original_len,
        'already_inserted': already_inserted,
        'diff_kind': cex_info.get('diff_kind'),
    }

    if cex_processing == 'longest_prefix':
        processed_trace = _shortest_unsupported_hypothesis_prefix(hypothesis, io_trace)
        already_inserted = _counterexample_already_in_tree(tree, processed_trace)
        diagnostics.update({
            'processed_len': len(processed_trace) // 2,
            'trimmed': len(processed_trace) < len(io_trace),
            'trim_reason': 'unsupported_hypothesis_prefix',
        })
    else:
        processed_trace = io_trace

    tree_cex_processing = 'binary' if cex_processing == 'binary' else 'full'
    if cex_info.get('already_observed') or cex_info.get('source') == 'observation_tree_scan':
        tree_cex_processing = 'binary'

    processed_trace = tree.process_counter_example(
        hypothesis,
        processed_trace,
        already_inserted=already_inserted,
        cex_processing=tree_cex_processing,
        cex_info=cex_info,
    )
    diagnostics['processed_len'] = len(processed_trace) // 2
    diagnostics['already_inserted'] = _counterexample_already_in_tree(tree, processed_trace)
    return processed_trace, diagnostics


SUPPORTED_DIFF_STRATEGIES = ('classic', 'normal', 'chi2')
PUBLIC_DIFF_STRATEGIES = SUPPORTED_DIFF_STRATEGIES
PUBLIC_AUTOMATON_TYPES = ('smm', 'stochastic_mealy')


def create_diff_checker(strategy='normal', alpha=0.05, use_diff=True, diff_checker=None):
    if diff_checker is not None:
        if not isinstance(diff_checker, DifferenceChecker):
            raise TypeError('diff_checker must implement DifferenceChecker.')
        return diff_checker

    if strategy == 'classic':
        return HoeffdingChecker(alpha=alpha)
    if strategy == 'normal':
        return AdvancedHoeffdingChecker(alpha=alpha, use_diff=use_diff)
    if strategy == 'chi2':
        return ChiSquareChecker(alpha=alpha, use_diff_value=use_diff)
    raise ValueError(
        f"Unsupported strategy {strategy!r}. Choose one of: {', '.join(SUPPORTED_DIFF_STRATEGIES)}."
    )


def _coerce_public_input_alphabet(input_alphabet=None):
    if input_alphabet is None:
        raise ValueError('input_alphabet must not be None.')
    return tuple(input_alphabet)


def _coerce_public_automaton_type(automaton_type):
    if automaton_type in ('smm', 'stochastic_mealy'):
        return 'stochastic_mealy'
    if automaton_type == 'mdp':
        raise NotImplementedError(
            'run_stochastic_Lsharp currently supports only SMM / stochastic_mealy targets. '
            'Use run_stochastic_Lstar for MDP learning.'
        )
    raise ValueError(
        f"Unsupported automaton_type {automaton_type!r}. Choose one of: {', '.join(PUBLIC_AUTOMATON_TYPES)}."
    )


def create_apartness_checker(
    apartness_checker=None,
    diff_checker=None,
    alpha=0.05,
    max_depth=None,
    max_nodes_visited=None,
    max_queue_size=None,
):
    from aalpy.learning_algs.stochastic.StochasticApartness import BFSApartnessChecker

    if apartness_checker is None:
        return BFSApartnessChecker(
            alpha=alpha,
            max_depth=max_depth,
            max_nodes_visited=max_nodes_visited,
            max_queue_size=max_queue_size,
            diff_checker=diff_checker,
        )

    if diff_checker is not None and hasattr(apartness_checker, 'set_diff_checker'):
        apartness_checker.set_diff_checker(diff_checker)
    if max_depth is not None and hasattr(apartness_checker, 'max_depth'):
        apartness_checker.max_depth = max_depth
    if max_nodes_visited is not None and hasattr(apartness_checker, 'max_nodes_visited'):
        apartness_checker.max_nodes_visited = max_nodes_visited
    if max_queue_size is not None and hasattr(apartness_checker, 'max_queue_size'):
        apartness_checker.max_queue_size = max_queue_size
    return apartness_checker


def _run_stochastic_lsharp_internal(
    alphabet,
    sul,
    eq_oracle,
    apartness_checker,
    diff_checker=None,
    strategy='normal',
    automaton_type='stochastic_mealy',
    n_c=20,
    max_learning_rounds=None,
    cex_processing='binary',
    return_data=False,
    n_resample=None,
    print_level=2,
    internal_config=None,
    observation_tree_cls=StochasticObservationTree,
    state_cls=None,
    machine_cls=None,
    debug=False,
    logger=None,
    query_strategy='symbolic',
    model_strategy='symbolic',
    llm_querier=None,
    llm_modeler=None,
    llm_config=None,
):
    if cex_processing not in ('full', 'longest_prefix', 'binary'):
        raise ValueError("cex_processing must be one of: 'full', 'longest_prefix', 'binary'.")
    query_strategy = _coerce_query_strategy(query_strategy)
    model_strategy = _coerce_model_strategy(model_strategy)
    llm_config = llm_config or LLMRunConfig()

    if diff_checker is not None and hasattr(apartness_checker, 'set_diff_checker'):
        apartness_checker.set_diff_checker(diff_checker)

    log = _make_debug_log_fn(debug, logger)
    tree = observation_tree_cls(
        alphabet=alphabet,
        sul=sul,
        automaton_type=automaton_type,
    )
    if hasattr(tree, 'set_debug'):
        tree.set_debug(debug, logger)
    if hasattr(eq_oracle, 'set_debug'):
        eq_oracle.set_debug(debug, logger)
    if hasattr(eq_oracle, 'attach_tree'):
        eq_oracle.attach_tree(tree, diff_checker=diff_checker)

    internal_config = (internal_config or StochasticLSharpInternalConfig()).resolved(n_c=n_c)

    if internal_config.apartness_max_depth is not None and hasattr(apartness_checker, 'max_depth'):
        apartness_checker.max_depth = internal_config.apartness_max_depth
    if internal_config.apartness_max_nodes_visited is not None and hasattr(apartness_checker, 'max_nodes_visited'):
        apartness_checker.max_nodes_visited = internal_config.apartness_max_nodes_visited
    if internal_config.apartness_max_queue_size is not None and hasattr(apartness_checker, 'max_queue_size'):
        apartness_checker.max_queue_size = internal_config.apartness_max_queue_size

    resample_hits = _resolve_resample_hits(n_resample, n_c)
    identification_budget = _resolve_identification_budget(resample_hits, internal_config)
    eq_distribution_min_total = _resolve_eq_distribution_min_total(resample_hits, internal_config)
    internal_config.eq_distribution_min_total = eq_distribution_min_total
    if hasattr(eq_oracle, 'configure_distribution_ce'):
        eq_oracle.configure_distribution_ce(
            alpha=internal_config.eq_distribution_alpha,
            probability_tolerance=internal_config.eq_probability_tolerance,
            min_total=eq_distribution_min_total,
            refinement_hits=resample_hits,
            identification_budget=identification_budget,
        )
    if hasattr(tree, 'configure_learning'):
        tree.configure_learning(
            apartness_checker=apartness_checker,
            diff_checker=diff_checker,
            n_c=n_c,
            apartness_n_c=internal_config.apartness_n_c,
            completion_max_attempts=internal_config.completion_max_attempts,
            store_failed_prefixes=internal_config.store_failed_prefixes,
            completion_rollout_len=internal_config.completion_rollout_len,
            identification_budget=identification_budget,
            resample_hits=resample_hits,
            chaos_reach_threshold=internal_config.chaos_reach_threshold,
            chaos_adaptive=internal_config.chaos_adaptive,
            chaos_max_reach_threshold=internal_config.chaos_max_reach_threshold,
            chaos_growth_factor=internal_config.chaos_growth_factor,
            chaos_stagnation_rounds=internal_config.chaos_stagnation_rounds,
        )

    learning_rounds = 0
    counterexamples = []
    learner_stats = {
        'nr_hypothesis_builds': 0,
        'nr_eq_calls': 0,
        'nr_counterexample_insertions': 0,
        'nr_learning_rounds_started': 0,
        'nr_llm_querier_calls': 0,
        'nr_llm_input_sequences': 0,
        'nr_llm_executed_traces': 0,
        'nr_llm_querier_skipped_empty_pool': 0,
        'nr_llm_modeler_calls': 0,
        'nr_llm_modeler_accepted': 0,
        'nr_llm_modeler_rejected': 0,
        'nr_llm_modeler_validation_failures': 0,
        'nr_llm_modeler_budget_skips': 0,
        'nr_symbolic_budget_interrupts': 0,
        'nr_symbolic_stalls_before_budget': 0,
    }
    timings = {
        'total_learning_time_sec': 0.0,
        'hypothesis_build_time_sec': 0.0,
        'llm_modeler_time_sec': 0.0,
        'eq_time_sec': 0.0,
        'counterexample_processing_time_sec': 0.0,
    }
    phase_sul_stats = {
        'llm_querier': None,
        'build_hypothesis': None,
        'eq': None,
        'counterexample_processing': None,
        'total': None,
    }

    overall_t0 = perf_counter()
    initial_sul_stats = _snapshot_stats(sul)
    last_hypothesis = None

    def _finish_with_hypothesis(hypothesis, *, converged, termination_reason):
        if hasattr(tree, 'clear_sampling_budget_limit'):
            tree.clear_sampling_budget_limit()
        if hasattr(sul, 'clear_sampling_budget_limit'):
            sul.clear_sampling_budget_limit()
        timings['total_learning_time_sec'] = perf_counter() - overall_t0
        final_sul_stats = _snapshot_stats(sul)
        phase_sul_stats['total'] = _stats_delta(final_sul_stats, initial_sul_stats)
        tree_stats = _snapshot_stats(tree, 'debug_stats')
        eq_oracle_stats = _snapshot_stats(eq_oracle, 'stats')
        sul_stats = final_sul_stats
        query_counts = {
            'learning_rounds': learning_rounds,
            'eq_calls': learner_stats['nr_eq_calls'],
            'hypothesis_builds': learner_stats['nr_hypothesis_builds'],
            'counterexamples': len(counterexamples),
            'candidate_sequences_checked': (eq_oracle_stats or {}).get('candidate_sequences_checked', 0),
            'llm_querier_calls': learner_stats.get('nr_llm_querier_calls', 0),
            'llm_input_sequences': learner_stats.get('nr_llm_input_sequences', 0),
            'llm_executed_traces': learner_stats.get('nr_llm_executed_traces', 0),
            'llm_modeler_calls': learner_stats.get('nr_llm_modeler_calls', 0),
            'llm_modeler_accepted': learner_stats.get('nr_llm_modeler_accepted', 0),
            'llm_modeler_rejected': learner_stats.get('nr_llm_modeler_rejected', 0),
            'llm_modeler_validation_failures': learner_stats.get('nr_llm_modeler_validation_failures', 0),
            'llm_modeler_budget_skips': learner_stats.get('nr_llm_modeler_budget_skips', 0),
            'symbolic_budget_interrupts': learner_stats.get('nr_symbolic_budget_interrupts', 0),
            'symbolic_stalls_before_budget': learner_stats.get('nr_symbolic_stalls_before_budget', 0),
            'total_sul_episodes': (sul_stats or {}).get('episodes', 0),
        }
        sampling_counts = {
            'total_sul_steps': (sul_stats or {}).get('steps', 0),
            'eq_sampled_traces': (eq_oracle_stats or {}).get('sampled_traces', 0),
            'eq_sampled_steps': (eq_oracle_stats or {}).get('sampled_steps', 0),
            'completion_successful_samples': (tree_stats or {}).get('ensure_complete_matched_attempts', 0),
            'completion_failed_replays': (tree_stats or {}).get('ensure_complete_failed_replays', 0),
            'completion_attempts': (tree_stats or {}).get('ensure_complete_attempts', 0),
            'identify_frontier_samples': (tree_stats or {}).get('identify_frontier_samples', 0),
            'refinement_queries': (tree_stats or {}).get('refinement_queries', 0),
            'refinement_attempts': (tree_stats or {}).get('refinement_attempts', 0),
            'refinement_target_hits': (tree_stats or {}).get('refinement_target_hits', 0),
            'refinement_targets_completed': (tree_stats or {}).get('refinement_targets_completed', 0),
            'guided_completion_queries': (tree_stats or {}).get('guided_completion_queries', 0),
            'guided_frontier_queries': (tree_stats or {}).get('guided_frontier_queries', 0),
            'llm_executed_traces': learner_stats.get('nr_llm_executed_traces', 0),
        }
        result_info = {
            'learning_rounds': learning_rounds,
            'nr_counterexamples': len(counterexamples),
            'counterexamples': list(counterexamples),
            'observation_tree': tree,
            'learner_stats': learner_stats,
            'timings': timings,
            'tree_stats': tree_stats,
            'eq_oracle_stats': eq_oracle_stats,
            'sul_stats': sul_stats,
            'phase_sul_stats': phase_sul_stats,
            'query_counts': query_counts,
            'sampling_counts': sampling_counts,
            'converged': bool(converged),
            'termination_reason': termination_reason,
        }
        if return_data:
            return hypothesis, result_info
        return hypothesis

    def _modeler_budget_progress():
        stats = _snapshot_stats(sul)
        unit = getattr(llm_config, 'modeler_budget_unit', 'episodes')
        base_stats = initial_sul_stats or {}
        if unit == 'steps':
            return max(0, _stats_value(stats, 'steps', 'num_steps') - _stats_value(base_stats, 'steps', 'num_steps'))
        return max(0, _stats_value(stats, 'episodes', 'queries', 'num_queries') - _stats_value(base_stats, 'episodes', 'queries', 'num_queries'))

    def _modeler_budget_absolute_limit():
        budget = getattr(llm_config, 'modeler_activation_sample_budget', None)
        if budget is None:
            return None
        unit = getattr(llm_config, 'modeler_budget_unit', 'episodes')
        base_stats = initial_sul_stats or {}
        if unit == 'steps':
            return _stats_value(base_stats, 'steps', 'num_steps') + int(budget)
        return _stats_value(base_stats, 'episodes', 'queries', 'num_queries') + int(budget)

    def _activate_tree_sampling_budget():
        limit = _modeler_budget_absolute_limit()
        if limit is None:
            return
        unit = getattr(llm_config, 'modeler_budget_unit', 'episodes')
        if hasattr(tree, 'set_sampling_budget_limit'):
            tree.set_sampling_budget_limit(limit, unit=unit)
        if hasattr(sul, 'set_sampling_budget_limit'):
            sul.set_sampling_budget_limit(limit, unit=unit)

    def _clear_tree_sampling_budget():
        # Sampling budget is global for the learner. It must remain active
        # across hypothesis construction, EQ, and CE processing. It is cleared
        # only when the learner returns.
        return

    def _modeler_budget_configured():
        return getattr(llm_config, 'modeler_activation_sample_budget', None) is not None

    def _modeler_budget_reached():
        budget = getattr(llm_config, 'modeler_activation_sample_budget', None)
        if budget is None:
            return True
        return _modeler_budget_progress() >= int(budget)

    def _log_modeler_budget_skip():
        learner_stats['nr_llm_modeler_budget_skips'] = learner_stats.get('nr_llm_modeler_budget_skips', 0) + 1
        log(
            'llm_modeler_skipped_before_sample_budget',
            component='learner',
            learning_round=learning_rounds,
            budget=getattr(llm_config, 'modeler_activation_sample_budget', None),
            budget_unit=getattr(llm_config, 'modeler_budget_unit', 'episodes'),
            current_samples=_modeler_budget_progress(),
        )

    def _readonly_symbolic_modelability():
        basis = list(getattr(tree, 'basis', []) or [])
        alphabet = list(getattr(tree, 'alphabet', []) or [])
        incomplete_basis = []
        for basis_state in basis:
            for inp in alphabet:
                total = basis_state.get_frequency_sum(inp)
                successors = basis_state.successors.get(inp, {})
                if total <= 0 or not successors:
                    incomplete_basis.append({
                        'basis_node_id': getattr(basis_state, 'id', None),
                        'input': inp,
                        'samples': total,
                    })
        if incomplete_basis:
            return False, f'incomplete_basis={incomplete_basis[:20]}'

        basis_set = set(basis)
        current_frontier = getattr(tree, 'frontier_to_basis_dict', {}) or {}
        exact_frontier = []
        if hasattr(tree, 'get_observed_successors'):
            seen = set()
            for basis_state in basis:
                for successor in tree.get_observed_successors(basis_state):
                    if successor in basis_set or successor in seen:
                        continue
                    seen.add(successor)
                    exact_frontier.append(successor)
        else:
            exact_frontier = list(current_frontier.keys())

        missing_frontiers = [
            getattr(node, 'id', None)
            for node in exact_frontier
            if node not in current_frontier
        ]
        if missing_frontiers:
            return False, f'missing_frontiers={missing_frontiers[:50]}'

        unresolved = {}
        for node in exact_frontier:
            candidates = current_frontier.get(node, set())
            use_chaos = False
            if hasattr(tree, '_should_use_chaos_for_frontier'):
                use_chaos = tree._should_use_chaos_for_frontier(node)
            if len(candidates) != 1 and not use_chaos:
                unresolved[getattr(node, 'id', None)] = [
                    getattr(candidate, 'id', None)
                    for candidate in candidates
                ]
        if unresolved:
            return False, f'unresolved_frontiers={unresolved}'
        return True, None

    def _construct_symbolic_if_budget_exhausted(*, require_budget=True):
        try:
            if require_budget and not _modeler_budget_reached():
                return None, 'budget_not_reached'
            modelable, reason = _readonly_symbolic_modelability()
            if not modelable:
                return None, reason
            return tree.construct_hypothesis(state_cls=state_cls, machine_cls=machine_cls), None
        except Exception as exc:
            return None, repr(exc)

    while True:
        if max_learning_rounds is not None and learning_rounds >= max_learning_rounds:
            if last_hypothesis is not None:
                log(
                    'max_rounds_reached_returning_partial_hypothesis',
                    component='learner',
                    learning_round=learning_rounds,
                    max_learning_rounds=max_learning_rounds,
                    nr_states=len(getattr(last_hypothesis, 'states', []) or []),
                )
                return _finish_with_hypothesis(
                    last_hypothesis,
                    converged=False,
                    termination_reason='max_rounds_with_partial_hypothesis',
                )
            raise RuntimeError(
                f'run_stochastic_Lsharp did not terminate within {max_learning_rounds} learning rounds.'
            )

        learning_rounds += 1
        learner_stats['nr_learning_rounds_started'] += 1
        log('learning_round_start', component='learner', learning_round=learning_rounds)
        _activate_tree_sampling_budget()

        if query_strategy == 'llm':
            if llm_querier is None:
                llm_querier = _make_default_llm_querier(llm_config=llm_config)
            try:
                llm_querier, _, _ = _run_llm_querier_from_input_sequences_file(
                    llm_querier=llm_querier,
                    tree=tree,
                    sul=sul,
                    alphabet=alphabet,
                    learning_round=learning_rounds,
                    llm_config=llm_config,
                    phase_sul_stats=phase_sul_stats,
                    learner_stats=learner_stats,
                    log=log,
                )
            except SamplingBudgetExceeded as exc:
                if model_strategy != 'llm':
                    raise
                learner_stats['nr_symbolic_budget_interrupts'] = (
                    learner_stats.get('nr_symbolic_budget_interrupts', 0) + 1
                )
                log(
                    'llm_querier_stopped_at_sample_budget',
                    component='learner',
                    learning_round=learning_rounds,
                    current_samples=exc.current_samples,
                    budget=getattr(llm_config, 'modeler_activation_sample_budget', None),
                    budget_unit=getattr(llm_config, 'modeler_budget_unit', 'episodes'),
                )
                t0 = perf_counter()
                llm_modeler, _, hypothesis = _run_llm_budget_finalizer(
                    llm_modeler=llm_modeler,
                    tree=tree,
                    learning_round=learning_rounds,
                    state_cls=state_cls,
                    machine_cls=machine_cls,
                    learner_stats=learner_stats,
                    llm_config=llm_config,
                    log=log,
                )
                timings['llm_modeler_time_sec'] += perf_counter() - t0
                learner_stats['nr_hypothesis_builds'] += 1
                last_hypothesis = hypothesis
                return _finish_with_hypothesis(
                    hypothesis,
                    converged=False,
                    termination_reason='budget_exhausted_llm_merged_hypothesis',
                )

        sul_before = _snapshot_stats(sul)
        t0 = perf_counter()
        modeler_used = False
        finish_after_hypothesis = False
        finish_reason = None
        retry_symbolic_before_budget = False
        try:
            _activate_tree_sampling_budget()
            hypothesis = tree.build_hypothesis(
                state_cls=state_cls,
                machine_cls=machine_cls,
            )
            if model_strategy == 'llm' and _modeler_budget_configured() and _modeler_budget_reached():
                log(
                    'sampling_budget_reached_using_llm_state_merge',
                    component='learner',
                    learning_round=learning_rounds,
                    current_samples=_modeler_budget_progress(),
                    budget=getattr(llm_config, 'modeler_activation_sample_budget', None),
                )
                llm_modeler, _, hypothesis = _run_llm_budget_finalizer(
                    llm_modeler=llm_modeler,
                    tree=tree,
                    learning_round=learning_rounds,
                    state_cls=state_cls,
                    machine_cls=machine_cls,
                    learner_stats=learner_stats,
                    llm_config=llm_config,
                    log=log,
                )
                modeler_used = True
                finish_after_hypothesis = True
                finish_reason = 'budget_exhausted_llm_merged_hypothesis'
        except SamplingBudgetExceeded as exc:
            _clear_tree_sampling_budget()
            learner_stats['nr_symbolic_budget_interrupts'] = (
                learner_stats.get('nr_symbolic_budget_interrupts', 0) + 1
            )
            if model_strategy != 'llm':
                raise
            log(
                'symbolic_model_build_stopped_at_sample_budget',
                component='learner',
                learning_round=learning_rounds,
                budget=getattr(llm_config, 'modeler_activation_sample_budget', None),
                budget_unit=getattr(llm_config, 'modeler_budget_unit', 'episodes'),
                current_samples=exc.current_samples,
            )
            log(
                'symbolic_budget_exhausted_using_llm_state_merge',
                component='learner',
                learning_round=learning_rounds,
            )
            llm_modeler, _, hypothesis = _run_llm_budget_finalizer(
                llm_modeler=llm_modeler,
                tree=tree,
                learning_round=learning_rounds,
                state_cls=state_cls,
                machine_cls=machine_cls,
                learner_stats=learner_stats,
                llm_config=llm_config,
                log=log,
            )
            modeler_used = True
            finish_after_hypothesis = True
            finish_reason = 'budget_exhausted_llm_merged_hypothesis'
        except RuntimeError as exc:
            _clear_tree_sampling_budget()
            if model_strategy != 'llm' or not _is_recoverable_symbolic_model_error(exc):
                raise
            budget_reached = _modeler_budget_reached()
            if not budget_reached:
                learner_stats['nr_symbolic_stalls_before_budget'] = (
                    learner_stats.get('nr_symbolic_stalls_before_budget', 0) + 1
                )
            log(
                'symbolic_model_build_failed_using_llm_modeler'
                if budget_reached
                else 'symbolic_model_build_stalled_before_budget_using_llm_modeler',
                component='learner',
                learning_round=learning_rounds,
                error=str(exc),
                budget_reached=budget_reached,
                current_samples=_modeler_budget_progress(),
                budget=getattr(llm_config, 'modeler_activation_sample_budget', None),
            )
            if not budget_reached:
                retry_symbolic_before_budget = True
                hypothesis = None
                log(
                    'symbolic_model_build_stalled_before_budget_continuing',
                    component='learner',
                    learning_round=learning_rounds,
                    original_error=str(exc),
                )
            else:
                log(
                    'symbolic_recoverable_error_using_llm_state_merge',
                    component='learner',
                    learning_round=learning_rounds,
                    original_error=str(exc),
                )
                llm_modeler, _, hypothesis = _run_llm_budget_finalizer(
                    llm_modeler=llm_modeler,
                    tree=tree,
                    learning_round=learning_rounds,
                    state_cls=state_cls,
                    machine_cls=machine_cls,
                    learner_stats=learner_stats,
                    llm_config=llm_config,
                    log=log,
                )
                modeler_used = True
                finish_after_hypothesis = True
                finish_reason = 'budget_exhausted_llm_merged_hypothesis'
        finally:
            _clear_tree_sampling_budget()
        elapsed_model_build = perf_counter() - t0
        timings['hypothesis_build_time_sec'] += elapsed_model_build
        if modeler_used:
            timings['llm_modeler_time_sec'] += elapsed_model_build
        if retry_symbolic_before_budget:
            learner_stats['nr_hypothesis_build_failures'] = (
                learner_stats.get('nr_hypothesis_build_failures', 0) + 1
            )
            phase_sul_stats['build_hypothesis'] = _accumulate_stats(
                phase_sul_stats['build_hypothesis'],
                _stats_delta(_snapshot_stats(sul), sul_before),
            )
            continue
        learner_stats['nr_hypothesis_builds'] += 1
        last_hypothesis = hypothesis
        phase_sul_stats['build_hypothesis'] = _accumulate_stats(
            phase_sul_stats['build_hypothesis'],
            _stats_delta(_snapshot_stats(sul), sul_before),
        )
        log(
            'hypothesis_built',
            component='learner',
            learning_round=learning_rounds,
            nr_states=len(getattr(hypothesis, 'states', []) or []),
            basis_size=len(getattr(tree, 'basis', [])),
            frontier_size=len(getattr(tree, 'frontier_to_basis_dict', {})),
            model_strategy=model_strategy,
            modeler_used=modeler_used,
        )
        if print_level > 1:
            print(f'Hypothesis {learning_rounds}: {len(getattr(hypothesis, "states", []) or [])} states.')
        if print_level == 3:
            print(hypothesis)

        if finish_after_hypothesis:
            return _finish_with_hypothesis(
                hypothesis,
                converged=False,
                termination_reason=finish_reason or 'budget_final_hypothesis',
            )

        sul_before = _snapshot_stats(sul)
        t0 = perf_counter()
        try:
            cex = _find_counterexample(eq_oracle, hypothesis)
        except SamplingBudgetExceeded as exc:
            timings['eq_time_sec'] += perf_counter() - t0
            learner_stats['nr_eq_calls'] += 1
            phase_sul_stats['eq'] = _accumulate_stats(
                phase_sul_stats['eq'],
                _stats_delta(_snapshot_stats(sul), sul_before),
            )
            if model_strategy != 'llm':
                raise
            learner_stats['nr_symbolic_budget_interrupts'] = (
                learner_stats.get('nr_symbolic_budget_interrupts', 0) + 1
            )
            log(
                'eq_stopped_at_sample_budget_using_llm_state_merge',
                component='learner',
                learning_round=learning_rounds,
                current_samples=exc.current_samples,
                budget=getattr(llm_config, 'modeler_activation_sample_budget', None),
                budget_unit=getattr(llm_config, 'modeler_budget_unit', 'episodes'),
            )
            t_modeler = perf_counter()
            llm_modeler, _, hypothesis = _run_llm_budget_finalizer(
                llm_modeler=llm_modeler,
                tree=tree,
                learning_round=learning_rounds,
                state_cls=state_cls,
                machine_cls=machine_cls,
                learner_stats=learner_stats,
                llm_config=llm_config,
                log=log,
            )
            timings['llm_modeler_time_sec'] += perf_counter() - t_modeler
            learner_stats['nr_hypothesis_builds'] += 1
            last_hypothesis = hypothesis
            return _finish_with_hypothesis(
                hypothesis,
                converged=False,
                termination_reason='budget_exhausted_llm_merged_hypothesis',
            )
        timings['eq_time_sec'] += perf_counter() - t0
        learner_stats['nr_eq_calls'] += 1
        phase_sul_stats['eq'] = _accumulate_stats(
            phase_sul_stats['eq'],
            _stats_delta(_snapshot_stats(sul), sul_before),
        )
        log(
            'eq_finished',
            component='learner',
            learning_round=learning_rounds,
            counterexample=cex,
            eq_last_result=getattr(eq_oracle, 'last_result', None),
        )
        if print_level > 2:
            print(f'Counterexample: {cex}')

        if cex is None:
            return _finish_with_hypothesis(
                hypothesis,
                converged=True,
                termination_reason='equivalence_oracle_acceptance',
            )

        sul_before = _snapshot_stats(sul)
        t0 = perf_counter()
        try:
            io_trace, cex_processing_details = _process_counterexample(
                tree=tree,
                sul=sul,
                hypothesis=hypothesis,
                cex=cex,
                cex_processing=cex_processing,
            )
        except SamplingBudgetExceeded as exc:
            timings['counterexample_processing_time_sec'] += perf_counter() - t0
            phase_sul_stats['counterexample_processing'] = _accumulate_stats(
                phase_sul_stats['counterexample_processing'],
                _stats_delta(_snapshot_stats(sul), sul_before),
            )
            if model_strategy != 'llm':
                raise
            learner_stats['nr_symbolic_budget_interrupts'] = (
                learner_stats.get('nr_symbolic_budget_interrupts', 0) + 1
            )
            log(
                'counterexample_processing_stopped_at_sample_budget_using_llm_state_merge',
                component='learner',
                learning_round=learning_rounds,
                current_samples=exc.current_samples,
                budget=getattr(llm_config, 'modeler_activation_sample_budget', None),
                budget_unit=getattr(llm_config, 'modeler_budget_unit', 'episodes'),
            )
            t_modeler = perf_counter()
            llm_modeler, _, hypothesis = _run_llm_budget_finalizer(
                llm_modeler=llm_modeler,
                tree=tree,
                learning_round=learning_rounds,
                state_cls=state_cls,
                machine_cls=machine_cls,
                learner_stats=learner_stats,
                llm_config=llm_config,
                log=log,
            )
            timings['llm_modeler_time_sec'] += perf_counter() - t_modeler
            learner_stats['nr_hypothesis_builds'] += 1
            last_hypothesis = hypothesis
            return _finish_with_hypothesis(
                hypothesis,
                converged=False,
                termination_reason='budget_exhausted_llm_merged_hypothesis',
            )
        if hasattr(eq_oracle, 'last_cex'):
            eq_oracle.last_cex = None
        if hasattr(eq_oracle, 'last_result'):
            eq_oracle.last_result = None
        counterexamples.append(io_trace)
        timings['counterexample_processing_time_sec'] += perf_counter() - t0
        learner_stats['nr_counterexample_insertions'] += 1
        phase_sul_stats['counterexample_processing'] = _accumulate_stats(
            phase_sul_stats['counterexample_processing'],
            _stats_delta(_snapshot_stats(sul), sul_before),
        )
        log(
            'counterexample_processed',
            component='learner',
            learning_round=learning_rounds,
            nr_counterexamples=len(counterexamples),
            io_trace=io_trace,
            cex_processing=cex_processing,
            cex_processing_details=cex_processing_details,
        )


def run_stochastic_Lsharp(
    input_alphabet=None,
    sul=None,
    eq_oracle=None,
    max_rounds=200,
    automaton_type='smm',
    strategy='normal',
    cex_processing=None,
    return_data=False,
    n_c=20,
    n_resample=None,
    print_level=2,
    *,
    alpha=0.05,
    apartness_checker=None,
    diff_checker=None,
    internal_config=None,
    observation_tree_cls=StochasticObservationTree,
    state_cls=None,
    machine_cls=None,
    debug=False,
    logger=None,
    query_strategy='symbolic',
    model_strategy='symbolic',
    llm_querier=None,
    llm_modeler=None,
    llm_config=None,
    **legacy_kwargs,
):
    """
    Sampling-based LSharp for stochastic Mealy machines.

    The public API is intentionally small and AALpy-like. Advanced sampling /
    apartness knobs are grouped under `internal_config` instead of appearing as
    many top-level parameters.
    """
    if sul is None:
        raise ValueError('sul must not be None.')
    if eq_oracle is None:
        raise ValueError('eq_oracle must not be None.')

    internal_config, legacy_kwargs = _coerce_internal_config(
        internal_config,
        n_c=n_c,
        legacy_kwargs=legacy_kwargs,
    )
    if legacy_kwargs:
        unexpected = ', '.join(sorted(legacy_kwargs.keys()))
        raise TypeError(f'Unexpected keyword argument(s): {unexpected}.')

    llm_config = llm_config or LLMRunConfig()
    resolved_alphabet = _coerce_public_input_alphabet(input_alphabet=input_alphabet)
    resolved_automaton_type = _coerce_public_automaton_type(automaton_type)

    if cex_processing is None:
        internal_cex_processing = 'binary'
    elif cex_processing in ('full', 'longest_prefix', 'binary'):
        internal_cex_processing = cex_processing
    else:
        raise ValueError("cex_processing must be one of: None, 'full', 'longest_prefix', 'binary'.")

    resolved_max_learning_rounds = max_rounds
    resolved_resample_hits = _resolve_resample_hits(n_resample, n_c)
    internal_config.identification_budget = _resolve_identification_budget(resolved_resample_hits, internal_config)
    internal_config.eq_distribution_min_total = _resolve_eq_distribution_min_total(resolved_resample_hits, internal_config)

    resolved_strategy_name = strategy if isinstance(strategy, str) else type(strategy).__name__
    if diff_checker is None:
        if isinstance(strategy, DifferenceChecker):
            diff_checker = strategy
        else:
            diff_checker = create_diff_checker(strategy=strategy, alpha=alpha, use_diff=True)

    apartness_checker = create_apartness_checker(
        apartness_checker=apartness_checker,
        diff_checker=diff_checker,
        alpha=alpha,
        max_depth=internal_config.apartness_max_depth,
        max_nodes_visited=internal_config.apartness_max_nodes_visited,
        max_queue_size=internal_config.apartness_max_queue_size,
    )

    result = _run_stochastic_lsharp_internal(
        alphabet=resolved_alphabet,
        sul=sul,
        eq_oracle=eq_oracle,
        apartness_checker=apartness_checker,
        diff_checker=diff_checker,
        strategy=resolved_strategy_name,
        automaton_type=resolved_automaton_type,
        n_c=n_c,
        max_learning_rounds=resolved_max_learning_rounds,
        cex_processing=internal_cex_processing,
        return_data=return_data,
        n_resample=resolved_resample_hits,
        print_level=print_level,
        internal_config=internal_config,
        observation_tree_cls=observation_tree_cls,
        state_cls=state_cls,
        machine_cls=machine_cls,
        debug=debug,
        logger=logger,
        query_strategy=query_strategy,
        model_strategy=model_strategy,
        llm_querier=llm_querier,
        llm_modeler=llm_modeler,
        llm_config=llm_config,
    )

    if not return_data:
        return result

    hypothesis, info = result
    info.setdefault('api_alignment', {})
    info['api_alignment'].update({
        'public_input_alphabet': resolved_alphabet,
        'public_automaton_type': automaton_type,
        'resolved_automaton_type': resolved_automaton_type,
        'public_strategy': resolved_strategy_name,
        'public_cex_processing': cex_processing,
        'resolved_cex_processing': internal_cex_processing,
        'public_query_strategy': query_strategy,
        'public_model_strategy': model_strategy,
        'public_max_rounds': max_rounds,
        'resolved_max_learning_rounds': resolved_max_learning_rounds,
        'n_resample_as_refinement_hits': resolved_resample_hits,
        'identification_budget': internal_config.identification_budget,
        'internal_config': asdict(internal_config),
        'llm_config': asdict(llm_config),
    })
    return hypothesis, info
