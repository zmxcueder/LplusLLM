from collections import defaultdict
from datetime import datetime
from math import log, sqrt
from pathlib import Path
from random import choice
import json


aut_type = ['stochastic_mealy']


class SamplingBudgetExceeded(RuntimeError):
    """Raised when learner-side SUL sampling reaches the configured budget."""

    def __init__(self, *, current_samples, limit, unit='episodes'):
        self.current_samples = int(current_samples)
        self.limit = int(limit)
        self.unit = unit
        super().__init__(
            f'Sampling budget exceeded: {self.current_samples} {self.unit} >= {self.limit}.'
        )


class ObservationTreeAdequacyError(RuntimeError):
    """Raised when unresolved frontier states keep the tree inadequate."""

    def __init__(self, unresolved, debug_snapshot=None):
        self.unresolved = unresolved
        self.debug_snapshot = debug_snapshot or {}
        super().__init__(
            f'Observation tree is not adequate after repeated identification rounds. '
            f'Unresolved frontier candidates: {unresolved}'
        )


class StochasticMealyNode:
    _id_counter = 0
    __slots__ = ['id', 'output', 'successors', 'parent', 'input_to_parent', 'frequency', 'input_frequencies', 'output_frequencies']

    def __init__(self, parent=None, input_to_parent=None, output=None):
        StochasticMealyNode._id_counter += 1
        self.id = StochasticMealyNode._id_counter
        self.parent = parent
        self.output = output
        self.input_to_parent = input_to_parent
        self.frequency = 0
        self.successors = defaultdict(dict)  # successors[input][output] = successor
        self.input_frequencies = defaultdict(int)  # input -> total count
        # output_frequencies[input][output] counts how often the immediate
        # one-step observation (input, output) was seen from *this* node.
        #
        # This must be stored explicitly at the parent node. Using the child
        # node frequency is incorrect, because deeper completion / replay
        # sampling revisits the child prefix many additional times and inflates
        # the apparent probability of that branch.
        self.output_frequencies = defaultdict(lambda: defaultdict(int))

    def __hash__(self):
        return hash(self.id)

    def id_counter(self):
        return self._id_counter

    def get_successor(self, inp, out):
        return self.successors.get(inp, {}).get(out)

    def get_output_frequencies(self, inp):
        return dict(self.output_frequencies.get(inp, {}))

    def get_output_frequency(self, inp, out):
        return self.output_frequencies.get(inp, {}).get(out, 0)

    def get_frequency_sum(self, inp):
        return self.input_frequencies[inp]


class StochasticObservationTree:
    """
    Observation-tree style learner state for stochastic Mealy machines.

    This class is intentionally structured to mirror deterministic LSharp's
    ObservationTree responsibilities:
      - maintain basis and frontier-to-basis candidate mapping
      - make the observation tree adequate before hypothesis construction
      - build a hypothesis from basis/frontier identification
      - process counterexamples inside the tree

    The stochastic adaptation keeps the current sampling primitives:
      - ensure_complete(...)
      - stochastic apartness via BFSApartnessChecker
    """

    def __init__(self, alphabet, sul, automaton_type):
        assert automaton_type in aut_type
        self.automaton_type = automaton_type
        self.alphabet = tuple(alphabet)
        self.sul = sul
        self.root = StochasticMealyNode()
        self.basis = [self.root]
        self.frontier_to_basis_dict = {}
        # Backward-compatible alias used by the current stochastic codebase.
        self.frontier = self.frontier_to_basis_dict
        self.guided_query_sequences = []
        self.guided_query_max_length = None
        self.guided_query_cursor = 0
        self.witness_cache = {}
        self.states_dict = {}
        self.debug = False
        self._debug_logger = None
        self.debug_stats = {
            'ensure_complete_calls': 0,
            'ensure_complete_attempts': 0,
            'ensure_complete_matched_attempts': 0,
            'ensure_complete_failed_replays': 0,
            'ensure_complete_runtime_errors': 0,
            'ensure_complete_rollout_steps': 0,
            'ensure_complete_rollout_traces': 0,
            'identify_frontier_calls': 0,
            'identify_frontier_samples': 0,
            'identify_frontier_witness_found': 0,
            'identify_frontier_missing_witness': 0,
            'identify_frontier_sample_attempts': 0,
            'identify_frontier_replay_matches': 0,
            'identify_frontier_replay_failures': 0,
            'identify_frontier_candidate_reductions': 0,
            'identify_frontier_no_progress': 0,
            'identify_frontier_single_candidate_returns': 0,
            'identify_frontier_zero_candidate_promotions': 0,
            'identify_frontier_disambiguation_targets': 0,
            'identify_frontier_nodes': {},
            'guided_completion_queries': 0,
            'guided_frontier_queries': 0,
            'single_candidate_rechecks': 0,
            'frontier_promotions': 0,
            'frontier_forced_merges': 0,
            'frontier_identification_escalations': 0,
            'process_counterexample_calls': 0,
            'probabilistic_counterexample_calls': 0,
            'probabilistic_counterexample_refinements': 0,
            'probabilistic_counterexample_binary_searches': 0,
            'probabilistic_counterexample_diff_cells': 0,
            'probabilistic_counterexample_unknown_cells': 0,
            'probabilistic_counterexample_full_scans': 0,
            'probabilistic_counterexample_insufficient_power': 0,
            'binary_search_calls': 0,
            'chaos_threshold_adaptations': 0,
            'sampling_budget_interruptions': 0,
            'refinement_targets_registered': 0,
            'refinement_rounds': 0,
            'refinement_queries': 0,
            'refinement_attempts': 0,
            'refinement_target_hits': 0,
            'refinement_targets_improved': 0,
            'refinement_targets_completed': 0,
            'refinement_target_duplicates': 0,
        }
        self._pending_refinement_targets = {}
        self._sampling_budget_limit = None
        self._sampling_budget_unit = 'episodes'
        self._active_apartness_checker = None
        self._active_diff_checker = None
        self._active_n_c = 0
        self._active_apartness_n_c = 0
        self._active_completion_max_attempts = None
        self._active_store_failed_prefixes = True
        self._active_completion_rollout_len = 1
        self._active_chaos_reach_threshold = 0.0
        self._initial_chaos_reach_threshold = 0.0
        self._active_chaos_adaptive = False
        self._active_chaos_max_reach_threshold = 0.0
        self._active_chaos_growth_factor = 2.0
        self._active_chaos_stagnation_rounds = 2
        self._manual_chaos_frontier_ids = set()
        self._manual_frontier_merges = {}
        self._active_identification_budget = 1
        self._active_resample_hits = 1
        self._base_apartness_n_c = 0
        self._base_identification_budget = 1
        self._base_apartness_max_depth = None
        self._apartness_n_c_cap = 0
        self._identification_budget_cap = 64
        self._apartness_max_depth_cap = None
        self._identification_escalations = 0
        self._max_identification_escalations = 12

    def configure_learning(
        self,
        apartness_checker=None,
        diff_checker=None,
        n_c=0,
        apartness_n_c=None,
        completion_max_attempts=None,
        store_failed_prefixes=True,
        completion_rollout_len=1,
        identification_budget=None,
        resample_hits=None,
        chaos_reach_threshold=0.0,
        chaos_adaptive=False,
        chaos_max_reach_threshold=None,
        chaos_growth_factor=2.0,
        chaos_stagnation_rounds=2,
    ):
        self._active_apartness_checker = apartness_checker
        self._active_diff_checker = diff_checker
        self._active_n_c = int(n_c or 0)
        self._active_apartness_n_c = int(apartness_n_c if apartness_n_c is not None else self._active_n_c)
        self._active_completion_max_attempts = completion_max_attempts
        self._active_store_failed_prefixes = bool(store_failed_prefixes)
        self._active_completion_rollout_len = int(completion_rollout_len or 1)
        self._active_chaos_reach_threshold = max(0.0, float(chaos_reach_threshold or 0.0))
        self._initial_chaos_reach_threshold = self._active_chaos_reach_threshold
        self._active_chaos_adaptive = bool(chaos_adaptive)
        if chaos_max_reach_threshold is None:
            chaos_max_reach_threshold = self._active_chaos_reach_threshold
        self._active_chaos_max_reach_threshold = max(
            self._active_chaos_reach_threshold,
            float(chaos_max_reach_threshold or 0.0),
        )
        self._active_chaos_growth_factor = max(1.0, float(chaos_growth_factor or 1.0))
        self._active_chaos_stagnation_rounds = max(1, int(chaos_stagnation_rounds or 1))
        if identification_budget is None:
            identification_budget = max(1, min(5, self._active_apartness_n_c if self._active_apartness_n_c > 0 else 1))
        self._active_identification_budget = int(identification_budget)
        if resample_hits is None:
            resample_hits = max(1, self._active_apartness_n_c)
        self._active_resample_hits = max(1, int(resample_hits or 1))
        self._base_apartness_n_c = self._active_apartness_n_c
        self._base_identification_budget = self._active_identification_budget
        self._base_apartness_max_depth = getattr(apartness_checker, 'max_depth', None) if apartness_checker is not None else None
        self._apartness_n_c_cap = max(self._base_apartness_n_c + 6, self._active_n_c + 4 if self._active_n_c > 0 else self._base_apartness_n_c + 6)
        self._identification_budget_cap = max(self._base_identification_budget * 8, 64)
        if self._base_apartness_max_depth is None:
            self._apartness_max_depth_cap = None
        else:
            # Small caps such as 6 work on toy examples, but they are too low for
            # grid benchmarks where some state pairs are only distinguished after
            # longer paths to wall/goal regions. Use a much looser cap.
            self._apartness_max_depth_cap = max(16, self._base_apartness_max_depth + 12, self._base_apartness_max_depth * 4)
        self._identification_escalations = 0
        return self


    def _escalate_identification_settings(self, reason=None):
        if self._identification_escalations >= self._max_identification_escalations:
            return False

        changed = False
        old_budget = self._active_identification_budget
        old_n_c = self._active_apartness_n_c
        old_depth = getattr(self._active_apartness_checker, 'max_depth', None) if self._active_apartness_checker is not None else None

        if self._active_identification_budget < self._identification_budget_cap:
            self._active_identification_budget = min(
                self._identification_budget_cap,
                max(self._active_identification_budget + 1, self._active_identification_budget * 2),
            )
            changed = changed or (self._active_identification_budget != old_budget)

        if self._active_apartness_n_c > 0 and self._active_apartness_n_c < self._apartness_n_c_cap:
            self._active_apartness_n_c = min(self._apartness_n_c_cap, self._active_apartness_n_c + 1)
            changed = changed or (self._active_apartness_n_c != old_n_c)

        if self._active_apartness_checker is not None and hasattr(self._active_apartness_checker, 'max_depth'):
            current_depth = getattr(self._active_apartness_checker, 'max_depth', None)
            if current_depth is not None:
                if self._apartness_max_depth_cap is None or current_depth < self._apartness_max_depth_cap:
                    setattr(self._active_apartness_checker, 'max_depth', current_depth + 1)
                    changed = True

        if changed:
            self._identification_escalations += 1
            self._invalidate_witness_cache(reason=reason or 'identification_escalation')
            self._log(
                'identify_settings_escalated',
                reason=reason,
                escalation=self._identification_escalations,
                apartness_n_c=self._active_apartness_n_c,
                identification_budget=self._active_identification_budget,
                max_depth=getattr(self._active_apartness_checker, 'max_depth', None) if self._active_apartness_checker is not None else None,
            )
        return changed

    def set_debug(self, enabled=True, logger=None):
        self.debug = bool(enabled)
        if logger is not None:
            self._debug_logger = logger
        return self

    def snapshot_stats(self):
        return dict(self.debug_stats)

    def _frontier_identification_record(self, node):
        records = self.debug_stats.setdefault('identify_frontier_nodes', {})
        key = str(getattr(node, 'id', id(node)))
        record = records.setdefault(key, {
            'frontier_node_id': getattr(node, 'id', None),
            'calls': 0,
            'witness_found': 0,
            'missing_witness': 0,
            'sample_attempts': 0,
            'replay_matches': 0,
            'replay_failures': 0,
            'inserted_samples': 0,
            'candidate_reductions': 0,
            'no_progress': 0,
            'single_candidate_returns': 0,
            'zero_candidate_promotions': 0,
            'last_access_len': None,
            'last_frequency': None,
            'last_candidates_before': [],
            'last_candidates_after': [],
            'last_witness': None,
            'last_witness_len': None,
            'last_sample_stats': {},
            'last_reason': None,
        })
        record['last_access_len'] = len(self.get_access_sequence(node))
        record['last_frequency'] = getattr(node, 'frequency', None)
        return record

    def snapshot_adequacy_debug(self, unresolved_entries=None):
        unresolved_entries = unresolved_entries or self._unresolved_frontier_entries()
        unresolved = {}
        for node, candidates in unresolved_entries:
            unresolved[str(node.id)] = {
                'frontier_node_id': node.id,
                'access_len': len(self.get_access_sequence(node)),
                'frequency': getattr(node, 'frequency', None),
                'candidate_ids': [candidate.id for candidate in candidates],
                'reach_probability': self.estimate_reach_probability(node),
                'identification_stats': self.debug_stats.get('identify_frontier_nodes', {}).get(str(node.id), {}),
            }
        return {
            'tree_stats': self.snapshot_stats(),
            'unresolved_frontiers': unresolved,
            'basis_size': len(self.basis),
            'frontier_size': len(self.frontier_to_basis_dict),
            'active_identification_budget': self._active_identification_budget,
            'active_resample_hits': self._active_resample_hits,
            'active_apartness_n_c': self._active_apartness_n_c,
            'active_completion_max_attempts': self._active_completion_max_attempts,
            'active_chaos_reach_threshold': self._active_chaos_reach_threshold,
        }

    def _log(self, event, message=None, **data):
        if not getattr(self, 'debug', False):
            return
        payload = {
            'ts': datetime.now().strftime('%H:%M:%S'),
            'component': 'observation_tree',
            'event': event,
        }
        payload.update(data)
        line = f"[{payload['ts']}] [observation_tree] {event}"
        if message:
            line += f": {message}"
        if data:
            ordered = ', '.join(f"{k}={payload[k]!r}" for k in data.keys())
            line += f" | {ordered}"
        logger = getattr(self, '_debug_logger', None)
        if logger is not None:
            logger(payload, line)
        else:
            print(line)

    def _invalidate_witness_cache(self, reason=None):
        if self.witness_cache:
            self.witness_cache.clear()
            self._log('witness_cache_cleared', reason=reason)

    # ------------------------------------------------------------------
    # Trace storage / navigation
    # ------------------------------------------------------------------
    def insert_observation(self, inputs, outputs):
        assert len(inputs) == len(outputs)
        node = self.root
        for inp, out in zip(inputs, outputs):
            node.input_frequencies[inp] += 1
            node.output_frequencies[inp][out] += 1
            successor = node.get_successor(inp, out)
            if successor is None:
                successor = StochasticMealyNode(
                    output=out,
                    parent=node,
                    input_to_parent=inp,
                )
                node.successors[inp][out] = successor
            node = successor
            node.frequency += 1
        if inputs:
            self._invalidate_witness_cache(reason='new_observation')
        return node

    def insert_trace(self, io_trace):
        inputs = io_trace[0::2]
        outputs = io_trace[1::2]
        return self.insert_observation(inputs, outputs)

    def get_successor(self, io_trace):
        node = self.root
        for inp, out in zip(io_trace[0::2], io_trace[1::2]):
            node = node.get_successor(inp, out)
            if node is None:
                return None
        return node

    def get_output_frequencies(self, node, inp):
        return node.get_output_frequencies(inp)

    def get_access_sequence(self, node):
        seq = []
        while node is not self.root:
            seq.append(node.input_to_parent)
            node = node.parent
        seq.reverse()
        return tuple(seq)

    def get_io_trace(self, node):
        trace = []
        while node is not self.root:
            trace.append(node.output)
            trace.append(node.input_to_parent)
            node = node.parent
        trace.reverse()
        return tuple(trace)

    def get_observed_successors(self, node):
        observed = set()
        for out_map in node.successors.values():
            observed.update(out_map.values())
        return observed

    def is_frontier_node(self, node):
        return node in self.frontier_to_basis_dict

    def is_basis_or_frontier_node(self, node):
        return node in self.basis or node in self.frontier_to_basis_dict

    # ------------------------------------------------------------------
    # LLM serialization helpers (read-only)
    # ------------------------------------------------------------------
    def _node_role(self, node, basis_set=None, frontier_set=None):
        if basis_set is None:
            basis_set = set(self.basis)
        if frontier_set is None:
            frontier_set = set(self.frontier_to_basis_dict.keys())
        if node in basis_set:
            return 'basis'
        if node in frontier_set:
            return 'frontier'
        return 'remain'

    def iter_nodes_with_roles(self, max_depth=None, max_nodes=None, frontier_suffix_depth=None):
        """
        Iterate over currently observed tree nodes as ``(node, role, depth)``.

        This method is intentionally read-only: it does not refresh Basis /
        Frontier metadata and therefore cannot affect the learner state.
        Node roles are derived from the *current* partition snapshot.
        """
        basis_set = set(self.basis)
        frontier_set = set(self.frontier_to_basis_dict.keys())
        suffix_limit = None if frontier_suffix_depth is None else int(frontier_suffix_depth)
        stack = [(self.root, 0, None)]
        visited = set()
        emitted = 0

        while stack:
            node, depth, frontier_suffix = stack.pop()
            if node is None or node.id in visited:
                continue
            visited.add(node.id)

            role = self._node_role(node, basis_set=basis_set, frontier_set=frontier_set)
            if frontier_suffix is None and role == 'frontier':
                frontier_suffix = 0
            yield node, role, depth
            emitted += 1

            if max_nodes is not None and emitted >= max_nodes:
                return
            if max_depth is not None and depth >= max_depth:
                continue
            if suffix_limit is not None and frontier_suffix is not None and frontier_suffix >= suffix_limit:
                continue

            for inp in sorted(node.successors.keys(), key=repr, reverse=True):
                out_map = node.successors[inp]
                for out in sorted(out_map.keys(), key=repr, reverse=True):
                    child_suffix = None if frontier_suffix is None else frontier_suffix + 1
                    stack.append((out_map[out], depth + 1, child_suffix))

    def _serialize_transition_for_llm(self, node, inp, n_c=0):
        total = node.get_frequency_sum(inp)
        outputs = []
        for out, count in sorted(node.get_output_frequencies(inp).items(), key=lambda item: repr(item[0])):
            succ = node.get_successor(inp, out)
            outputs.append({
                'output': out,
                'count': count,
                'prob_emp': round(count / total, 6) if total > 0 else 0.0,
                'successor_id': succ.id if succ is not None else None,
            })
        return {
            'sample_count': total,
            'complete': bool(total >= n_c) if n_c > 0 else True,
            'outputs': outputs,
        }

    def serialize_node_for_llm(self, node, role=None, depth=None, n_c=None):
        """
        Serialize a single observed node into a JSON-compatible dictionary.

        The serialization is read-only and does not trigger extra sampling,
        apartness checks, or partition updates.
        """
        if role is None:
            role = self._node_role(node)
        if depth is None:
            depth = len(self.get_access_sequence(node))
        if n_c is None:
            n_c = int(getattr(self, '_active_n_c', 0) or 0)

        transitions = {
            inp: self._serialize_transition_for_llm(node, inp, n_c=n_c)
            for inp in self.alphabet
        }

        candidate_basis_ids = None
        if role == 'frontier':
            candidate_basis_ids = [
                basis_node.id
                for basis_node in sorted(
                    self.frontier_to_basis_dict.get(node, set()),
                    key=lambda cand: cand.id,
                )
            ]

        return {
            'id': node.id,
            'role': role,
            'depth': depth,
            'access_sequence': list(self.get_access_sequence(node)),
            'io_trace': list(self.get_io_trace(node)),
            'last_output': node.output,
            'frequency': getattr(node, 'frequency', 0),
            'frontier_candidate_basis_ids': candidate_basis_ids,
            'transitions': transitions,
        }

    def _serialize_witness_cache_for_llm(self):
        serialized = []
        for key, witness in sorted(self.witness_cache.items(), key=lambda item: item[0]):
            if not isinstance(key, tuple) or len(key) < 2:
                continue
            entry = {
                'pair': list(key[:2]),
                'witness': list(witness) if witness is not None else None,
            }
            if len(key) > 2:
                entry['context'] = list(key[2:])
            serialized.append(entry)
        return serialized

    def get_node_by_id(self, node_id, *, max_depth=None, max_nodes=None, frontier_suffix_depth=None):
        """
        Return the observed tree node with the given id, or None if it is not
        currently reachable in the serialized observation tree view.

        This is a read-only navigation helper for LLM-side validation and does
        not refresh the basis/frontier partition.
        """
        for node, _, _ in self.iter_nodes_with_roles(
            max_depth=max_depth,
            max_nodes=max_nodes,
            frontier_suffix_depth=frontier_suffix_depth,
        ):
            if node.id == node_id:
                return node
        return None

    def serialize_learning_parameters_for_llm(self):
        """
        Serialize observation-tree-owned learning parameters only.

        Query-planning controls such as remaining budget, allowed purposes,
        query mode, and maximum query length are intentionally not included
        here. They should be added by LLMQuerierPromptBuilder or the outer
        learning controller.
        """
        return {
            'active_n_c': int(getattr(self, '_active_n_c', 0) or 0),
            'active_apartness_n_c': int(getattr(self, '_active_apartness_n_c', 0) or 0),
            'completion_max_attempts': getattr(self, '_active_completion_max_attempts', None),
            'completion_rollout_len': int(getattr(self, '_active_completion_rollout_len', 1) or 1),
            'identification_budget': int(getattr(self, '_active_identification_budget', 1) or 1),
            'resample_hits': int(getattr(self, '_active_resample_hits', 1) or 1),
            'chaos_reach_threshold': float(getattr(self, '_active_chaos_reach_threshold', 0.0) or 0.0),
            'chaos_initial_reach_threshold': float(getattr(self, '_initial_chaos_reach_threshold', 0.0) or 0.0),
            'chaos_adaptive': bool(getattr(self, '_active_chaos_adaptive', False)),
            'chaos_max_reach_threshold': float(getattr(self, '_active_chaos_max_reach_threshold', 0.0) or 0.0),
            'chaos_growth_factor': float(getattr(self, '_active_chaos_growth_factor', 1.0) or 1.0),
            'chaos_stagnation_rounds': int(getattr(self, '_active_chaos_stagnation_rounds', 1) or 1),
        }

    def serialize_unresolved_frontier_for_llm(self):
        """
        Return frontier nodes whose current basis representative is not unique.

        A frontier node with zero candidates is isolated under the current
        evidence; a frontier node with multiple candidates is ambiguous.
        """
        unresolved = []
        for node, candidates in sorted(self.frontier_to_basis_dict.items(), key=lambda item: item[0].id):
            candidate_ids = [
                candidate.id
                for candidate in sorted(candidates, key=lambda cand: cand.id)
            ]
            if len(candidate_ids) == 1:
                continue
            unresolved.append({
                'node_id': node.id,
                'role': 'frontier',
                'access_sequence': list(self.get_access_sequence(node)),
                'io_trace': list(self.get_io_trace(node)),
                'candidate_basis_ids': candidate_ids,
                'status': 'isolated' if len(candidate_ids) == 0 else 'ambiguous',
            })
        return unresolved

    def serialize_apartness_witness_for_llm(self):
        """
        Return established apartness witnesses in a prompt-friendly schema.

        Only non-empty cached witnesses are exported. A cached None means no
        witness has been found under current limits and must not be treated as
        apartness evidence.
        """
        witnesses = []
        seen = set()
        for key, witness in sorted(self.witness_cache.items(), key=lambda item: repr(item[0])):
            if not witness:
                continue
            if not isinstance(key, tuple) or len(key) < 2:
                continue
            pair = tuple(sorted((key[0], key[1])))
            if pair in seen:
                continue
            seen.add(pair)
            witnesses.append({
                'state_id_1': pair[0],
                'state_id_2': pair[1],
                'witness': list(witness),
            })
        return witnesses

    def _collect_incomplete_for_llm(self, *, max_depth=None, max_nodes=None, frontier_suffix_depth=None):
        """
        Return compact node-input pairs whose one-step observations have not
        reached the active completion threshold.

        This helper is read-only and is intended for Querier-side planning
        context, not for the core observation-tree evidence snapshot.
        """
        active_n_c = int(getattr(self, '_active_n_c', 0) or 0)
        if active_n_c <= 0:
            return []

        incomplete = []
        for node, role, _ in self.iter_nodes_with_roles(
            max_depth=max_depth,
            max_nodes=max_nodes,
            frontier_suffix_depth=frontier_suffix_depth,
        ):
            for inp in self.alphabet:
                observed = node.get_frequency_sum(inp)
                if observed < active_n_c:
                    incomplete.append({
                        'node_id': node.id,
                        'role': role,
                        'input': inp,
                        'input_suffix': [inp],
                        'missing': active_n_c - observed,
                    })
        return incomplete

    def serialize_for_llm(
        self,
        *,
        max_depth=None,
        max_nodes=None,
        frontier_suffix_depth=None,
        include_learning_parameters=True,
    ):
        """
        Return a read-only, JSON-compatible snapshot of the current
        observation tree itself.

        This snapshot is intentionally limited to observation-tree evidence:
        metadata, tree-owned learning parameters, and serialized nodes with
        empirical transition summaries. Querier-specific planning signals such
        as incomplete branches, unresolved frontier summaries, apartness
        witnesses, remaining budget, allowed purposes, and query mode are not
        included here. They should be generated separately, immediately before
        invoking the Querier.

        This method does not modify the learner state: it does not sample, does
        not recompute frontier candidates, and does not compute new witnesses.
        """
        active_n_c = int(getattr(self, '_active_n_c', 0) or 0)
        active_apartness_n_c = int(getattr(self, '_active_apartness_n_c', 0) or 0)

        nodes = []
        role_counts = {
            'basis': 0,
            'frontier': 0,
            'remain': 0,
        }

        for node, role, depth in self.iter_nodes_with_roles(
            max_depth=max_depth,
            max_nodes=max_nodes,
            frontier_suffix_depth=frontier_suffix_depth,
        ):
            role_counts[role] += 1
            nodes.append(self.serialize_node_for_llm(node, role=role, depth=depth, n_c=active_n_c))

        snapshot = {
            'meta': {
                'automaton_type': self.automaton_type,
                'alphabet': list(self.alphabet),
                'root_id': self.root.id,
                'active_n_c': active_n_c,
                'active_apartness_n_c': active_apartness_n_c,
                'basis_size': len(self.basis),
                'frontier_size': len(self.frontier_to_basis_dict),
                'serialized_node_count': len(nodes),
                'role_counts': role_counts,
                'max_depth': max_depth,
                'max_nodes': max_nodes,
                'frontier_suffix_depth': frontier_suffix_depth,
            },
            'nodes': nodes,
        }

        if include_learning_parameters:
            snapshot['learning_parameters'] = self.serialize_learning_parameters_for_llm()

        return snapshot

    def serialize_querier_context_for_llm(
        self,
        *,
        max_depth=None,
        max_nodes=None,
        frontier_suffix_depth=None,
        include_incomplete=True,
        include_unresolved_frontier=True,
        include_apartness_witness=True,
    ):
        """
        Return a compact, read-only context object for LLM Querier planning.

        This context contains derived planning signals that are useful only
        when the learner is about to call the Querier. It is deliberately kept
        outside ``serialize_for_llm`` so the observation-tree snapshot remains
        a factual tree evidence document.
        """
        context = {
            'meta': {
                'root_id': self.root.id,
                'alphabet': list(self.alphabet),
                'active_n_c': int(getattr(self, '_active_n_c', 0) or 0),
                'active_apartness_n_c': int(getattr(self, '_active_apartness_n_c', 0) or 0),
                'max_depth': max_depth,
                'max_nodes': max_nodes,
                'frontier_suffix_depth': frontier_suffix_depth,
            }
        }

        if include_incomplete:
            context['incomplete'] = self._collect_incomplete_for_llm(
                max_depth=max_depth,
                max_nodes=max_nodes,
                frontier_suffix_depth=frontier_suffix_depth,
            )
        if include_unresolved_frontier:
            context['unresolved_frontier'] = self.serialize_unresolved_frontier_for_llm()
        if include_apartness_witness:
            context['apartness_witness'] = self.serialize_apartness_witness_for_llm()

        return context

    @staticmethod
    def _default_serialization_output_dir():
        """Return the directory that contains this source file."""
        return Path(__file__).resolve().parent

    @classmethod
    def _default_serialized_tree_path(cls):
        return cls._default_serialization_output_dir() / 'serialized_observation_tree.json'

    @classmethod
    def _default_querier_context_path(cls):
        return cls._default_serialization_output_dir() / 'querier_context.json'

    @staticmethod
    def _save_json_document(payload, path, *, indent=2, ensure_ascii=False):
        output_path = Path(path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(payload, f, indent=indent, ensure_ascii=ensure_ascii, sort_keys=True)
            f.write('\n')
        return output_path

    def save_serialized_for_llm(self, **serialize_kwargs):
        """
        Save ``serialize_for_llm(...)`` to ``serialized_observation_tree.json``
        in the same directory as this Python file and return the absolute path.
        """
        payload = self.serialize_for_llm(**serialize_kwargs)
        return self._save_json_document(payload, self._default_serialized_tree_path())

    def save_querier_context_for_llm(self, **serialize_kwargs):
        """
        Save ``serialize_querier_context_for_llm(...)`` to
        ``querier_context.json`` in the same directory as this Python file and
        return the absolute path. Call this immediately before invoking
        LLMQuerier.
        """
        payload = self.serialize_querier_context_for_llm(**serialize_kwargs)
        return self._save_json_document(payload, self._default_querier_context_path())

    # ------------------------------------------------------------------
    # Sampling support
    # ------------------------------------------------------------------
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

    def _snapshot_sul_stats(self):
        if self.sul is None:
            return {}
        if hasattr(self.sul, 'snapshot_stats'):
            snap = self.sul.snapshot_stats()
            return dict(snap) if snap is not None else {}
        stats = getattr(self.sul, 'stats', None)
        if isinstance(stats, dict):
            return dict(stats)
        return {}

    def _current_sampling_budget_value(self):
        stats = self._snapshot_sul_stats()
        unit = getattr(self, '_sampling_budget_unit', 'episodes')
        if unit == 'steps':
            return int(stats.get('steps', stats.get('num_steps', 0)) or 0)
        return int(stats.get('episodes', stats.get('queries', stats.get('num_queries', 0))) or 0)

    def _check_sampling_budget(self):
        limit = getattr(self, '_sampling_budget_limit', None)
        if limit is None:
            return
        current = self._current_sampling_budget_value()
        if current >= int(limit):
            self.debug_stats['sampling_budget_interruptions'] += 1
            raise SamplingBudgetExceeded(
                current_samples=current,
                limit=int(limit),
                unit=getattr(self, '_sampling_budget_unit', 'episodes'),
            )

    def _replay_io_trace(self, io_trace):
        observed = []
        for exp_inp, exp_out in zip(io_trace[0::2], io_trace[1::2]):
            if getattr(self, '_sampling_budget_unit', 'episodes') == 'steps':
                self._check_sampling_budget()
            obs_out = self.sul.step(exp_inp)
            observed.extend([exp_inp, obs_out])
            if obs_out != exp_out:
                return tuple(observed), False
        return tuple(observed), True

    def _sample_rollout_suffix(self, steps):
        if steps <= 0:
            return tuple()
        suffix = []
        for _ in range(steps):
            rollout_inp = choice(self.alphabet)
            if getattr(self, '_sampling_budget_unit', 'episodes') == 'steps':
                self._check_sampling_budget()
            rollout_out = self.sul.step(rollout_inp)
            suffix.extend([rollout_inp, rollout_out])
        self.debug_stats['ensure_complete_rollout_steps'] += steps
        self.debug_stats['ensure_complete_rollout_traces'] += 1
        return tuple(suffix)

    def _extend_input_sequence_for_refinement(self, inputs, extra_steps):
        seq = list(inputs or ())
        extra_steps = max(0, int(extra_steps or 0))
        if extra_steps <= 0 or not self.alphabet:
            return tuple(seq)
        alphabet = list(self.alphabet)
        start = 0
        if seq and seq[-1] in alphabet:
            start = (alphabet.index(seq[-1]) + 1) % len(alphabet)
        for offset in range(extra_steps):
            seq.append(alphabet[(start + offset) % len(alphabet)])
        return tuple(seq)

    def _register_refinement_target(
        self,
        inputs,
        *,
        reason,
        mode=None,
        node=None,
        inp=None,
        witness=None,
        required_hits=None,
        min_attempts=None,
        max_attempts=None,
        samples_per_target=None,
    ):
        inputs = tuple(inputs or ())
        if not inputs:
            return False
        if mode is None:
            mode = 'completion' if reason == 'complete_cell' else 'resample'
        mode = str(mode or 'resample')
        key = (mode, inputs)
        requested_hits = required_hits
        # Backward-compatible alias. Older callers used samples_per_target as
        # a query count; new refinement semantics interpret it as desired
        # effective hits on the target cell.
        if requested_hits is None:
            requested_hits = samples_per_target
        if requested_hits is not None:
            try:
                requested_hits = max(1, int(requested_hits))
            except (TypeError, ValueError):
                requested_hits = None
        requested_attempts = max_attempts
        requested_min_attempts = min_attempts
        if requested_min_attempts is not None:
            try:
                requested_min_attempts = max(0, int(requested_min_attempts))
            except (TypeError, ValueError):
                requested_min_attempts = None
        if requested_attempts is not None:
            try:
                requested_attempts = max(1, int(requested_attempts))
            except (TypeError, ValueError):
                requested_attempts = None
        if samples_per_target is not None:
            try:
                samples_per_target = max(1, int(samples_per_target))
            except (TypeError, ValueError):
                samples_per_target = None
        if key in self._pending_refinement_targets:
            self.debug_stats['refinement_target_duplicates'] += 1
            existing = self._pending_refinement_targets[key]
            existing_hits = existing.get('required_hits')
            existing_min_attempts = existing.get('min_attempts')
            existing_attempts = existing.get('max_attempts')
            changed = False
            if requested_hits is not None and (existing_hits is None or requested_hits > existing_hits):
                existing['required_hits'] = requested_hits
                existing['samples_per_target'] = requested_hits
                changed = True
            if requested_min_attempts is not None and (
                existing_min_attempts is None or requested_min_attempts > existing_min_attempts
            ):
                existing['min_attempts'] = requested_min_attempts
                changed = True
            if requested_attempts is not None and (existing_attempts is None or requested_attempts > existing_attempts):
                existing['max_attempts'] = requested_attempts
                changed = True
            if changed:
                existing['reason'] = reason
                existing['mode'] = mode
                self._log(
                    'refinement_target_updated',
                    reason=reason,
                    mode=mode,
                    node_id=getattr(node, 'id', None),
                    inp=inp,
                    inputs=inputs,
                    required_hits=existing.get('required_hits'),
                    min_attempts=existing.get('min_attempts'),
                    max_attempts=existing.get('max_attempts'),
                )
                return True
            return False
        self._pending_refinement_targets[key] = {
            'inputs': inputs,
            'reason': reason,
            'mode': mode,
            'node_ref': node,
            'node_id': getattr(node, 'id', None),
            'inp': inp,
            'witness': tuple(witness) if witness is not None else None,
            'start_count': node.get_frequency_sum(inp) if node is not None and inp is not None else None,
            'required_hits': requested_hits,
            'min_attempts': requested_min_attempts,
            'max_attempts': requested_attempts,
            'samples_per_target': requested_hits,
        }
        self.debug_stats['refinement_targets_registered'] += 1
        self._log(
            'refinement_target_registered',
            reason=reason,
            mode=mode,
            node_id=getattr(node, 'id', None),
            inp=inp,
            inputs=inputs,
            witness=tuple(witness) if witness is not None else None,
            required_hits=requested_hits,
            min_attempts=requested_min_attempts,
            max_attempts=requested_attempts,
        )
        return True

    def sample_cell_hits(
        self,
        node,
        inp,
        required_hits,
        *,
        reason,
        mode='resample',
        inputs=None,
        min_attempts=None,
        max_attempts=None,
        completion_rollout_len=1,
        witness=None,
        start_count=None,
        store_failed_prefixes=True,
        count_as_completion=False,
    ):
        """Sample a concrete node/input cell until it gains required_hits hits.

        The caller decides whether sampling is needed and passes the number of
        additional effective hits. This method deliberately does not compare
        the cell against n_c; it only measures growth from its own start count.
        """
        try:
            required_hits = int(required_hits or 0)
        except (TypeError, ValueError):
            required_hits = 0
        required_hits = max(0, required_hits)
        if max_attempts is None:
            max_attempts = max(
                required_hits,
                int(getattr(self, '_active_identification_budget', 1) or 1),
            )
        try:
            max_attempts = max(0, int(max_attempts or 0))
        except (TypeError, ValueError):
            max_attempts = max(
                required_hits,
                int(getattr(self, '_active_identification_budget', 1) or 1),
            )
        # ``n_resample`` is the number of additional effective hits requested
        # for a target, not a lower bound on replay attempts.
        min_attempts = 0
        completion_rollout_len = max(1, int(completion_rollout_len or 1))

        if inputs is None:
            suffix_inputs = tuple(witness or ())
            if not suffix_inputs:
                inputs = self.get_access_sequence(node) + (inp,)
                inputs = self._extend_input_sequence_for_refinement(
                    inputs,
                    completion_rollout_len - 1,
                )
            else:
                if suffix_inputs[0] != inp:
                    suffix_inputs = (inp,) + suffix_inputs
                inputs = self.get_access_sequence(node) + suffix_inputs
        inputs = tuple(inputs or ())

        if start_count is None:
            start_count = node.get_frequency_sum(inp) if node is not None and inp is not None else 0
        else:
            try:
                start_count = int(start_count)
            except (TypeError, ValueError):
                start_count = node.get_frequency_sum(inp) if node is not None and inp is not None else 0
        result = {
            'node_id': getattr(node, 'id', None),
            'inp': inp,
            'inputs': inputs,
            'reason': reason,
            'mode': mode,
            'required_hits': required_hits,
            'min_attempts': min_attempts,
            'max_attempts': max_attempts,
            'attempts': 0,
            'queries': 0,
            'matched_attempts': 0,
            'failed_replays': 0,
            'start_count': start_count,
            'final_count': start_count,
            'hits_gained': 0,
            'completed': required_hits <= 0,
            'aborted': False,
        }

        if required_hits <= 0 or node is None or inp is None or not inputs:
            self._log(
                'sample_cell_hits_skipped',
                reason=reason,
                mode=mode,
                node_id=getattr(node, 'id', None),
                inp=inp,
                inputs=inputs,
                required_hits=required_hits,
            )
            return result

        self._log(
            'sample_cell_hits_start',
            reason=reason,
            mode=mode,
            node_id=getattr(node, 'id', None),
            inp=inp,
            inputs=inputs,
            witness=tuple(witness) if witness is not None else None,
            start_count=start_count,
            required_hits=required_hits,
            min_attempts=min_attempts,
            max_attempts=max_attempts,
        )

        attempts = 0
        queries = 0
        matched_attempts = 0
        failed_replays = 0
        while attempts < max_attempts:
            current = node.get_frequency_sum(inp)
            if current - start_count >= required_hits:
                break
            before_attempt = current
            self._sample_query_once(inputs, store_failed_prefixes=store_failed_prefixes)
            attempts += 1
            queries += 1
            after_attempt = node.get_frequency_sum(inp)
            if after_attempt > before_attempt:
                matched_attempts += 1
            else:
                failed_replays += 1

        final_count = node.get_frequency_sum(inp)
        hits_gained = max(0, final_count - start_count)
        completed = hits_gained >= required_hits
        result.update({
            'attempts': attempts,
            'queries': queries,
            'matched_attempts': matched_attempts,
            'failed_replays': failed_replays,
            'final_count': final_count,
            'hits_gained': hits_gained,
            'completed': completed,
            'aborted': not completed and attempts >= max_attempts,
        })
        if count_as_completion:
            self.debug_stats['ensure_complete_attempts'] += attempts
            self.debug_stats['ensure_complete_matched_attempts'] += matched_attempts
            self.debug_stats['ensure_complete_failed_replays'] += failed_replays
        self._log(
            'sample_cell_hits_done',
            reason=reason,
            mode=mode,
            node_id=getattr(node, 'id', None),
            inp=inp,
            attempts=attempts,
            matched_attempts=matched_attempts,
            failed_replays=failed_replays,
            start_count=start_count,
            final_count=final_count,
            gained=hits_gained,
            required_hits=required_hits,
            completed=completed,
            aborted=result['aborted'],
        )
        return result

    def sample_node_input(
        self,
        node,
        inp,
        required_hits,
        *,
        reason,
        mode='resample',
        min_attempts=None,
        max_attempts=None,
        completion_rollout_len=1,
        witness=None,
    ):
        """Register a target that will add required_hits hits when executed."""
        try:
            required_hits = int(required_hits or 0)
        except (TypeError, ValueError):
            required_hits = 0
        if required_hits <= 0:
            return False
        completion_rollout_len = max(1, int(completion_rollout_len or 1))
        inputs = self.get_access_sequence(node) + (inp,)
        inputs = self._extend_input_sequence_for_refinement(
            inputs,
            completion_rollout_len - 1,
        )
        if max_attempts is None:
            if mode == 'completion':
                max_attempts = getattr(self, '_active_completion_max_attempts', None)
                if max_attempts is None:
                    max_attempts = getattr(self, '_active_identification_budget', None)
            else:
                max_attempts = getattr(self, '_active_identification_budget', None)
        return self._register_refinement_target(
            inputs,
            reason=reason,
            mode=mode,
            node=node,
            inp=inp,
            witness=witness,
            required_hits=required_hits,
            max_attempts=max_attempts,
            min_attempts=min_attempts,
        )

    def _pop_refinement_targets(self):
        targets = list(self._pending_refinement_targets.values())
        self._pending_refinement_targets.clear()
        return targets

    def _run_refinement_targets(self, targets, *, samples_per_target=None):
        targets = list(targets or [])
        if not targets:
            return {
                'target_count': 0,
                'queries': 0,
                'attempts': 0,
                'hits_gained': 0,
                'targets_improved': 0,
                'targets_completed': 0,
            }
        default_required_hits = max(1, int(getattr(self, '_active_resample_hits', 1) or 1))
        if samples_per_target is not None:
            default_required_hits = max(1, int(samples_per_target or 1))
        default_max_attempts = max(1, int(getattr(self, '_active_identification_budget', 1) or 1))

        before_counts = {}
        for target in targets:
            node_id = target.get('node_id')
            inp = target.get('inp')
            if node_id is None or inp is None:
                continue
            node = target.get('node_ref') or self.get_node_by_id(node_id)
            if node is not None:
                before_counts[(node_id, inp)] = node.get_frequency_sum(inp)

        queries = 0
        attempts = 0
        total_hits_gained = 0
        targets_improved = 0
        targets_completed = 0
        self.debug_stats['refinement_rounds'] += 1
        max_required_hits = default_required_hits
        max_attempts_used = default_max_attempts
        for target in targets:
            required_hits = target.get('required_hits')
            if required_hits is None:
                required_hits = target.get('samples_per_target')
            if required_hits is None:
                required_hits = default_required_hits
            try:
                required_hits = max(1, int(required_hits))
            except (TypeError, ValueError):
                required_hits = default_required_hits
            max_attempts = target.get('max_attempts')
            if max_attempts is None:
                max_attempts = default_max_attempts
            try:
                max_attempts = max(1, int(max_attempts))
            except (TypeError, ValueError):
                max_attempts = default_max_attempts
            min_attempts = 0
            target['required_hits'] = required_hits
            target['min_attempts'] = min_attempts
            target['max_attempts'] = max_attempts
            target['samples_per_target'] = required_hits
            max_required_hits = max(max_required_hits, required_hits)
            max_attempts_used = max(max_attempts_used, max_attempts)
            inputs = tuple(target.get('inputs') or ())
            if not inputs:
                continue
            node_id = target.get('node_id')
            inp = target.get('inp')
            node = target.get('node_ref') or self.get_node_by_id(node_id)
            start_count = before_counts.get((node_id, inp))
            if node is None:
                target_attempts = 0
                fallback_attempts = min(max_attempts, required_hits)
                while target_attempts < fallback_attempts:
                    self._sample_query_once(inputs)
                    queries += 1
                    attempts += 1
                    target_attempts += 1
                continue

            result = self.sample_cell_hits(
                node,
                inp,
                required_hits,
                reason=target.get('reason'),
                mode=target.get('mode') or 'resample',
                inputs=inputs,
                witness=target.get('witness'),
                min_attempts=min_attempts,
                max_attempts=max_attempts,
                start_count=start_count,
                store_failed_prefixes=getattr(self, '_active_store_failed_prefixes', True),
                count_as_completion=(target.get('mode') == 'completion'),
            )
            queries += result.get('queries', 0)
            attempts += result.get('attempts', 0)
            gained = max(0, int(result.get('hits_gained', 0) or 0))
            total_hits_gained += gained
            if gained > 0:
                targets_improved += 1
            if result.get('completed'):
                targets_completed += 1

        self.debug_stats['refinement_queries'] += queries
        self.debug_stats['refinement_attempts'] += attempts
        self.debug_stats['refinement_targets_improved'] += targets_improved
        self.debug_stats['refinement_targets_completed'] += targets_completed
        self.debug_stats['refinement_target_hits'] += total_hits_gained
        self._log(
            'refinement_round_done',
            target_count=len(targets),
            max_required_hits=max_required_hits,
            max_attempts=max_attempts_used,
            queries=queries,
            attempts=attempts,
            hits_gained=total_hits_gained,
            targets_improved=targets_improved,
            targets_completed=targets_completed,
        )
        return {
            'target_count': len(targets),
            'queries': queries,
            'attempts': attempts,
            'hits_gained': total_hits_gained,
            'targets_improved': targets_improved,
            'targets_completed': targets_completed,
        }

    def is_complete(self, node, inp, n_c):
        assert n_c >= 0
        return node.get_frequency_sum(inp) >= n_c

    def ensure_complete(
        self,
        node,
        inp,
        n_c,
        max_attempts=None,
        store_failed_prefixes=True,
        completion_rollout_len=1,
    ):
        completion_rollout_len = max(1, int(completion_rollout_len))

        if not self.is_basis_or_frontier_node(node):
            self._log(
                'ensure_complete_skipped',
                node_id=node.id,
                node_trace=self.get_io_trace(node),
                inp=inp,
                reason='node_not_in_basis_or_frontier',
                target_count=n_c,
            )
            return node.get_frequency_sum(inp)

        self.debug_stats['ensure_complete_calls'] += 1

        start_count = node.get_frequency_sum(inp)
        target_prefix = self.get_io_trace(node)

        self._log(
            'ensure_complete_start',
            node_id=node.id,
            node_trace=target_prefix,
            inp=inp,
            start_count=start_count,
            target_count=n_c,
            max_attempts=max_attempts,
            completion_rollout_len=completion_rollout_len,
        )

        missing = max(0, int(n_c or 0) - start_count)
        if missing > 0:
            target_max_attempts = max_attempts
            if target_max_attempts is None:
                target_max_attempts = getattr(self, '_active_completion_max_attempts', None)
            if target_max_attempts is None:
                target_max_attempts = getattr(self, '_active_identification_budget', None)
            inputs = self.get_access_sequence(node) + (inp,)
            inputs = self._extend_input_sequence_for_refinement(
                inputs,
                completion_rollout_len - 1,
            )
            self._register_refinement_target(
                inputs,
                reason='complete_cell',
                mode='completion',
                node=node,
                inp=inp,
                required_hits=missing,
                max_attempts=target_max_attempts,
            )

        final_count = node.get_frequency_sum(inp)
        self._log(
            'ensure_complete_done',
            node_id=node.id,
            inp=inp,
            attempts=0,
            matched_attempts=0,
            failed_replays=0,
            start_count=start_count,
            final_count=final_count,
            gained=final_count - start_count,
            aborted=False,
            completion_rollout_len=completion_rollout_len,
        )
        return final_count

    def _sample_query_once(self, inputs, store_failed_prefixes=True):
        observed = []
        self._check_sampling_budget()
        self.sul.pre()
        try:
            for inp in inputs:
                if getattr(self, '_sampling_budget_unit', 'episodes') == 'steps':
                    self._check_sampling_budget()
                out = self.sul.step(inp)
                observed.extend([inp, out])
        finally:
            self.sul.post()
        io_trace = tuple(observed)
        if io_trace:
            self.insert_trace(io_trace)
        return io_trace

    def execute_odq(self, inputs, num_samples=1):
        """
        Execute an ordinary deterministic input query multiple times.

        Each execution runs the input sequence from the SUL initial state,
        records the observed I/O trace, and inserts that trace into the
        observation tree. This is the public sampling hook used by LLMQuerier
        and by learner-side file-driven query execution.
        """
        if self.sul is None:
            raise ValueError('Cannot execute query: observation tree has no SUL.')
        inputs = tuple(inputs or ())
        if not inputs:
            return []
        invalid = [inp for inp in inputs if inp not in self.alphabet]
        if invalid:
            raise ValueError(f'Input sequence contains symbols outside alphabet: {invalid!r}.')

        traces = []
        for _ in range(max(1, int(num_samples or 1))):
            traces.append(self._sample_query_once(inputs))
        return traces

    def execute_input_sequences(self, input_sequences, num_samples=1):
        """Execute a collection of input-only queries and insert all traces."""
        traces = []
        for inputs in input_sequences or []:
            traces.extend(self.execute_odq(tuple(inputs), num_samples=num_samples))
        return traces

    def execute_input_sequences_from_file(self, path, num_samples=1):
        """
        Load ``{"input_sequences": [[...], ...]}`` and execute all queries.

        This helper is intentionally small and file-format specific so the
        learner can consume LLMQuerier's final artifact directly.
        """
        input_path = Path(path).resolve()
        with input_path.open('r', encoding='utf-8') as f:
            payload = json.load(f)

        if isinstance(payload, dict):
            input_sequences = payload.get('input_sequences', [])
        elif isinstance(payload, list):
            input_sequences = payload
        else:
            raise ValueError(f'Unsupported input-sequence payload in {input_path!s}.')

        if not isinstance(input_sequences, list):
            raise ValueError(f'Expected input_sequences to be a list in {input_path!s}.')

        return self.execute_input_sequences(input_sequences, num_samples=num_samples)

    def set_guided_query_sequences(self, input_sequences=None, max_query_length=None):
        self.guided_query_sequences = [tuple(seq) for seq in (input_sequences or []) if seq]
        self.guided_query_max_length = int(max_query_length) if max_query_length else None
        self.guided_query_cursor = 0

    def _extend_inputs_to_guided_limit(self, prefix):
        seq = list(prefix or ())
        limit = self.guided_query_max_length
        if limit is None or limit <= 0 or len(seq) >= limit or not self.alphabet:
            return tuple(seq)
        alphabet = list(self.alphabet)
        start = 0
        if seq and seq[-1] in alphabet:
            start = (alphabet.index(seq[-1]) + 1) % len(alphabet)
        while len(seq) < limit:
            seq.append(alphabet[(start + len(seq) - len(prefix or ())) % len(alphabet)])
        return tuple(seq)

    def _choose_guided_sequence(self, required_prefix):
        required_prefix = tuple(required_prefix or ())
        if not required_prefix:
            return None
        sequences = list(getattr(self, 'guided_query_sequences', []) or [])
        for offset in range(len(sequences)):
            idx = (getattr(self, 'guided_query_cursor', 0) + offset) % len(sequences)
            seq = tuple(sequences[idx])
            if len(seq) >= len(required_prefix) and seq[:len(required_prefix)] == required_prefix:
                self.guided_query_cursor = idx + 1
                return seq
        return self._extend_inputs_to_guided_limit(required_prefix)

    def _execute_guided_input_sequence(self, required_prefix):
        seq = self._choose_guided_sequence(required_prefix)
        if not seq:
            return None
        traces = self.execute_odq(seq, num_samples=1)
        return traces[0] if traces else None

    def _sample_from_node_with_witness(self, node, witness, num_samples=1, sample_stats=None):
        if self.sul is None or not witness:
            return []
        if sample_stats is None:
            sample_stats = {}

        def inc(key):
            sample_stats[key] = sample_stats.get(key, 0) + 1

        sampled = []
        access_prefix = self.get_access_sequence(node)
        for _ in range(max(1, int(num_samples))):
            inc('attempts')
            self.debug_stats['identify_frontier_sample_attempts'] += 1
            if getattr(self, 'guided_query_max_length', None):
                inc('guided_attempts')
                trace = self._execute_guided_input_sequence(access_prefix + tuple(witness))
                if trace:
                    sampled.append(trace)
                    inc('inserted_samples')
                    inc('guided_inserted_samples')
                    self.debug_stats['identify_frontier_samples'] += 1
                    self.debug_stats['guided_frontier_queries'] += 1
                    continue
            full_trace = self._sample_query_once(access_prefix + tuple(witness))
            sampled.append(full_trace)
            inc('inserted_samples')
            self.debug_stats['identify_frontier_samples'] += 1
        return sampled

    # ------------------------------------------------------------------
    # Deterministic-LSharp-style frontier / basis management
    # ------------------------------------------------------------------
    def update_frontier(self):
        """
        Frontier is exactly the set of direct non-basis successors of basis nodes.
        All other observed nodes remain in the tree as remain nodes.
        """
        basis_set = set(self.basis)
        old_frontier = dict(self.frontier_to_basis_dict)
        new_frontier = {}
        for basis_node in self.basis:
            for succ in self.get_observed_successors(basis_node):
                if succ in basis_set:
                    continue
                new_frontier[succ] = set(old_frontier.get(succ, set()))
        self.frontier_to_basis_dict = new_frontier
        self.frontier = self.frontier_to_basis_dict
        self._log(
            'frontier_updated',
            basis_size=len(self.basis),
            frontier_size=len(self.frontier_to_basis_dict),
            frontier_node_ids=[node.id for node in self.frontier_to_basis_dict.keys()],
        )
        return self.frontier_to_basis_dict

    def _common_prefix_len(self, seq1, seq2):
        n = min(len(seq1), len(seq2))
        i = 0
        while i < n and seq1[i] == seq2[i]:
            i += 1
        return i

    def _ordered_basis_nodes_for_frontier(self, frontier_node):
        frontier_access = self.get_access_sequence(frontier_node)
        return sorted(
            self.basis,
            key=lambda basis_node: (
                -self._common_prefix_len(frontier_access, self.get_access_sequence(basis_node)),
                len(self.get_io_trace(basis_node)),
                -basis_node.frequency,
                basis_node.id,
            )
        )

    def update_basis_candidates(self, frontier_state):
        if frontier_state not in self.frontier_to_basis_dict:
            return set()
        candidates = self.find_basis_candidates(frontier_state)
        self.frontier_to_basis_dict[frontier_state] = candidates
        return candidates

    def update_frontier_to_basis_dict(self):
        # Recompute the exact frontier first: only direct successors of basis stay in frontier.
        self.update_frontier()
        for frontier_state in list(self.frontier_to_basis_dict.keys()):
            forced_basis_id = self._manual_frontier_merges.get(getattr(frontier_state, 'id', None))
            if forced_basis_id is not None:
                forced_basis = next((node for node in self.basis if node.id == forced_basis_id), None)
                if forced_basis is not None:
                    self.frontier_to_basis_dict[frontier_state] = {forced_basis}
                    continue
            if self._should_use_chaos_for_frontier(frontier_state):
                continue
            if self._active_apartness_checker is None:
                self.frontier_to_basis_dict[frontier_state] = set(
                    self.frontier_to_basis_dict.get(frontier_state, set())
                )
                continue
            self.frontier_to_basis_dict[frontier_state] = self.find_basis_candidates(frontier_state)
        self.frontier = self.frontier_to_basis_dict
        return self.frontier_to_basis_dict

    def find_basis_candidates(self, frontier_node):
        apartness_checker = self._active_apartness_checker
        diff_checker = self._active_diff_checker
        if apartness_checker is None:
            raise RuntimeError('apartness_checker is not configured on the observation tree.')
        candidates = set()
        for basis_node in self._ordered_basis_nodes_for_frontier(frontier_node):
            apart = apartness_checker.states_are_apart(
                frontier_node,
                basis_node,
                self,
                diff_checker=diff_checker,
                n_c=self._active_apartness_n_c,
                ensure_complete=self.ensure_complete,
                completion_max_attempts=self._active_completion_max_attempts,
                store_failed_prefixes=self._active_store_failed_prefixes,
                completion_rollout_len=self._active_completion_rollout_len,
            )
            if not apart:
                candidates.add(basis_node)
        return candidates

    def check_frontier_consistency(self):
        """
        Keep frontier semantics aligned with deterministic LSharp:
        frontier contains exactly the direct non-basis successors of basis nodes.
        """
        self.update_frontier_to_basis_dict()
        self.frontier = self.frontier_to_basis_dict
        return self.frontier_to_basis_dict

    def _frontier_promotion_key(self, node):
        return (
            -len(self.get_access_sequence(node)),
            node.frequency,
            -node.id,
        )

    def _choose_frontier_representative(self, frontier_state, candidates):
        candidates = set(candidates or ())
        if not candidates:
            return None
        ordered_basis = self._ordered_basis_nodes_for_frontier(frontier_state)
        for basis_node in ordered_basis:
            if basis_node in candidates:
                return basis_node
        return sorted(
            candidates,
            key=lambda node: (
                len(self.get_io_trace(node)),
                -getattr(node, 'frequency', 0),
                node.id,
            )
        )[0]

    def promote_frontier_state(self, frontier_state=None):
        self.update_frontier()

        if frontier_state is None:
            promotable = [
                node for node, candidates in self.frontier_to_basis_dict.items()
                if len(candidates) == 0 and not self._should_use_chaos_for_frontier(node)
            ]
            if not promotable:
                return None
            frontier_state = max(
                promotable,
                key=self._frontier_promotion_key,
            )
        else:
            if frontier_state not in self.frontier_to_basis_dict:
                return None
            self.update_basis_candidates(frontier_state)
            if self._should_use_chaos_for_frontier(frontier_state):
                return None
            if len(self.frontier_to_basis_dict.get(frontier_state, set())) != 0:
                return None

        return self._promote_frontier_state_without_refresh(frontier_state)

    def _promote_frontier_state_without_refresh(self, frontier_state):
        if frontier_state not in self.basis:
            old_basis = list(self.basis)
            self.basis.append(frontier_state)
            if self._active_apartness_checker is not None:
                for basis_state in old_basis:
                    self.get_or_compute_witness(basis_state, frontier_state)

        self.frontier_to_basis_dict.pop(frontier_state, None)
        self.frontier = self.frontier_to_basis_dict
        self.debug_stats['frontier_promotions'] += 1
        self._log(
            'frontier_promoted',
            node_id=frontier_state.id,
            node_trace=self.get_io_trace(frontier_state),
            basis_size=len(self.basis),
            frontier_size=len(self.frontier_to_basis_dict),
        )
        return frontier_state

    def promote_frontier_states(self, max_promotions=None):
        promoted = []
        limit = None if max_promotions is None else max(0, int(max_promotions))

        while limit is None or len(promoted) < limit:
            self.update_frontier_to_basis_dict()
            promotable = [
                node for node, candidates in self.frontier_to_basis_dict.items()
                if len(candidates) == 0 and not self._should_use_chaos_for_frontier(node)
            ]
            if not promotable:
                break

            frontier_state = max(
                promotable,
                key=self._frontier_promotion_key,
            )
            promoted.append(self._promote_frontier_state_without_refresh(frontier_state))

        return promoted

    def update_frontier_and_basis(self):
        self.update_frontier_to_basis_dict()
        self.promote_frontier_states()
        self.check_frontier_consistency()
        self.update_frontier_to_basis_dict()

    def make_basis_complete(self):
        for basis_state in list(self.basis):
            for inp in self.alphabet:
                current = basis_state.get_frequency_sum(inp)
                missing = max(0, int(self._active_n_c or 0) - current)
                if missing > 0:
                    self.debug_stats['ensure_complete_calls'] += 1
                    self.sample_node_input(
                        basis_state,
                        inp,
                        missing,
                        reason='complete_cell',
                        mode='completion',
                        max_attempts=self._active_completion_max_attempts,
                        completion_rollout_len=self._active_completion_rollout_len,
                    )
        self.check_frontier_consistency()

    def get_or_compute_witness(self, state_one, state_two):
        if state_one.id < state_two.id:
            pair = (state_one.id, state_two.id)
        else:
            pair = (state_two.id, state_one.id)

        if pair in self.witness_cache:
            return self.witness_cache.get(pair)

        apartness_checker = self._active_apartness_checker
        if apartness_checker is None:
            raise RuntimeError('apartness_checker is not configured on the observation tree.')

        witness = apartness_checker.compute_witness(
            state_one,
            state_two,
            self,
            diff_checker=self._active_diff_checker,
            n_c=self._active_apartness_n_c,
            ensure_complete=self.ensure_complete,
            completion_max_attempts=self._active_completion_max_attempts,
            store_failed_prefixes=self._active_store_failed_prefixes,
            completion_rollout_len=self._active_completion_rollout_len,
        )
        self.witness_cache[pair] = witness
        return witness

    def _register_frontier_candidate_refinements(self, frontier_state, basis_candidates):
        basis_candidates = list(basis_candidates or [])
        result = {
            'registered': False,
            'registered_targets': 0,
            'witness_count': 0,
            'missing_witness_count': 0,
            'witnesses': [],
        }
        if frontier_state is None or len(basis_candidates) < 2:
            return result

        anchor = basis_candidates[0]
        max_pairs = min(2, len(basis_candidates) - 1)
        required_hits = self._frontier_identification_samples(frontier_state)
        max_attempts = getattr(self, '_active_identification_budget', None)

        for candidate in basis_candidates[1: 1 + max_pairs]:
            witness = self.get_or_compute_witness(anchor, candidate)
            if not witness:
                result['missing_witness_count'] += 1
                self._log(
                    'identify_frontier_missing_basis_witness',
                    frontier_node_id=getattr(frontier_state, 'id', None),
                    anchor_id=getattr(anchor, 'id', None),
                    candidate_id=getattr(candidate, 'id', None),
                )
                continue

            witness = tuple(witness)
            result['witness_count'] += 1
            result['witnesses'].append(witness)

            changed = self._register_refinement_target(
                self.get_access_sequence(frontier_state) + witness,
                reason='identify_frontier',
                node=frontier_state,
                inp=witness[0] if witness else None,
                witness=witness,
                required_hits=required_hits,
                max_attempts=max_attempts,
            )
            if changed:
                result['registered'] = True
                result['registered_targets'] += 1

        self.debug_stats['identify_frontier_disambiguation_targets'] += result['registered_targets']
        self._log(
            'identify_frontier_refinement_targets',
            frontier_node_id=getattr(frontier_state, 'id', None),
            candidate_ids=[getattr(node, 'id', None) for node in basis_candidates],
            witness_count=result['witness_count'],
            missing_witness_count=result['missing_witness_count'],
            registered_targets=result['registered_targets'],
            required_hits=required_hits,
            max_attempts=max_attempts,
            reach_probability=self.estimate_reach_probability(frontier_state),
        )
        return result

    def _frontier_identification_samples(self, frontier_state):
        return max(1, int(getattr(self, '_active_resample_hits', 1) or 1))

    def identify_frontier(self, frontier_state):
        if frontier_state not in self.frontier_to_basis_dict:
            return
        if self._should_use_chaos_for_frontier(frontier_state):
            return

        self.debug_stats['identify_frontier_calls'] += 1
        record = self._frontier_identification_record(frontier_state)
        record['calls'] += 1
        self.update_basis_candidates(frontier_state)
        old_candidate_size = len(self.frontier_to_basis_dict.get(frontier_state, set()))
        record['last_candidates_before'] = [
            node.id for node in self.frontier_to_basis_dict.get(frontier_state, set())
        ]
        if old_candidate_size == 0:
            self.debug_stats['identify_frontier_zero_candidate_promotions'] += 1
            record['zero_candidate_promotions'] += 1
            record['last_reason'] = 'zero_candidates_promoted'
            self._promote_frontier_state_without_refresh(frontier_state)
            return

        if old_candidate_size < 2:
            self.debug_stats['identify_frontier_single_candidate_returns'] += 1
            record['single_candidate_returns'] += 1
            record['last_reason'] = 'single_candidate'
            return

        basis_candidates = sorted(
            self.frontier_to_basis_dict[frontier_state],
            key=lambda node: (
                -node.frequency,
                len(self.get_io_trace(node)),
                node.id,
            )
        )
        refinement_result = self._register_frontier_candidate_refinements(
            frontier_state,
            basis_candidates,
        )
        if refinement_result.get('witness_count', 0) <= 0:
            escalated = self._escalate_identification_settings(reason='missing_basis_witness')
            if escalated:
                self.update_basis_candidates(frontier_state)
                basis_candidates = sorted(
                    self.frontier_to_basis_dict.get(frontier_state, set()),
                    key=lambda node: (
                        -node.frequency,
                        len(self.get_io_trace(node)),
                        node.id,
                    )
                )
                refinement_result = self._register_frontier_candidate_refinements(
                    frontier_state,
                    basis_candidates,
                )
        if refinement_result.get('witness_count', 0) <= 0:
            self.debug_stats['identify_frontier_missing_witness'] += 1
            record['missing_witness'] += 1
            record['last_witness'] = None
            record['last_witness_len'] = 0
            record['last_reason'] = 'missing_basis_witness'
            return

        self.debug_stats['identify_frontier_witness_found'] += 1
        record['witness_found'] += 1
        witnesses = refinement_result.get('witnesses') or []
        record['last_witness'] = list(witnesses[0]) if witnesses else None
        record['last_witness_len'] = len(witnesses[0]) if witnesses else None
        record['last_reason'] = 'refinement_registered'

        self._log(
            'identify_frontier_start',
            frontier_node_id=frontier_state.id,
            frontier_trace=self.get_io_trace(frontier_state),
            candidate_ids=[node.id for node in basis_candidates],
            witness=tuple(witnesses[0]) if witnesses else None,
            witness_count=refinement_result.get('witness_count', 0),
            old_candidate_size=old_candidate_size,
        )

        registered = bool(refinement_result.get('registered'))
        sample_stats = {
            'registered_targets': refinement_result.get('registered_targets', 0),
            'witness_count': refinement_result.get('witness_count', 0),
            'missing_witness_count': refinement_result.get('missing_witness_count', 0),
            'required_hits': self._frontier_identification_samples(frontier_state),
            'max_attempts': getattr(self, '_active_identification_budget', None),
        }

        record['sample_attempts'] += sample_stats.get('attempts', 0)
        record['replay_matches'] += sample_stats.get('replay_matches', 0)
        record['replay_failures'] += sample_stats.get('replay_failures', 0)
        record['inserted_samples'] += sample_stats.get('inserted_samples', 0)
        record['last_sample_stats'] = dict(sample_stats)
        final_size = len(self.frontier_to_basis_dict.get(frontier_state, set()))
        record['last_candidates_after'] = [
            node.id for node in self.frontier_to_basis_dict.get(frontier_state, set())
        ]
        if final_size >= old_candidate_size and not registered:
            self.debug_stats['identify_frontier_no_progress'] += 1
            record['no_progress'] += 1
            record['last_reason'] = 'no_candidate_reduction'
            if self._escalate_identification_settings(reason='frontier_identification_no_progress'):
                self.debug_stats['frontier_identification_escalations'] += 1

        self._log(
            'identify_frontier_done',
            frontier_node_id=frontier_state.id,
            candidate_ids=[node.id for node in self.frontier_to_basis_dict.get(frontier_state, set())],
            candidate_size=len(self.frontier_to_basis_dict.get(frontier_state, set())),
        )

    def make_frontiers_identified(self):
        for frontier_state in list(self.frontier_to_basis_dict.keys()):
            self.identify_frontier(frontier_state)

    def is_observation_tree_adequate(self):
        if self._pending_refinement_targets:
            return False
        self.check_frontier_consistency()
        if self._pending_refinement_targets:
            return False
        for basis_state in self.basis:
            for inp in self.alphabet:
                if basis_state.get_frequency_sum(inp) <= 0:
                    return False
        for frontier_state, basis_list in self.frontier_to_basis_dict.items():
            if len(basis_list) != 1 and not self._should_use_chaos_for_frontier(frontier_state):
                return False
        return True

    def _frontier_signature(self):
        return {
            node.id: tuple(sorted(candidate.id for candidate in candidates))
            for node, candidates in self.frontier_to_basis_dict.items()
        }

    def _unresolved_frontier_entries(self):
        return [
            (node, candidates)
            for node, candidates in self.frontier_to_basis_dict.items()
            if len(candidates) != 1 and not self._should_use_chaos_for_frontier(node)
        ]

    def _force_merge_ambiguous_frontiers(self, unresolved_entries):
        changed = False
        for node, candidates in unresolved_entries or []:
            candidates = set(candidates or ())
            if not candidates:
                continue
            representative = self._choose_frontier_representative(node, candidates)
            if representative is None:
                continue
            self._manual_frontier_merges[getattr(node, 'id', None)] = getattr(representative, 'id', None)
            self.frontier_to_basis_dict[node] = {representative}
            self.debug_stats['frontier_forced_merges'] += 1
            changed = True
            self._log(
                'frontier_forced_merge',
                frontier_node_id=getattr(node, 'id', None),
                representative_id=getattr(representative, 'id', None),
                candidate_ids=[getattr(candidate, 'id', None) for candidate in candidates],
                reason='stagnated_ambiguous_frontier',
            )
        self.frontier = self.frontier_to_basis_dict
        return changed

    def _adapt_chaos_threshold_for_unresolved(self, unresolved_entries):
        if not getattr(self, '_active_chaos_adaptive', False):
            return False
        if not unresolved_entries:
            return False
        current = float(getattr(self, '_active_chaos_reach_threshold', 0.0) or 0.0)
        maximum = float(getattr(self, '_active_chaos_max_reach_threshold', current) or current)
        if current <= 0.0 or current >= maximum:
            return False

        reach_probabilities = [self.estimate_reach_probability(node) for node, _ in unresolved_entries]
        above_current = [probability for probability in reach_probabilities if probability >= current]
        if not above_current:
            return False

        next_reach = min(above_current)
        grown = current * float(getattr(self, '_active_chaos_growth_factor', 2.0) or 2.0)
        proposed = max(grown, next_reach * (1.0 + 1e-9))
        new_threshold = min(maximum, proposed)
        if new_threshold <= current:
            return False

        self._active_chaos_reach_threshold = new_threshold
        self.debug_stats['chaos_threshold_adaptations'] += 1
        self.debug_stats['chaos_current_reach_threshold'] = new_threshold
        self.debug_stats['chaos_last_adapted_unresolved_count'] = len(unresolved_entries)
        self.debug_stats['chaos_last_min_unresolved_reach_probability'] = min(reach_probabilities)
        return True

    def make_observation_tree_adequate(self):
        self.update_frontier_and_basis()
        stable_rounds = 0
        max_local_rounds = max(10, 2 * (len(self.basis) + len(self.frontier_to_basis_dict) + 1))
        stagnation_rounds = 0

        while not self.is_observation_tree_adequate():
            before_basis_size = len(self.basis)
            before_signature = self._frontier_signature()

            self.make_basis_complete()
            self.make_frontiers_identified()
            refinement_targets = self._pop_refinement_targets()
            refinement_result = self._run_refinement_targets(refinement_targets)
            promoted = self.promote_frontier_states()
            self.check_frontier_consistency()
            self.update_frontier_to_basis_dict()

            stable_rounds += 1
            after_signature = self._frontier_signature()
            progress = (
                bool(promoted)
                or (len(self.basis) != before_basis_size)
                or (after_signature != before_signature)
                or refinement_result.get('targets_improved', 0) > 0
            )

            if progress:
                stagnation_rounds = 0
            else:
                stagnation_rounds += 1
                escalated = self._escalate_identification_settings(reason='adequacy_stagnation')
                if escalated:
                    stable_rounds = 0
                    continue

            if stable_rounds > max_local_rounds or stagnation_rounds > 3:
                unresolved_entries = self._unresolved_frontier_entries()
                should_adapt = (
                    stable_rounds > max_local_rounds
                    or stagnation_rounds >= int(getattr(self, '_active_chaos_stagnation_rounds', 2) or 2)
                )
                if should_adapt and self._adapt_chaos_threshold_for_unresolved(unresolved_entries):
                    stable_rounds = 0
                    stagnation_rounds = 0
                    self.update_frontier_to_basis_dict()
                    continue
                unresolved = {
                    node.id: [candidate.id for candidate in candidates]
                    for node, candidates in unresolved_entries
                }
                if not unresolved:
                    return
                if all(candidates for _, candidates in unresolved_entries):
                    if self._force_merge_ambiguous_frontiers(unresolved_entries):
                        stable_rounds = 0
                        stagnation_rounds = 0
                        continue
                raise ObservationTreeAdequacyError(
                    unresolved,
                    debug_snapshot=self.snapshot_adequacy_debug(unresolved_entries),
                )

    # ------------------------------------------------------------------
    # Hypothesis construction
    # ------------------------------------------------------------------
    def estimate_reach_probability(self, node):
        if node is self.root:
            return 1.0

        probability = 1.0
        current = node
        while current is not None and current is not self.root:
            parent = current.parent
            if parent is None:
                return 0.0
            inp = current.input_to_parent
            out = current.output
            total = parent.get_frequency_sum(inp)
            if total <= 0:
                return 0.0
            edge_count = parent.get_output_frequency(inp, out)
            if edge_count <= 0:
                return 0.0
            probability *= edge_count / total
            current = parent
        return probability

    def _should_use_chaos_for_frontier(self, frontier_state):
        if getattr(frontier_state, 'id', None) in getattr(self, '_manual_chaos_frontier_ids', set()):
            return True
        threshold = float(getattr(self, '_active_chaos_reach_threshold', 0.0) or 0.0)
        if threshold <= 0.0:
            return False
        if frontier_state not in self.frontier_to_basis_dict:
            return False
        candidates = self.frontier_to_basis_dict.get(frontier_state, set())
        if len(candidates) == 1:
            return False
        return self.estimate_reach_probability(frontier_state) < threshold

    def mark_frontier_as_chaos(self, frontier_state):
        if frontier_state not in self.frontier_to_basis_dict:
            raise KeyError(f'Node {frontier_state.id} is not a current frontier node.')
        self._manual_chaos_frontier_ids.add(frontier_state.id)

    def snapshot_frontier_resolutions(self):
        return {
            'frontier_to_basis_dict': {
                node: set(candidates)
                for node, candidates in self.frontier_to_basis_dict.items()
            },
            'manual_chaos_frontier_ids': set(getattr(self, '_manual_chaos_frontier_ids', set())),
            'manual_frontier_merges': dict(getattr(self, '_manual_frontier_merges', {})),
        }

    def restore_frontier_resolutions(self, snapshot):
        snapshot = snapshot or {}
        self.frontier_to_basis_dict = {
            node: set(candidates)
            for node, candidates in snapshot.get('frontier_to_basis_dict', {}).items()
        }
        self.frontier = self.frontier_to_basis_dict
        self._manual_chaos_frontier_ids = set(snapshot.get('manual_chaos_frontier_ids', set()))
        self._manual_frontier_merges = dict(snapshot.get('manual_frontier_merges', {}))

    def apply_frontier_resolutions(self, resolutions):
        by_id = {node.id: node for node in self.frontier_to_basis_dict}
        basis_by_id = {node.id: node for node in self.basis}
        applied = []
        for resolution in resolutions or []:
            frontier_id = int(resolution.get('frontier_node_id'))
            action = str(resolution.get('action') or '').lower()
            frontier_node = by_id.get(frontier_id)
            if frontier_node is None:
                raise ValueError(f'Unknown frontier_node_id in resolution: {frontier_id}.')
            if action == 'merge':
                basis_id = int(resolution.get('basis_node_id'))
                basis_node = basis_by_id.get(basis_id)
                if basis_node is None:
                    raise ValueError(f'Unknown basis_node_id in resolution: {basis_id}.')
                self.frontier_to_basis_dict[frontier_node] = {basis_node}
                self._manual_frontier_merges[frontier_id] = basis_id
                self._manual_chaos_frontier_ids.discard(frontier_id)
            elif action == 'chaos':
                self._manual_frontier_merges.pop(frontier_id, None)
                self.mark_frontier_as_chaos(frontier_node)
            else:
                raise ValueError(f"Unsupported frontier resolution action: {action!r}.")
            applied.append({'frontier_node_id': frontier_id, 'action': action})
        self.frontier = self.frontier_to_basis_dict
        return applied

    def find_observation_counterexample(self, hypothesis, *, stop_at_chaos=True, validated_frontier_ids=None):
        validated_frontier_ids = set(validated_frontier_ids or [])
        for node, _, _ in self.iter_nodes_with_roles(max_depth=None, max_nodes=None):
            io_trace = self.get_io_trace(node)
            if not io_trace:
                continue
            current_tree_node = self.root
            current_hyp_state = hypothesis.initial_state
            prefix = []
            under_validated_resolution = False
            for step, (inp, out) in enumerate(zip(io_trace[0::2], io_trace[1::2]), start=1):
                hyp_state_id = getattr(
                    current_hyp_state,
                    'state_id',
                    getattr(current_hyp_state, 'name', None),
                )
                if stop_at_chaos and str(hyp_state_id) == 'chaos':
                    break
                successor = current_tree_node.get_successor(inp, out) if current_tree_node is not None else None
                prefix.extend([inp, out])
                if stop_at_chaos and successor is not None and self._should_use_chaos_for_frontier(successor):
                    break
                transitions = getattr(current_hyp_state, 'transitions', {}).get(inp, [])
                matched = None
                for target_state, output, probability in transitions:
                    if output == out and probability > 0:
                        matched = target_state
                        break
                if matched is None:
                    if under_validated_resolution:
                        return {
                            'io_trace': list(prefix),
                            'failure_step': step,
                            'reason': 'observed_output_not_supported_after_llm_frontier_merge',
                            'observed_node_id': successor.id if successor is not None else node.id,
                            'input': inp,
                            'output': out,
                            'candidate_state': getattr(current_hyp_state, 'state_id', getattr(current_hyp_state, 'name', str(current_hyp_state))),
                        }
                    break
                current_hyp_state = matched
                if successor is not None and successor.id in validated_frontier_ids:
                    under_validated_resolution = True
                current_tree_node = successor
        return None

    def _make_chaos_state(self, state_cls):
        chaos_state = state_cls('chaos')
        chaos_state.transitions = getattr(chaos_state, 'transitions', {})
        for inp in self.alphabet:
            chaos_state.transitions[inp] = [(chaos_state, 'chaos', 1.0)]
        return chaos_state

    def choose_representative(self, node):
        if node in self.basis:
            return node

        if node not in self.frontier_to_basis_dict:
            raise KeyError(
                f'Node {node.id} is neither in basis nor in frontier; cannot choose representative.'
            )

        candidates = self.frontier_to_basis_dict[node]
        if not candidates:
            raise ValueError(
                f'Frontier node {node.id} has no compatible basis candidate.'
            )
        if len(candidates) > 1:
            raise RuntimeError(
                f'Frontier node {node.id} is not identified yet: multiple basis candidates {[c.id for c in candidates]}.'
            )
        return next(iter(candidates))

    def construct_hypothesis_states(self, state_cls=None):
        self.states_dict = {}
        self._chaos_hyp_state = None
        if state_cls is None:
            try:
                from aalpy.automata import StochasticMealyState
            except ImportError:
                from aalpy.automata.StochasticMealyMachine import StochasticMealyState
            state_cls = StochasticMealyState

        for state_counter, basis_state in enumerate(self.basis):
            state_id = f's{state_counter}'
            state = state_cls(state_id)
            state.transitions = getattr(state, 'transitions', {})
            self.states_dict[basis_state] = state

        if any(self._should_use_chaos_for_frontier(node) for node in self.frontier_to_basis_dict):
            self._chaos_hyp_state = self._make_chaos_state(state_cls)

    def construct_hypothesis_transitions(self):
        for basis_state in self.basis:
            hyp_state = self.states_dict[basis_state]
            for input_val in self.alphabet:
                total = basis_state.get_frequency_sum(input_val)
                if total <= 0:
                    raise RuntimeError(
                        f'Cannot construct hypothesis: basis state {basis_state.id} has no samples for input {input_val!r}.'
                    )
                transitions = []
                out_map = basis_state.successors.get(input_val, {})
                if not out_map:
                    raise RuntimeError(
                        f'Cannot construct hypothesis: basis state {basis_state.id} has no observed successors for input {input_val!r}.'
                    )
                for out, successor in out_map.items():
                    edge_count = basis_state.get_output_frequency(input_val, out)
                    representative = successor
                    destination = None
                    if successor in self.frontier_to_basis_dict:
                        candidates = self.frontier_to_basis_dict[successor]
                        if len(candidates) > 1:
                            if self._should_use_chaos_for_frontier(successor):
                                destination = self._chaos_hyp_state
                            else:
                                raise RuntimeError('Multiple basis candidates for a single frontier state.')
                        if len(candidates) == 0:
                            if self._should_use_chaos_for_frontier(successor):
                                destination = self._chaos_hyp_state
                            else:
                                raise RuntimeError('Frontier successor has no basis candidate.')
                        if destination is None:
                            representative = next(iter(candidates))
                    if destination is None:
                        if representative not in self.states_dict:
                            raise RuntimeError(
                                'Successor is not in the basisToStateMap. '
                                f'basis_state={basis_state.id}, input={input_val!r}, output={out!r}, '
                                f'successor={getattr(successor, "id", None)}, '
                                f'representative={getattr(representative, "id", None)}, '
                                f'in_frontier={successor in self.frontier_to_basis_dict}, '
                                f'candidate_ids={[candidate.id for candidate in self.frontier_to_basis_dict.get(successor, set())]}, '
                                f'basis_ids={[node.id for node in self.basis]}'
                            )
                        destination = self.states_dict[representative]
                    if destination is None:
                        raise RuntimeError('Chaos transition requested but chaos state is not configured.')
                    probability = edge_count / total if edge_count else 0.0
                    transitions.append((destination, out, probability))
                hyp_state.transitions[input_val] = transitions

    def construct_hypothesis(self, state_cls=None, machine_cls=None):
        self.update_frontier_to_basis_dict()
        self.construct_hypothesis_states(state_cls=state_cls)
        self.construct_hypothesis_transitions()

        if machine_cls is None:
            try:
                from aalpy.automata import StochasticMealyMachine
            except ImportError:
                from aalpy.automata.StochasticMealyMachine import StochasticMealyMachine
            machine_cls = StochasticMealyMachine

        states = list(self.states_dict.values())
        if self._chaos_hyp_state is not None:
            states.append(self._chaos_hyp_state)
        hypothesis = machine_cls(self.states_dict[self.root], states)
        if hasattr(hypothesis, 'compute_prefixes'):
            hypothesis.compute_prefixes()
        return hypothesis

    def build_hypothesis(self, state_cls=None, machine_cls=None):
        self.make_observation_tree_adequate()
        if not self.is_observation_tree_adequate():
            unresolved = {
                node.id: [candidate.id for candidate in candidates]
                for node, candidates in self.frontier_to_basis_dict.items()
                if len(candidates) != 1 and not self._should_use_chaos_for_frontier(node)
            }
            if not unresolved:
                return self.construct_hypothesis(state_cls=state_cls, machine_cls=machine_cls)
            raise RuntimeError(
                'Observation tree is still not adequate before hypothesis construction. '
                f'Unresolved frontier candidates: {unresolved}'
            )
        return self.construct_hypothesis(state_cls=state_cls, machine_cls=machine_cls)

    # ------------------------------------------------------------------
    # Counterexample processing
    # ------------------------------------------------------------------
    def reset_basis_and_frontier(self, *, preserve_witness_cache=False):
        """
        Reset partition metadata after a counterexample has been inserted.

        The observation tree itself is kept intact; only the current Basis /
        Frontier partition is discarded so that the next adequacy pass rebuilds
        it from the updated tree starting from the root.
        """
        self.basis = [self.root]
        self.frontier_to_basis_dict = {}
        self.frontier = self.frontier_to_basis_dict
        self._manual_chaos_frontier_ids.clear()
        self.states_dict = {}
        if not preserve_witness_cache:
            self._invalidate_witness_cache(reason='counterexample_basis_reset')
        self._log(
            'basis_frontier_reset',
            basis_size=len(self.basis),
            frontier_size=len(self.frontier_to_basis_dict),
            root_id=self.root.id,
            preserve_witness_cache=preserve_witness_cache,
        )

    def recompute_budget_finalization_partition(self):
        """
        Rebuild the current Basis/Frontier metadata without sampling.

        After the global sampling budget is exhausted, the learner cannot gather
        more evidence to justify probabilistic splits. This reset keeps the
        existing observation tree and hard apartness witnesses, but removes
        earlier partition choices so an LLM modeler can merge states subject to
        the already proven illegal-merge constraints.
        """
        self.reset_basis_and_frontier(preserve_witness_cache=True)
        self.update_frontier()
        apart_pairs = set(getattr(self, 'witness_cache', {}) or {})
        for frontier_state in list(self.frontier_to_basis_dict.keys()):
            candidates = set()
            for basis_state in self.basis:
                pair = (
                    (basis_state.id, frontier_state.id)
                    if basis_state.id < frontier_state.id
                    else (frontier_state.id, basis_state.id)
                )
                if pair not in apart_pairs:
                    candidates.add(basis_state)
            self.frontier_to_basis_dict[frontier_state] = candidates
        self.frontier = self.frontier_to_basis_dict
        self._log(
            'budget_finalization_partition_recomputed',
            basis_size=len(self.basis),
            frontier_size=len(self.frontier_to_basis_dict),
            frontier_candidates={
                node.id: [candidate.id for candidate in candidates]
                for node, candidates in self.frontier_to_basis_dict.items()
            },
        )
        return self.frontier_to_basis_dict

    def process_counter_example(self, hypothesis, io_trace, *, already_inserted=False, cex_processing='binary', cex_info=None):
        self.debug_stats['process_counterexample_calls'] += 1
        if not already_inserted:
            self.insert_trace(io_trace)
        processed_trace = tuple(io_trace)
        cex_info = dict(cex_info or {})
        if cex_processing == 'binary' and cex_info.get('diff_kind') == 'distribution_mismatch':
            processed_trace, changed = self._process_distribution_mismatch_counterexample(
                hypothesis,
                processed_trace,
                cex_info,
            )
        elif cex_processing == 'binary':
            processed_trace = self._process_binary_search(hypothesis, processed_trace)
        self.reset_basis_and_frontier(preserve_witness_cache=already_inserted)
        return processed_trace

    def _basis_node_from_hypothesis_state(self, hyp_state):
        for node, state in self.states_dict.items():
            if state is hyp_state:
                return node
        return None

    def _get_hypothesis_successor_from_io_trace(self, hypothesis, io_trace):
        current_state = hypothesis.initial_state
        for inp, out in zip(io_trace[0::2], io_trace[1::2]):
            transitions = getattr(current_state, 'transitions', {}).get(inp, [])
            matched = None
            for target_state, output, probability in transitions:
                if output == out and probability > 0:
                    matched = target_state
                    break
            if matched is None:
                return None
            current_state = matched
        return current_state

    def _get_counterexample_prefix_index(self, hypothesis, io_trace):
        """
        Return the number of input/output steps in the shortest prefix that is
        not supported by the current hypothesis as an exact I/O path.

        If the entire exact trace is supported, return the full trace length so
        that subsequent CE processing can still refine a probabilistic mismatch.
        """
        current_state = hypothesis.initial_state
        matched_steps = 0
        for matched_steps, (inp, out) in enumerate(zip(io_trace[0::2], io_trace[1::2]), start=1):
            transitions = getattr(current_state, 'transitions', {}).get(inp, [])
            matched = None
            for target_state, output, probability in transitions:
                if output == out and probability > 0:
                    matched = target_state
                    break
            if matched is None:
                return matched_steps
            current_state = matched
        return len(io_trace) // 2

    def _promote_counterexample_prefix_if_needed(self, prefix_io, hypothesis=None):
        tree_node = self.get_successor(prefix_io)
        if tree_node is None:
            return None

        self.update_frontier_and_basis()
        if tree_node in self.basis:
            return tree_node

        # Only frontier states may be promoted to basis. Deeper remain nodes must
        # stay remain until one of their ancestors enters the basis and they become
        # direct successors of a basis node.
        if tree_node not in self.frontier_to_basis_dict:
            return tree_node

        hyp_state = None
        if hypothesis is not None:
            hyp_state = self._get_hypothesis_successor_from_io_trace(hypothesis, prefix_io)

        candidates = self.frontier_to_basis_dict.get(tree_node, set())
        should_promote = False
        if hyp_state is None:
            should_promote = True
        elif len(candidates) == 0:
            should_promote = True

        if should_promote:
            self.promote_frontier_state(tree_node)
            self.check_frontier_consistency()
            self.update_frontier_to_basis_dict()

        return tree_node

    def _register_counterexample_path_refinements(self, prefix_io, reason='counterexample_incomplete_cell'):
        prefix_io = tuple(prefix_io or ())
        if len(prefix_io) < 2:
            return False
        target = max(0, int(getattr(self, '_active_n_c', 0) or 0))
        default_hits = max(1, int(getattr(self, '_active_resample_hits', 1) or 1))
        targets = []
        changed_any = False
        nr_steps = len(prefix_io) // 2
        for step_index in range(nr_steps):
            parent_io = prefix_io[: 2 * step_index]
            inp = prefix_io[2 * step_index]
            parent_node = self.get_successor(parent_io) if parent_io else self.root
            if parent_node is None:
                continue
            current = parent_node.get_frequency_sum(inp)
            if target > 0 and current >= target:
                continue
            required_hits = max(default_hits, max(0, target - current))
            inputs = tuple(parent_io[0::2]) + (inp,)
            changed = self._register_refinement_target(
                inputs,
                reason=reason,
                node=parent_node,
                inp=inp,
                witness=prefix_io[: 2 * (step_index + 1)],
                required_hits=required_hits,
                max_attempts=getattr(self, '_active_identification_budget', None),
            )
            changed_any = changed_any or changed
            target_entry = self._pending_refinement_targets.pop(inputs, None)
            if target_entry is not None:
                targets.append(target_entry)
        if targets:
            self._run_refinement_targets(targets)
            return True
        return changed_any

    def _suffix_output_distribution(self, start_node, inputs):
        inputs = tuple(inputs or ())
        if start_node is None or not inputs:
            return {}

        paths = [(start_node, tuple())]
        for inp in inputs:
            next_paths = []
            for node, outputs in paths:
                for out, successor in node.successors.get(inp, {}).items():
                    next_paths.append((successor, outputs + (out,)))
            if not next_paths:
                return {}
            paths = next_paths

        counts = {}
        for node, outputs in paths:
            counts[outputs] = counts.get(outputs, 0) + getattr(node, 'frequency', 0)
        return counts

    def _register_suffix_refinement(
        self,
        node,
        suffix_inputs,
        reason,
        required_hits=None,
        max_attempts=None,
        samples_per_target=None,
    ):
        suffix_inputs = tuple(suffix_inputs or ())
        if node is None or not suffix_inputs:
            return False
        return self._register_refinement_target(
            self.get_access_sequence(node) + suffix_inputs,
            reason=reason,
            node=node,
            inp=suffix_inputs[0],
            witness=suffix_inputs,
            required_hits=required_hits,
            max_attempts=max_attempts,
            samples_per_target=samples_per_target,
        )

    def _probabilistic_ce_alpha(self):
        alpha = getattr(self._active_diff_checker, 'alpha', 0.05) if self._active_diff_checker is not None else 0.05
        try:
            alpha = float(alpha)
        except (TypeError, ValueError):
            alpha = 0.05
        return min(max(alpha, 1e-12), 1.0 - 1e-12)

    def _two_sample_threshold(self, left_total, right_total, alpha=None):
        left_total = int(left_total or 0)
        right_total = int(right_total or 0)
        if left_total <= 0 or right_total <= 0:
            return float('inf')
        if alpha is None:
            alpha = self._probabilistic_ce_alpha()
        return (sqrt(1 / left_total) + sqrt(1 / right_total)) * sqrt(0.5 * log(2 / alpha))

    def _probabilistic_ce_requested_samples(self, diagnostics):
        diagnostics = diagnostics or {}
        default_samples = max(1, int(getattr(self, '_active_resample_hits', 1) or 1))
        tree_total = int(diagnostics.get('tree_total', 0) or 0)
        basis_total = int(diagnostics.get('basis_total', 0) or 0)
        threshold = int(diagnostics.get('threshold', 0) or 0)
        missing = max(
            threshold - tree_total,
            threshold - basis_total,
            0,
        )
        return max(default_samples, missing)

    def _compare_observed_suffix_distributions(self, tree_node, basis_node, suffix_inputs):
        suffix_inputs = tuple(suffix_inputs or ())
        if tree_node is None or basis_node is None or not suffix_inputs:
            return 'UNKNOWN', {}

        tree_counts = self._suffix_output_distribution(tree_node, suffix_inputs)
        basis_counts = self._suffix_output_distribution(basis_node, suffix_inputs)
        tree_total = sum(tree_counts.values())
        basis_total = sum(basis_counts.values())
        threshold = max(1, int(getattr(self, '_active_apartness_n_c', 0) or getattr(self, '_active_n_c', 0) or 1))
        diagnostics = {
            'tree_node_id': tree_node.id,
            'basis_node_id': basis_node.id,
            'suffix_inputs': suffix_inputs,
            'tree_total': tree_total,
            'basis_total': basis_total,
            'threshold': threshold,
        }

        if tree_total < threshold or basis_total < threshold:
            return 'UNKNOWN', diagnostics

        support = set(tree_counts.keys()).union(basis_counts.keys())
        max_gap = 0.0
        witness_output = None
        for output in support:
            tree_probability = tree_counts.get(output, 0) / tree_total if tree_total else 0.0
            basis_probability = basis_counts.get(output, 0) / basis_total if basis_total else 0.0
            gap = abs(tree_probability - basis_probability)
            if gap > max_gap:
                max_gap = gap
                witness_output = output
        two_sample_threshold = self._two_sample_threshold(tree_total, basis_total)
        diagnostics.update({
            'max_gap': max_gap,
            'witness_output': witness_output,
            'two_sample_threshold': two_sample_threshold,
            'insufficient_power': max_gap > 0.0 and max_gap < two_sample_threshold,
        })

        diff_checker = self._active_diff_checker
        if diff_checker is not None:
            is_diff = diff_checker.are_cells_different(tree_counts, basis_counts)
        else:
            is_diff = tree_counts != basis_counts
        diagnostics['tree_counts'] = tree_counts
        diagnostics['basis_counts'] = basis_counts
        return ('DIFF' if is_diff else 'NO_DIFF'), diagnostics

    def _probabilistic_ce_split_candidate(self, hypothesis, prefix_io, mismatch_input, split_steps, *, source):
        split_io = tuple(prefix_io[:2 * split_steps])
        suffix_inputs = tuple(prefix_io[2 * split_steps::2]) + (mismatch_input,)
        tree_node = self.get_successor(split_io)
        hyp_state = self._get_hypothesis_successor_from_io_trace(hypothesis, split_io)
        basis_node = self._basis_node_from_hypothesis_state(hyp_state) if hyp_state is not None else None
        status, diagnostics = self._compare_observed_suffix_distributions(
            tree_node,
            basis_node,
            suffix_inputs,
        )
        diagnostics = dict(diagnostics or {})
        diagnostics.update({
            'split_steps': split_steps,
            'source': source,
        })
        effective_status = status
        if status == 'NO_DIFF' and diagnostics.get('insufficient_power'):
            effective_status = 'UNKNOWN'
            diagnostics['unknown_reason'] = 'insufficient_power'
        return {
            'split_steps': split_steps,
            'split_io': split_io,
            'suffix_inputs': suffix_inputs,
            'tree_node': tree_node,
            'basis_node': basis_node,
            'status': status,
            'effective_status': effective_status,
            'diagnostics': diagnostics,
        }

    def _probabilistic_ce_candidate_rank(self, candidate):
        diagnostics = candidate.get('diagnostics') or {}
        effective_status = candidate.get('effective_status')
        tree_node = candidate.get('tree_node')
        basis_node = candidate.get('basis_node')
        status_rank = {
            'DIFF': 3,
            'UNKNOWN': 2,
            'NO_DIFF': 1,
        }.get(effective_status, 0)
        different_nodes = 1 if tree_node is not None and basis_node is not None and tree_node is not basis_node else 0
        max_gap = float(diagnostics.get('max_gap', 0.0) or 0.0)
        tree_total = int(diagnostics.get('tree_total', 0) or 0)
        basis_total = int(diagnostics.get('basis_total', 0) or 0)
        threshold = int(diagnostics.get('threshold', 0) or 0)
        sample_shortage = max(threshold - tree_total, threshold - basis_total, 0)
        if effective_status == 'DIFF':
            split_preference = -int(candidate.get('split_steps', 0) or 0)
        else:
            split_preference = int(candidate.get('split_steps', 0) or 0)
        return (
            status_rank,
            different_nodes,
            max_gap,
            sample_shortage,
            split_preference,
        )

    def _choose_probabilistic_ce_candidate(self, candidates, *, allow_no_diff=False):
        useful = []
        for candidate in candidates:
            status = candidate.get('effective_status')
            if status in ('DIFF', 'UNKNOWN'):
                useful.append(candidate)
            elif allow_no_diff and (candidate.get('diagnostics') or {}).get('max_gap', 0.0) > 0.0:
                useful.append(candidate)
        if not useful:
            return None
        return max(useful, key=self._probabilistic_ce_candidate_rank)

    def _register_probabilistic_ce_candidate(self, candidate, *, reason):
        diagnostics = candidate.get('diagnostics') or {}
        tree_node = candidate.get('tree_node')
        basis_node = candidate.get('basis_node')
        suffix_inputs = candidate.get('suffix_inputs') or ()
        requested_hits = self._probabilistic_ce_requested_samples(diagnostics)
        max_attempts = getattr(self, '_active_identification_budget', None)
        status = candidate.get('effective_status')

        if status == 'DIFF':
            self.debug_stats['probabilistic_counterexample_diff_cells'] += 1
        else:
            self.debug_stats['probabilistic_counterexample_unknown_cells'] += 1
            if diagnostics.get('unknown_reason') == 'insufficient_power':
                self.debug_stats['probabilistic_counterexample_insufficient_power'] += 1

        changed = False
        changed = self._register_suffix_refinement(
            tree_node,
            suffix_inputs,
            f'{reason}_tree',
            required_hits=requested_hits,
            max_attempts=max_attempts,
        ) or changed
        changed = self._register_suffix_refinement(
            basis_node,
            suffix_inputs,
            f'{reason}_basis',
            required_hits=requested_hits,
            max_attempts=max_attempts,
        ) or changed
        if changed:
            self.debug_stats['probabilistic_counterexample_refinements'] += 1
        self._log(
            'probabilistic_counterexample_cell_selected',
            reason=reason,
            split_steps=candidate.get('split_steps'),
            status=candidate.get('status'),
            effective_status=status,
            tree_node_id=getattr(tree_node, 'id', None),
            basis_node_id=getattr(basis_node, 'id', None),
            suffix_inputs=suffix_inputs,
            required_hits=requested_hits,
            max_attempts=max_attempts,
            diagnostics=diagnostics,
            registered_refinement=changed,
        )
        return changed

    def _process_distribution_mismatch_counterexample(self, hypothesis, io_trace, cex_info):
        self.debug_stats['probabilistic_counterexample_calls'] += 1
        self.debug_stats['probabilistic_counterexample_binary_searches'] += 1
        prefix_io = tuple(cex_info.get('prefix_io') or io_trace[:-2])
        mismatch_input = cex_info.get('input')
        processed_trace = prefix_io
        if mismatch_input is not None:
            processed_trace = prefix_io + (mismatch_input,)
            if len(io_trace) >= len(prefix_io) + 2:
                processed_trace = tuple(io_trace[:len(prefix_io) + 2])

        if mismatch_input is None:
            return tuple(processed_trace), False

        nr_prefix_steps = len(prefix_io) // 2
        lo = 0
        hi = nr_prefix_steps
        candidates = []
        visited_splits = set()

        while lo <= hi:
            mid = (lo + hi) // 2
            visited_splits.add(mid)
            candidate = self._probabilistic_ce_split_candidate(
                hypothesis,
                prefix_io,
                mismatch_input,
                mid,
                source='binary',
            )
            candidates.append(candidate)
            status = candidate['status']
            diagnostics = candidate['diagnostics']
            self._log(
                'probabilistic_counterexample_binary_step',
                split_steps=mid,
                status=status,
                effective_status=candidate['effective_status'],
                tree_node_id=getattr(candidate.get('tree_node'), 'id', None),
                basis_node_id=getattr(candidate.get('basis_node'), 'id', None),
                suffix_inputs=candidate.get('suffix_inputs'),
                diagnostics=diagnostics,
            )

            if candidate['effective_status'] == 'DIFF':
                hi = mid - 1
            else:
                lo = mid + 1

        best = self._choose_probabilistic_ce_candidate(candidates)
        if best is None or best.get('effective_status') != 'DIFF':
            self.debug_stats['probabilistic_counterexample_full_scans'] += 1
            for split_steps in range(nr_prefix_steps + 1):
                if split_steps in visited_splits:
                    continue
                candidate = self._probabilistic_ce_split_candidate(
                    hypothesis,
                    prefix_io,
                    mismatch_input,
                    split_steps,
                    source='full_scan',
                )
                candidates.append(candidate)
                self._log(
                    'probabilistic_counterexample_scan_step',
                    split_steps=split_steps,
                    status=candidate.get('status'),
                    effective_status=candidate.get('effective_status'),
                    tree_node_id=getattr(candidate.get('tree_node'), 'id', None),
                    basis_node_id=getattr(candidate.get('basis_node'), 'id', None),
                    suffix_inputs=candidate.get('suffix_inputs'),
                    diagnostics=candidate.get('diagnostics'),
                )
            best = self._choose_probabilistic_ce_candidate(candidates)

        if best is not None:
            reason = 'probabilistic_counterexample_diff'
            if best.get('effective_status') != 'DIFF':
                reason = 'probabilistic_counterexample_unknown'
            self._register_probabilistic_ce_candidate(best, reason=reason)
            return tuple(processed_trace), True

        fallback = self._choose_probabilistic_ce_candidate(candidates, allow_no_diff=True)
        if fallback is not None:
            self._register_probabilistic_ce_candidate(
                fallback,
                reason='probabilistic_counterexample_gap',
            )
            return tuple(processed_trace), True

        local_node = self.get_successor(prefix_io)
        if local_node is not None and mismatch_input is not None:
            self._register_refinement_target(
                self.get_access_sequence(local_node) + (mismatch_input,),
                reason='probabilistic_counterexample',
                node=local_node,
                inp=mismatch_input,
                required_hits=max(1, int(getattr(self, '_active_resample_hits', 1) or 1)),
                max_attempts=getattr(self, '_active_identification_budget', None),
            )
            self.debug_stats['probabilistic_counterexample_refinements'] += 1
            return tuple(processed_trace), True

        return tuple(processed_trace), False

    def _find_observed_traces_from_node(self, start_node, inputs, limit=16):
        paths = [(start_node, [])]
        for inp in tuple(inputs or ()):
            next_paths = []
            for node, suffix in paths:
                for out, successor in node.successors.get(inp, {}).items():
                    next_paths.append((successor, suffix + [inp, out]))
                    if len(next_paths) >= limit:
                        break
                if len(next_paths) >= limit:
                    break
            if not next_paths:
                return []
            paths = next_paths
        return [tuple(suffix) for _, suffix in paths]

    def _process_binary_search(self, hypothesis, io_trace):
        self.debug_stats['binary_search_calls'] += 1
        tree_node = self.get_successor(io_trace)
        if tree_node is None:
            return tuple(io_trace)

        self.update_frontier_and_basis()
        nr_steps = len(io_trace) // 2

        hyp_state = self._get_hypothesis_successor_from_io_trace(hypothesis, io_trace)
        if hyp_state is None:
            prefix_len = self._get_counterexample_prefix_index(hypothesis, io_trace)
            prefix_io = tuple(io_trace[: 2 * prefix_len])
            if prefix_len > 0:
                state_prefix_io = tuple(io_trace[: 2 * (prefix_len - 1)])
                state_node = self.root if not state_prefix_io else self.get_successor(state_prefix_io)
                hyp_prefix_state = self._get_hypothesis_successor_from_io_trace(hypothesis, state_prefix_io)
                hyp_node = self._basis_node_from_hypothesis_state(hyp_prefix_state) if hyp_prefix_state is not None else None
                if state_node is not None and hyp_node is not None and state_node is not hyp_node:
                    mismatch_input = io_trace[2 * (prefix_len - 1)]
                    pair_ids = (
                        (state_node.id, hyp_node.id)
                        if state_node.id < hyp_node.id
                        else (hyp_node.id, state_node.id)
                    )
                    for cache_key in list(self.witness_cache.keys()):
                        if cache_key == pair_ids or (
                            isinstance(cache_key, tuple)
                            and len(cache_key) > 0
                            and cache_key[0] == pair_ids
                        ):
                            self.witness_cache.pop(cache_key, None)
                    witness = (mismatch_input,)
                    self.witness_cache[pair_ids] = witness
                    max_depth = (
                        getattr(self._active_apartness_checker, 'max_depth', None)
                        if self._active_apartness_checker is not None
                        else None
                    )
                    self.witness_cache[(pair_ids, self._active_apartness_n_c, max_depth)] = witness
            self._promote_counterexample_prefix_if_needed(prefix_io, hypothesis)
            self._register_counterexample_path_refinements(prefix_io)
            return prefix_io

        hyp_node = self._basis_node_from_hypothesis_state(hyp_state)
        if hyp_node is None:
            self._promote_counterexample_prefix_if_needed(io_trace, hypothesis)
            return tuple(io_trace)

        if tree_node in self.basis:
            return tuple(io_trace)

        if tree_node in self.frontier_to_basis_dict:
            candidates = self.frontier_to_basis_dict[tree_node]
            if len(candidates) == 0:
                self.promote_frontier_state(tree_node)
                return tuple(io_trace)
            if len(candidates) == 1 and hyp_node in candidates:
                if nr_steps <= 1:
                    return tuple(io_trace)
                prefix_io = io_trace[: 2 * max(1, nr_steps // 2)]
                return self._process_binary_search(hypothesis, prefix_io)

        if nr_steps <= 1:
            witness = self.get_or_compute_witness(tree_node, hyp_node)
            if not witness:
                self._promote_counterexample_prefix_if_needed(io_trace, hypothesis)
            return tuple(io_trace)

        frontier_prefix_steps = 0
        current = self.root
        for step, (inp, out) in enumerate(zip(io_trace[0::2], io_trace[1::2]), start=1):
            if current in self.frontier_to_basis_dict:
                break
            successor = current.get_successor(inp, out)
            if successor is None:
                break
            current = successor
            frontier_prefix_steps = step

        h = max(1, (frontier_prefix_steps + nr_steps) // 2)
        if h >= nr_steps:
            h = max(1, nr_steps // 2)
        prefix_io = io_trace[: 2 * h]
        suffix_inputs = tuple(io_trace[2 * h::2])

        hyp_prefix_state = self._get_hypothesis_successor_from_io_trace(hypothesis, prefix_io)
        if hyp_prefix_state is None:
            self._promote_counterexample_prefix_if_needed(prefix_io, hypothesis)
            return tuple(prefix_io)

        hyp_prefix_node = self._basis_node_from_hypothesis_state(hyp_prefix_state)
        if hyp_prefix_node is None:
            self._promote_counterexample_prefix_if_needed(prefix_io, hypothesis)
            return tuple(prefix_io)

        witness = self.get_or_compute_witness(tree_node, hyp_node)
        if witness is None:
            self._promote_counterexample_prefix_if_needed(io_trace, hypothesis)
            return tuple(io_trace)

        tree_prefix_node = self.get_successor(prefix_io)
        if tree_prefix_node is None:
            return tuple(io_trace)

        witness_prefix = self.get_or_compute_witness(tree_prefix_node, hyp_prefix_node)
        if witness_prefix is not None:
            self._promote_counterexample_prefix_if_needed(prefix_io, hypothesis)
            return self._process_binary_search(hypothesis, prefix_io)

        suffix_to_reuse = suffix_inputs + tuple(witness)
        observed_suffixes = self._find_observed_traces_from_node(
            hyp_prefix_node,
            suffix_to_reuse,
            limit=16,
        )
        if observed_suffixes:
            hyp_prefix_io = self.get_io_trace(hyp_prefix_node)
            for observed_suffix in observed_suffixes:
                candidate_trace = hyp_prefix_io + observed_suffix
                if len(candidate_trace) >= len(io_trace):
                    continue
                return self._process_binary_search(hypothesis, candidate_trace)

        return tuple(io_trace)
