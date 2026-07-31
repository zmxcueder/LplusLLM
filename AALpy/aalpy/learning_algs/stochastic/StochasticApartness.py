from abc import ABC, abstractmethod
from collections import deque, defaultdict
from math import sqrt, log

try:
    from .DifferenceChecker import AdvancedHoeffdingChecker, DifferenceChecker
except ImportError:  # pragma: no cover
    from DifferenceChecker import AdvancedHoeffdingChecker, DifferenceChecker


class StochasticApartnessChecker(ABC):

    @abstractmethod
    def states_are_apart(self, s1, s2, ob_tree, diff_checker=None, **kwargs) -> bool:
        pass

    @abstractmethod
    def compute_witness(self, s1, s2, ob_tree, diff_checker=None, **kwargs):
        pass

    @abstractmethod
    def compute_witness_in_tree_and_hypothesis_states(self, ob_tree, ob_tree_state, hyp_state, diff_checker=None, **kwargs):
        pass

    def use_witness_cache(self):
        return True


class BFSApartnessChecker(StochasticApartnessChecker):
    """
    Statistical apartness checker for a stochastic observation tree.

    This keeps the deterministic LSharp *shape* (witness search over the
    observation tree), but the actual cell comparison uses a stochastic
    DifferenceChecker so the public API stays aligned with AALpy's
    StochasticLStar strategy choices.

    Important adaptation for the stochastic LSharp observation tree:
    apartness search is restricted to the subgraph induced by Basis ∪ Frontier.
    Deep remain nodes may exist in large numbers due to random EQ sampling, but
    they do not participate in the learner's identification logic. Letting BFS
    walk through them causes huge search blow-ups and repeated
    ensure_complete_skipped events without helping frontier identification.
    """

    def __init__(
        self,
        alpha=0.05,
        max_depth=None,
        cache=True,
        max_nodes_visited=None,
        max_queue_size=None,
        depth_n_c_caps=None,
        diff_checker=None,
    ):
        self.alpha = alpha
        self.max_depth = max_depth
        self.cache = cache
        self.max_nodes_visited = max_nodes_visited
        self.max_queue_size = max_queue_size
        self.depth_n_c_caps = depth_n_c_caps or {
            0: None,
            1: 4,
            2: 3,
            3: 2,
        }
        self.diff_checker = diff_checker or AdvancedHoeffdingChecker(alpha=alpha, use_diff=True)

    def set_diff_checker(self, diff_checker: DifferenceChecker):
        self.diff_checker = diff_checker
        return self

    def _resolve_diff_checker(self, diff_checker=None):
        return diff_checker or self.diff_checker

    @staticmethod
    def _is_search_relevant_tree_node(ob_tree, node):
        if node is None:
            return False
        checker = getattr(ob_tree, 'is_basis_or_frontier_node', None)
        if callable(checker):
            return bool(checker(node))
        basis = getattr(ob_tree, 'basis', None) or ()
        frontier = getattr(ob_tree, 'frontier_to_basis_dict', None) or {}
        return (node in basis) or (node in frontier)

    def states_are_apart(self, s1, s2, ob_tree, diff_checker=None, **kwargs) -> bool:
        return self.compute_witness(s1, s2, ob_tree, diff_checker=diff_checker, **kwargs) is not None

    def compute_witness(self, s1, s2, ob_tree, diff_checker=None, **kwargs):
        if s1 is None or s2 is None:
            return None

        n_c = kwargs.get('n_c', 0)
        max_depth = kwargs.get('max_depth', self.max_depth)
        ensure_complete = kwargs.get('ensure_complete', None)
        self._current_completion_max_attempts = kwargs.get('completion_max_attempts', None)
        self._current_store_failed_prefixes = kwargs.get('store_failed_prefixes', True)
        self._current_completion_rollout_len = kwargs.get('completion_rollout_len', 1)
        effective_diff_checker = self._resolve_diff_checker(diff_checker)

        cache_key = None
        if self.cache and self.use_witness_cache() and hasattr(ob_tree, 'witness_cache'):
            cache_key = self._get_cache_key(s1, s2, n_c, max_depth)
            if cache_key in ob_tree.witness_cache:
                if hasattr(ob_tree, '_log'):
                    ob_tree._log(
                        'apartness_cache_hit',
                        s1_id=s1.id,
                        s2_id=s2.id,
                        n_c=n_c,
                        max_depth=max_depth,
                        witness=ob_tree.witness_cache[cache_key],
                    )
                return ob_tree.witness_cache[cache_key]

        witness = self._bfs_find_witness(
            s1=s1,
            s2=s2,
            ob_tree=ob_tree,
            n_c=n_c,
            ensure_complete=ensure_complete,
            max_depth=max_depth,
            diff_checker=effective_diff_checker,
        )

        if cache_key is not None:
            ob_tree.witness_cache[cache_key] = witness

        if hasattr(ob_tree, '_log'):
            ob_tree._log(
                'apartness_result',
                s1_id=s1.id,
                s2_id=s2.id,
                n_c=n_c,
                max_depth=max_depth,
                witness=witness,
                diff_checker=type(effective_diff_checker).__name__,
            )
        return witness

    def compute_witness_in_tree_and_hypothesis_states(self, ob_tree, ob_tree_state, hyp_state, diff_checker=None, **kwargs):
        if ob_tree_state is None or hyp_state is None:
            return None

        n_c = kwargs.get('n_c', 0)
        max_depth = kwargs.get('max_depth', self.max_depth)
        ensure_complete = kwargs.get('ensure_complete', None)
        completion_max_attempts = kwargs.get('completion_max_attempts', None)
        store_failed_prefixes = kwargs.get('store_failed_prefixes', True)
        completion_rollout_len = kwargs.get('completion_rollout_len', 1)
        effective_diff_checker = self._resolve_diff_checker(diff_checker)

        queue = deque([(ob_tree_state, hyp_state, tuple())])
        visited = set()

        while queue:
            if self.max_queue_size is not None and len(queue) > self.max_queue_size:
                return None

            tree_node, current_hyp_state, prefix = queue.popleft()
            if tree_node is None or current_hyp_state is None:
                continue
            if not self._is_search_relevant_tree_node(ob_tree, tree_node):
                continue

            state_pair = (tree_node.id, id(current_hyp_state))
            if state_pair in visited:
                continue
            visited.add(state_pair)

            if self.max_nodes_visited is not None and len(visited) > self.max_nodes_visited:
                return None
            if max_depth is not None and len(prefix) >= max_depth:
                continue

            local_n_c = self._effective_n_c(n_c, prefix)
            for inp in self._ordered_inputs_for_tree_hypothesis_pair(tree_node, current_hyp_state, ob_tree, local_n_c, effective_diff_checker):
                if ensure_complete is not None and local_n_c > 0 and not ob_tree.is_complete(tree_node, inp, local_n_c):
                    ensure_complete(
                        tree_node,
                        inp,
                        local_n_c,
                        max_attempts=completion_max_attempts,
                        store_failed_prefixes=store_failed_prefixes,
                        completion_rollout_len=completion_rollout_len,
                    )

                if local_n_c > 0 and not ob_tree.is_complete(tree_node, inp, local_n_c):
                    continue

                c_tree = ob_tree.get_output_frequencies(tree_node, inp)
                c_hyp, output_to_state = self._hypothesis_output_frequencies(current_hyp_state, inp)
                if self._tree_hypothesis_cells_are_apart(c_tree, c_hyp):
                    return prefix + (inp,)

                shared = set(c_tree.keys()).intersection(c_hyp.keys())
                for out in sorted(shared, key=lambda x: repr(x)):
                    succ_tree = tree_node.get_successor(inp, out)
                    succ_hyp = output_to_state.get(out)
                    if succ_tree is None or succ_hyp is None:
                        continue
                    if not self._is_search_relevant_tree_node(ob_tree, succ_tree):
                        continue
                    queue.append((succ_tree, succ_hyp, prefix + (inp,)))
        return None

    def _effective_n_c(self, base_n_c, prefix):
        if base_n_c <= 0:
            return 0
        depth = len(prefix)
        if depth >= 3:
            cap = self.depth_n_c_caps.get(3, 6)
        else:
            cap = self.depth_n_c_caps.get(depth)
        if cap is None:
            return base_n_c
        return min(base_n_c, cap)

    def _ordered_inputs_for_pair(self, node1, node2, ob_tree, local_n_c, diff_checker):
        def _key(inp):
            c1 = ob_tree.get_output_frequencies(node1, inp)
            c2 = ob_tree.get_output_frequencies(node2, inp)
            both_complete = int(
                ob_tree.is_complete(node1, inp, local_n_c)
                and ob_tree.is_complete(node2, inp, local_n_c)
            )
            total_samples = node1.get_frequency_sum(inp) + node2.get_frequency_sum(inp)
            diff_value = self.difference_value(c1, c2, diff_checker=diff_checker)
            return (
                both_complete,
                total_samples,
                diff_value,
                inp,
            )

        return sorted(ob_tree.alphabet, key=_key, reverse=True)

    def _ordered_inputs_for_tree_hypothesis_pair(self, tree_node, hyp_state, ob_tree, local_n_c, diff_checker):
        def _key(inp):
            c_tree = ob_tree.get_output_frequencies(tree_node, inp)
            c_hyp, _ = self._hypothesis_output_frequencies(hyp_state, inp)
            complete = int(ob_tree.is_complete(tree_node, inp, local_n_c))
            total_samples = tree_node.get_frequency_sum(inp)
            diff_value = self.tree_hypothesis_difference_value(c_tree, c_hyp)
            return (complete, total_samples, diff_value, inp)

        return sorted(ob_tree.alphabet, key=_key, reverse=True)

    @staticmethod
    def _ordered_shared_outputs(c1: dict, c2: dict):
        shared = set(c1.keys()).intersection(c2.keys())
        return sorted(shared, key=lambda out: (c1.get(out, 0) + c2.get(out, 0), repr(out)), reverse=True)

    def _bfs_find_witness(self, s1, s2, ob_tree, n_c=0, ensure_complete=None, max_depth=None, diff_checker=None):
        effective_diff_checker = self._resolve_diff_checker(diff_checker)
        initial_pair = (s1.id, s2.id)
        queue = deque([(s1, s2, tuple())])
        visited = set()
        queued = {initial_pair}
        nodes_visited = 0

        while queue:
            if self.max_queue_size is not None and len(queue) > self.max_queue_size:
                if hasattr(ob_tree, '_log'):
                    ob_tree._log(
                        'apartness_queue_limit_reached',
                        queue_size=len(queue),
                        max_queue_size=self.max_queue_size,
                    )
                return None

            node1, node2, prefix = queue.popleft()
            if node1 is None or node2 is None:
                continue

            state_pair = (node1.id, node2.id)
            queued.discard(state_pair)
            if state_pair in visited:
                continue
            visited.add(state_pair)
            nodes_visited += 1

            if self.max_nodes_visited is not None and nodes_visited > self.max_nodes_visited:
                if hasattr(ob_tree, '_log'):
                    ob_tree._log(
                        'apartness_nodes_limit_reached',
                        nodes_visited=nodes_visited,
                        max_nodes_visited=self.max_nodes_visited,
                    )
                return None

            if max_depth is not None and len(prefix) >= max_depth:
                continue

            local_n_c = self._effective_n_c(n_c, prefix)
            completion_max_attempts = getattr(self, '_current_completion_max_attempts', None)
            store_failed_prefixes = getattr(self, '_current_store_failed_prefixes', True)
            completion_rollout_len = getattr(self, '_current_completion_rollout_len', 1)
            ordered_inputs = self._ordered_inputs_for_pair(node1, node2, ob_tree, local_n_c, effective_diff_checker)

            for inp in ordered_inputs:
                if ensure_complete is not None and local_n_c > 0:
                    if not ob_tree.is_complete(node1, inp, local_n_c):
                        ensure_complete(
                            node1,
                            inp,
                            local_n_c,
                            max_attempts=completion_max_attempts,
                            store_failed_prefixes=store_failed_prefixes,
                            completion_rollout_len=completion_rollout_len,
                        )
                    if not ob_tree.is_complete(node2, inp, local_n_c):
                        ensure_complete(
                            node2,
                            inp,
                            local_n_c,
                            max_attempts=completion_max_attempts,
                            store_failed_prefixes=store_failed_prefixes,
                            completion_rollout_len=completion_rollout_len,
                        )

                if local_n_c > 0 and not ob_tree.is_complete(node1, inp, local_n_c):
                    continue
                if local_n_c > 0 and not ob_tree.is_complete(node2, inp, local_n_c):
                    continue

                c1 = ob_tree.get_output_frequencies(node1, inp)
                c2 = ob_tree.get_output_frequencies(node2, inp)
                if self._cells_are_apart(c1, c2, diff_checker=effective_diff_checker):
                    return prefix + (inp,)

                for out in self._ordered_shared_outputs(c1, c2):
                    succ1 = node1.get_successor(inp, out)
                    succ2 = node2.get_successor(inp, out)
                    if succ1 is None or succ2 is None:
                        continue
                    if not self._is_search_relevant_tree_node(ob_tree, succ1):
                        continue
                    if not self._is_search_relevant_tree_node(ob_tree, succ2):
                        continue

                    succ_pair = (succ1.id, succ2.id)
                    if succ_pair in visited or succ_pair in queued:
                        continue

                    queue.append((succ1, succ2, prefix + (inp,)))
                    queued.add(succ_pair)
        return None

    def _hypothesis_output_frequencies(self, hyp_state, inp):
        output_prob = defaultdict(float)
        output_to_state = {}
        for target_state, output, probability in getattr(hyp_state, 'transitions', {}).get(inp, []):
            output_prob[output] += probability
            output_to_state.setdefault(output, target_state)
        return dict(output_prob), output_to_state

    def _cells_are_apart(self, c1: dict, c2: dict, diff_checker=None) -> bool:
        effective_diff_checker = self._resolve_diff_checker(diff_checker)
        return effective_diff_checker.are_cells_different(c1, c2)

    def _tree_hypothesis_cells_are_apart(self, observed_counts: dict, hypothesis_probs: dict) -> bool:
        if any(hypothesis_probs.get(out, 0.0) <= 0.0 for out in observed_counts.keys()):
            return True
        is_different, _ = self._empirical_vs_hypothesis_different(observed_counts, hypothesis_probs)
        return is_different

    def _empirical_vs_hypothesis_different(self, observed_counts: dict, hypothesis_probs: dict):
        total = sum(observed_counts.values()) if observed_counts else 0
        if total <= 0:
            return False, {
                'threshold': None,
                'max_gap': 0.0,
            }

        support = set(observed_counts.keys()).union(hypothesis_probs.keys())
        nr_events = max(1, len(support))
        threshold = sqrt(log(2 * nr_events / self.alpha) / (2 * total))
        max_gap = 0.0
        for out in support:
            empirical = observed_counts.get(out, 0) / total
            expected = hypothesis_probs.get(out, 0.0)
            max_gap = max(max_gap, abs(empirical - expected))
        return max_gap > threshold, {
            'threshold': threshold,
            'max_gap': max_gap,
        }

    def tree_hypothesis_difference_value(self, observed_counts: dict, hypothesis_probs: dict):
        if any(hypothesis_probs.get(out, 0.0) <= 0.0 for out in observed_counts.keys()):
            return float('inf')
        _, diagnostics = self._empirical_vs_hypothesis_different(observed_counts, hypothesis_probs)
        threshold = diagnostics.get('threshold')
        if threshold is None:
            return float('-inf')
        return diagnostics.get('max_gap', 0.0) - threshold

    def difference_value(self, c1: dict, c2: dict, diff_checker=None):
        effective_diff_checker = self._resolve_diff_checker(diff_checker)
        if hasattr(effective_diff_checker, 'use_diff_value') and effective_diff_checker.use_diff_value():
            return effective_diff_checker.difference_value(c1, c2)

        n1 = sum(c1.values()) if c1 else 0
        n2 = sum(c2.values()) if c2 else 0
        if n1 <= 0 or n2 <= 0:
            return float('-inf')
        threshold = self._threshold(n1, n2)
        max_gap = 0.0
        for o in set(c1.keys()).union(c2.keys()):
            p1 = c1.get(o, 0) / n1
            p2 = c2.get(o, 0) / n2
            max_gap = max(max_gap, abs(p1 - p2))
        return max_gap - threshold

    def _threshold(self, n1, n2):
        return (sqrt(1 / n1) + sqrt(1 / n2)) * sqrt(0.5 * log(2 / self.alpha))

    @staticmethod
    def _shared_outputs(c1: dict, c2: dict):
        return set(c1.keys()).intersection(c2.keys())

    @staticmethod
    def _get_cache_key(s1, s2, n_c, max_depth):
        ids = tuple(sorted((s1.id, s2.id)))
        return ids, n_c, max_depth
