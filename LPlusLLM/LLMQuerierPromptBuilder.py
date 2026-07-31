from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_ALLOWED_PURPOSES = (
    "complete_branch",
    "distinguish_frontier",
    "refine_hypothesis",
)


@dataclass(slots=True)
class QuerierPromptConstraints:
    """
    Round-dependent prompt constraints for the LLM Querier.

    This object belongs to the PromptBuilder layer because these values are
    prompt-time planning controls. They are intentionally not stored inside the
    observation-tree evidence snapshot.
    """

    max_query_length: int = 16
    remaining_sampling_budget: int | None = None
    top_k: int | None = None
    allowed_purposes: tuple[str, ...] = DEFAULT_ALLOWED_PURPOSES
    query_mode: str = "extend_and_rank"  # rank_only | extend_and_rank | propose_and_rank
    max_candidate_queries: int = 64
    max_incomplete_items: int = 50
    max_unresolved_frontier_items: int = 30
    max_apartness_witness_items: int = 50
    include_full_nodes: bool = True

    def __post_init__(self) -> None:
        if self.max_query_length <= 0:
            raise ValueError("max_query_length must be positive.")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive when provided.")
        if self.max_candidate_queries <= 0:
            raise ValueError("max_candidate_queries must be positive.")
        if self.query_mode not in {"rank_only", "extend_and_rank", "propose_and_rank"}:
            raise ValueError("query_mode must be 'rank_only', 'extend_and_rank', or 'propose_and_rank'.")
        self.allowed_purposes = tuple(self.allowed_purposes or DEFAULT_ALLOWED_PURPOSES)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["allowed_purposes"] = list(self.allowed_purposes)
        return data


@dataclass(slots=True)
class QuerierPromptPackage:
    """PromptBuilder output consumed by LLMQuerier."""

    prompt: str
    candidate_pool: list[dict[str, Any]]
    constraints: dict[str, Any]
    prompt_view: dict[str, Any]
    prompt_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "candidate_pool": self.candidate_pool,
            "constraints": self.constraints,
            "prompt_view": self.prompt_view,
            "prompt_path": self.prompt_path,
        }


class LLMQuerierPromptBuilder:
    """
    Build structured prompts for the LLM Querier.

    Responsibilities of this class are deliberately narrow:
      1. receive or load serialized_observation_tree.json;
      2. receive or load querier_context.json;
      3. construct a symbolic candidate query pool;
      4. construct the final prompt text;
      5. save the prompt into this file's directory by default.

    It does not call the LLM, parse LLM output, filter proposals, or execute SUL
    sampling. Those steps belong to LLMQuerier.
    """

    DEFAULT_TREE_FILE = "serialized_observation_tree.json"
    DEFAULT_CONTEXT_FILE = "querier_context.json"
    DEFAULT_PROMPT_FILE = "querier_prompt.txt"

    def __init__(
        self,
        constraints: QuerierPromptConstraints | None = None,
        *,
        output_dir: str | Path | None = None,
    ) -> None:
        self.constraints = constraints or QuerierPromptConstraints()
        self.output_dir = Path(output_dir).resolve() if output_dir is not None else Path(__file__).resolve().parent

    # ------------------------------------------------------------------
    # File helpers
    # ------------------------------------------------------------------
    @staticmethod
    def load_json(path: str | Path) -> dict[str, Any]:
        with Path(path).open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected a JSON object in {path!s}.")
        return payload

    @staticmethod
    def save_text(text: str, path: str | Path) -> Path:
        output_path = Path(path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        return output_path

    def default_tree_path(self) -> Path:
        return self.output_dir / self.DEFAULT_TREE_FILE

    def default_context_path(self) -> Path:
        return self.output_dir / self.DEFAULT_CONTEXT_FILE

    def default_prompt_path(self) -> Path:
        return self.output_dir / self.DEFAULT_PROMPT_FILE

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return [value]

    @staticmethod
    def _as_dict(value: Any) -> dict[str, Any]:
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _json_dumps(payload: Any) -> str:
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)

    @classmethod
    def _node_id(cls, node: Mapping[str, Any]) -> int | None:
        try:
            return int(node.get("id"))
        except (TypeError, ValueError):
            return None

    @classmethod
    def _node_index(cls, tree_snapshot: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
        result: dict[int, dict[str, Any]] = {}
        for raw_node in cls._as_list(tree_snapshot.get("nodes")):
            node = cls._as_dict(raw_node)
            node_id = cls._node_id(node)
            if node_id is not None:
                result[node_id] = node
        return result

    @classmethod
    def _alphabet(cls, tree_snapshot: Mapping[str, Any], querier_context: Mapping[str, Any]) -> list[Any]:
        tree_meta = cls._as_dict(tree_snapshot.get("meta"))
        context_meta = cls._as_dict(querier_context.get("meta"))
        alphabet = tree_meta.get("alphabet") or context_meta.get("alphabet") or []
        return list(cls._as_list(alphabet))

    @staticmethod
    def _dedupe_key(inputs: Sequence[Any], purpose: str, target_node_id: Any) -> tuple[Any, ...]:
        return (tuple(inputs), purpose, target_node_id)

    # ------------------------------------------------------------------
    # Candidate pool construction
    # ------------------------------------------------------------------
    def build_candidate_pool(
        self,
        tree_snapshot: Mapping[str, Any],
        querier_context: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Construct symbolic candidates from current tree evidence.

        Candidate sources:
          - incomplete node/input pairs from CQ-style context;
          - unresolved frontier nodes plus available apartness witnesses;
          - short local frontier refinement continuations.
        """
        constraints = self.constraints
        node_by_id = self._node_index(tree_snapshot)
        alphabet = self._alphabet(tree_snapshot, querier_context)
        candidates: list[dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()

        def add_candidate(
            inputs: Sequence[Any],
            *,
            purpose: str,
            target_node_id: int | None,
            source: str,
            reason: str,
            evidence: dict[str, Any] | None = None,
        ) -> None:
            if purpose not in constraints.allowed_purposes:
                return
            seq = list(inputs)
            if not seq or len(seq) > constraints.max_query_length:
                return
            key = self._dedupe_key(seq, purpose, target_node_id)
            if key in seen:
                return
            seen.add(key)
            extension_capacity = max(0, constraints.max_query_length - len(seq))
            evidence_payload = dict(evidence or {})
            evidence_payload.setdefault("required_prefix", list(seq))
            evidence_payload.setdefault("extension_capacity", extension_capacity)
            candidates.append(
                {
                    "id": f"cq{len(candidates) + 1}",
                    # In extend_and_rank mode this is a required prefix / seed.
                    # The LLM may append symbols after this prefix up to
                    # max_query_length, but it must not alter or remove it.
                    "inputs": seq,
                    "required_prefix": list(seq),
                    "extension_capacity": extension_capacity,
                    "extension_policy": (
                        "preserve required_prefix exactly; append alphabet symbols heuristically "
                        "until the full query is close to max_query_length"
                        if extension_capacity > 0
                        else "already at max_query_length; do not extend"
                    ),
                    "purpose": purpose,
                    "target_node_id": target_node_id,
                    "source": source,
                    "reason": reason,
                    "evidence": evidence_payload,
                }
            )

        # 1. Complete under-sampled branches. The executable query is the node's
        # access sequence plus the incomplete suffix/input.
        incomplete_items = self._as_list(querier_context.get("incomplete"))[: constraints.max_incomplete_items]
        role_priority = {"basis": 0, "frontier": 1, "remain": 2}
        incomplete_items = sorted(
            (self._as_dict(item) for item in incomplete_items),
            key=lambda item: (
                role_priority.get(str(item.get("role")), 9),
                -int(item.get("missing") or 0),
                int(item.get("node_id") or -1),
                repr(item.get("input")),
            ),
        )
        for item in incomplete_items:
            node_id = item.get("node_id")
            try:
                node_id_int = int(node_id)
            except (TypeError, ValueError):
                continue
            node = node_by_id.get(node_id_int)
            if node is None:
                continue
            suffix = item.get("input_suffix") or [item.get("input")]
            suffix = [x for x in self._as_list(suffix) if x is not None]
            access_sequence = list(self._as_list(node.get("access_sequence")))
            add_candidate(
                access_sequence + suffix,
                purpose="complete_branch",
                target_node_id=node_id_int,
                source="incomplete",
                reason=(
                    f"Complete under-sampled input {suffix!r} from node {node_id_int}; "
                    f"missing={item.get('missing')} samples."
                ),
                evidence={
                    "node_role": item.get("role"),
                    "missing": item.get("missing"),
                    "input_suffix": suffix,
                },
            )

        # 2. Use apartness witnesses between candidate basis nodes to target
        # unresolved frontier ambiguity.
        witness_by_pair: dict[frozenset[int], list[Any]] = {}
        for witness_item in self._as_list(querier_context.get("apartness_witness"))[: constraints.max_apartness_witness_items]:
            witness_item = self._as_dict(witness_item)
            try:
                s1 = int(witness_item.get("state_id_1"))
                s2 = int(witness_item.get("state_id_2"))
            except (TypeError, ValueError):
                continue
            witness = self._as_list(witness_item.get("witness"))
            if witness:
                witness_by_pair[frozenset((s1, s2))] = witness

        unresolved_frontier = self._as_list(querier_context.get("unresolved_frontier"))[: constraints.max_unresolved_frontier_items]
        unresolved_frontier_ids: set[int] = set()
        for item in unresolved_frontier:
            item = self._as_dict(item)
            try:
                frontier_node_id = int(item.get("node_id"))
            except (TypeError, ValueError):
                continue
            unresolved_frontier_ids.add(frontier_node_id)
            frontier_node = node_by_id.get(frontier_node_id)
            if frontier_node is None:
                continue
            candidate_basis_ids: list[int] = []
            for raw_id in self._as_list(item.get("candidate_basis_ids")):
                try:
                    candidate_basis_ids.append(int(raw_id))
                except (TypeError, ValueError):
                    pass
            access_sequence = list(self._as_list(frontier_node.get("access_sequence") or item.get("access_sequence")))
            for i, basis_1 in enumerate(candidate_basis_ids):
                for basis_2 in candidate_basis_ids[i + 1 :]:
                    witness = witness_by_pair.get(frozenset((basis_1, basis_2)))
                    if not witness:
                        continue
                    add_candidate(
                        access_sequence + witness,
                        purpose="distinguish_frontier",
                        target_node_id=frontier_node_id,
                        source="unresolved_frontier_with_witness",
                        reason=(
                            f"Resolve frontier node {frontier_node_id} between basis candidates "
                            f"{basis_1} and {basis_2} using known witness {witness!r}."
                        ),
                        evidence={
                            "candidate_basis_ids": candidate_basis_ids,
                            "witness": witness,
                            "basis_pair": [basis_1, basis_2],
                        },
                    )

        # 3. Local frontier refinement: short one-symbol continuations from
        # unresolved or weakly sampled frontier nodes.
        frontier_nodes = [node for node in node_by_id.values() if node.get("role") == "frontier"]
        frontier_nodes.sort(key=lambda node: (int(node.get("depth") or 0), -int(node.get("frequency") or 0), int(node.get("id") or -1)))
        for node in frontier_nodes:
            try:
                node_id = int(node.get("id"))
            except (TypeError, ValueError):
                continue
            access_sequence = list(self._as_list(node.get("access_sequence")))
            transitions = self._as_dict(node.get("transitions"))
            for inp in alphabet:
                transition = self._as_dict(transitions.get(inp))
                sample_count = int(transition.get("sample_count") or 0)
                add_candidate(
                    access_sequence + [inp],
                    purpose="distinguish_frontier",
                    target_node_id=node_id,
                    source="frontier_local_extension",
                    reason=(
                        f"Probe frontier-local behavior from node {node_id} using input {inp!r}; "
                        f"current sample_count={sample_count}."
                    ),
                    evidence={"node_role": "frontier", "input": inp, "sample_count": sample_count},
                )
                if len(candidates) >= constraints.max_candidate_queries:
                    return self._prioritize_unresolved_frontier_candidates(
                        candidates,
                        unresolved_frontier_ids,
                        constraints.max_candidate_queries,
                    )

        return self._prioritize_unresolved_frontier_candidates(
            candidates,
            unresolved_frontier_ids,
            constraints.max_candidate_queries,
        )

    @staticmethod
    def _prioritize_unresolved_frontier_candidates(
        candidates: Sequence[Mapping[str, Any]],
        unresolved_frontier_ids: set[int],
        max_candidates: int,
    ) -> list[dict[str, Any]]:
        """
        If frontier ambiguity blocks hypothesis construction, spend the LLM
        budget on resolving that ambiguity instead of completing unrelated
        branches. This is intentionally a hard priority because adequacy
        requires every frontier node to map to exactly one basis candidate.
        """
        normalized = [dict(candidate) for candidate in candidates]
        if unresolved_frontier_ids:
            frontier_candidates = [
                candidate
                for candidate in normalized
                if candidate.get("purpose") == "distinguish_frontier"
                and candidate.get("target_node_id") in unresolved_frontier_ids
            ]
            if frontier_candidates:
                return frontier_candidates[:max_candidates]
        return normalized[:max_candidates]

    # ------------------------------------------------------------------
    # Prompt view and prompt construction
    # ------------------------------------------------------------------
    def build_prompt_view(
        self,
        tree_snapshot: Mapping[str, Any],
        querier_context: Mapping[str, Any],
        candidate_pool: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        constraints = self.constraints.to_dict()
        tree_meta = self._as_dict(tree_snapshot.get("meta"))
        context_meta = self._as_dict(querier_context.get("meta"))

        prompt_view: dict[str, Any] = {
            "meta": {
                "automaton_type": tree_meta.get("automaton_type"),
                "alphabet": self._alphabet(tree_snapshot, querier_context),
                "root_id": tree_meta.get("root_id") or context_meta.get("root_id"),
                "active_n_c": tree_meta.get("active_n_c") or context_meta.get("active_n_c"),
                "active_apartness_n_c": tree_meta.get("active_apartness_n_c") or context_meta.get("active_apartness_n_c"),
                "basis_size": tree_meta.get("basis_size"),
                "frontier_size": tree_meta.get("frontier_size"),
                "serialized_node_count": tree_meta.get("serialized_node_count"),
                "role_counts": tree_meta.get("role_counts"),
            },
            "constraints": constraints,
            "planning_context": {
                "incomplete": self._as_list(querier_context.get("incomplete"))[: self.constraints.max_incomplete_items],
                "unresolved_frontier": self._as_list(querier_context.get("unresolved_frontier"))[
                    : self.constraints.max_unresolved_frontier_items
                ],
                "apartness_witness": self._as_list(querier_context.get("apartness_witness"))[
                    : self.constraints.max_apartness_witness_items
                ],
            },
            "candidate_pool": list(candidate_pool),
        }

        if self.constraints.include_full_nodes:
            prompt_view["observation_tree"] = {
                "learning_parameters": tree_snapshot.get("learning_parameters", {}),
                "nodes": self._as_list(tree_snapshot.get("nodes")),
            }

        return prompt_view

    def build_prompt(
        self,
        tree_snapshot: Mapping[str, Any],
        querier_context: Mapping[str, Any],
        candidate_pool: Sequence[Mapping[str, Any]],
    ) -> str:
        prompt_view = self.build_prompt_view(tree_snapshot, querier_context, candidate_pool)
        constraints = self.constraints

        if constraints.query_mode == "rank_only":
            mode_instruction = (
                "Rank only the provided candidate_pool entries. Treat each candidate inputs field as a complete query; "
                "do not extend and do not invent additional queries."
            )
        elif constraints.query_mode == "extend_and_rank":
            mode_instruction = (
                "Treat each candidate_pool entry as a query seed. Its inputs field is a required prefix. "
                "Rank the seeds and return full input sequences by preserving the required prefix exactly and "
                "heuristically appending symbols from the alphabet up to max_query_length when useful. "
                "Do not invent new required prefixes outside the candidate_pool."
            )
        else:
            mode_instruction = (
                "Rank the provided candidate_pool seeds and optionally add a small number of grounded supplementary queries. "
                "Any supplementary query must still be justified by the serialized observation tree and satisfy all constraints."
            )

        schema = {
            "ranked_queries": [
                {
                    "inputs": ["input_symbol_1", "input_symbol_2"],
                    "purpose": "complete_branch | distinguish_frontier | refine_hypothesis",
                    "target_node_id": 0,
                    "priority": 0.0,
                    "reason": "brief evidence-grounded reason",
                }
            ]
        }

        ranked_query_limit_instruction = (
            f"Return at most {constraints.top_k} ranked queries."
            if constraints.top_k is not None
            else "Return all admissible candidate queries, ranked from most useful to least useful."
        )

        sections = [
            "You are a query planner for stochastic active model learning of a black-box stochastic Mealy machine.",
            "Your role is to prioritize input sequences for the next sampling/refinement step. You are not a behavioral oracle.",
            "Use only the evidence provided in the observation tree, planning context, and candidate pool.",
            "",
            "Decision objective:",
            "1. Complete under-sampled branches when they block reliable statistical comparison.",
            "2. Distinguish unresolved frontier nodes when candidate basis mappings are ambiguous.",
            "3. Refine uncertain local regions of the current hypothesis structure.",
            "4. Prefer longer admissible input sequences because one sampled episode can insert a longer trace into the observation tree.",
            "5. When two queries have comparable refinement value, rank the longer valid query higher.",
            "",
            "Hard constraints:",
            f"- Input symbols must come only from: {self._alphabet(tree_snapshot, querier_context)!r}.",
            f"- Query length must be <= {constraints.max_query_length}.",
            f"- Allowed purposes are: {list(constraints.allowed_purposes)!r}.",
            f"- Remaining sampling budget is: {constraints.remaining_sampling_budget!r}. One sampled input sequence costs one episode budget, independent of its length.",
            f"- {ranked_query_limit_instruction}",
            "- Do not invent unseen states, node ids, outputs, witnesses, or behavioral facts.",
            "- Do not return explanations outside the JSON object.",
            f"- Mode: {constraints.query_mode}. {mode_instruction}",
            "- In extend_and_rank mode, each candidate_pool entry is a required-prefix seed. The returned inputs must start with that required_prefix exactly.",
            "- In extend_and_rank mode, use the serialized observation tree heuristically to append useful suffix symbols until the full query is close to max_query_length.",
            "- Do not shorten a required prefix. Do not exceed max_query_length. Do not use symbols outside the alphabet.",
            "- For complete_branch, preserve the incomplete branch prefix and extend only after that prefix.",
            "- For distinguish_frontier, preserve the frontier access-sequence prefix and extend only after that prefix.",
            "",
            "Required output schema:",
            self._json_dumps(schema),
            "",
            "Learner state for query planning:",
            self._json_dumps(prompt_view),
            "",
            "Return JSON only.",
        ]
        return "\n".join(sections)

    def build_prompt_package(
        self,
        tree_snapshot: Mapping[str, Any],
        querier_context: Mapping[str, Any],
        *,
        save_prompt: bool = True,
        prompt_path: str | Path | None = None,
    ) -> QuerierPromptPackage:
        candidate_pool = self.build_candidate_pool(tree_snapshot, querier_context)
        prompt_view = self.build_prompt_view(tree_snapshot, querier_context, candidate_pool)
        prompt = self.build_prompt(tree_snapshot, querier_context, candidate_pool)

        resolved_prompt_path: str | None = None
        if save_prompt:
            path = Path(prompt_path).resolve() if prompt_path is not None else self.default_prompt_path()
            resolved_prompt_path = str(self.save_text(prompt, path))

        return QuerierPromptPackage(
            prompt=prompt,
            candidate_pool=candidate_pool,
            constraints=self.constraints.to_dict(),
            prompt_view=prompt_view,
            prompt_path=resolved_prompt_path,
        )

    def build_prompt_package_from_files(
        self,
        *,
        tree_path: str | Path | None = None,
        context_path: str | Path | None = None,
        save_prompt: bool = True,
        prompt_path: str | Path | None = None,
    ) -> QuerierPromptPackage:
        tree_snapshot = self.load_json(tree_path or self.default_tree_path())
        querier_context = self.load_json(context_path or self.default_context_path())
        return self.build_prompt_package(
            tree_snapshot,
            querier_context,
            save_prompt=save_prompt,
            prompt_path=prompt_path,
        )


__all__ = [
    "QuerierPromptConstraints",
    "QuerierPromptPackage",
    "LLMQuerierPromptBuilder",
]


if __name__ == "__main__":
    builder = LLMQuerierPromptBuilder()
    package = builder.build_prompt_package_from_files(save_prompt=True)
    print(package.prompt_path)
