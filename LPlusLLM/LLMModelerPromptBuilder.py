from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(slots=True)
class ModelerPromptConstraints:
    """
    Prompt-time constraints for LLM-based candidate hypothesis synthesis.

    These values describe what the LLM is allowed to synthesize. They are kept
    outside the observation-tree evidence snapshot so the tree remains a factual
    record of sampled behavior.
    """

    max_states: int | None = None
    include_full_nodes: bool = True
    probability_sum_tolerance: float = 0.02
    observation_probability_tolerance: float = 0.15
    preserve_basis: bool = True
    enforce_apartness: bool = True
    strict_apartness: bool = False
    allow_inferred_states: bool = True
    require_complete_input_transitions: bool = True

    def __post_init__(self) -> None:
        if self.max_states is not None and self.max_states <= 0:
            raise ValueError("max_states must be positive when provided.")
        if self.probability_sum_tolerance < 0:
            raise ValueError("probability_sum_tolerance must be non-negative.")
        if self.observation_probability_tolerance < 0:
            raise ValueError("observation_probability_tolerance must be non-negative.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ModelerPromptPackage:
    """PromptBuilder output consumed by the future LLMModeler."""

    prompt: str
    constraints: dict[str, Any]
    prompt_view: dict[str, Any]
    response_schema: dict[str, Any]
    prompt_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "constraints": self.constraints,
            "prompt_view": self.prompt_view,
            "response_schema": self.response_schema,
            "prompt_path": self.prompt_path,
        }


class LLMModelerPromptBuilder:
    """
    Build structured prompts for LLM-based candidate SMM synthesis.

    Responsibilities are deliberately narrow:
      1. receive or load a serialized observation-tree snapshot;
      2. receive or load optional modeler context derived from the learner;
      3. construct a compact prompt view and strict complete-SMM output schema;
      4. save the prompt artifact when requested.

    This class does not call the LLM, parse model candidates, construct AALpy
    automata, or validate candidates. Those steps belong to LLMModeler.
    """

    DEFAULT_TREE_FILE = "serialized_observation_tree.json"
    DEFAULT_CONTEXT_FILE = "modeler_context.json"
    FALLBACK_CONTEXT_FILE = "querier_context.json"
    DEFAULT_PROMPT_FILE = "modeler_prompt.txt"

    def __init__(
        self,
        constraints: ModelerPromptConstraints | None = None,
        *,
        output_dir: str | Path | None = None,
    ) -> None:
        self.constraints = constraints or ModelerPromptConstraints()
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

    def fallback_context_path(self) -> Path:
        return self.output_dir / self.FALLBACK_CONTEXT_FILE

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
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    @classmethod
    def _nodes(cls, tree_snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
        return [cls._as_dict(node) for node in cls._as_list(tree_snapshot.get("nodes"))]

    @classmethod
    def _alphabet(cls, tree_snapshot: Mapping[str, Any], modeler_context: Mapping[str, Any]) -> list[Any]:
        tree_meta = cls._as_dict(tree_snapshot.get("meta"))
        context_meta = cls._as_dict(modeler_context.get("meta"))
        alphabet = tree_meta.get("alphabet") or context_meta.get("alphabet") or modeler_context.get("alphabet") or []
        return list(cls._as_list(alphabet))

    @classmethod
    def _output_alphabet(cls, tree_snapshot: Mapping[str, Any], modeler_context: Mapping[str, Any]) -> list[Any]:
        explicit = modeler_context.get("output_alphabet")
        if explicit is not None:
            outputs = sorted(cls._as_list(explicit), key=repr)
            return outputs or ["no_response"]

        outputs = set()
        for node in cls._nodes(tree_snapshot):
            transitions = cls._as_dict(node.get("transitions"))
            for transition in transitions.values():
                for output_item in cls._as_list(cls._as_dict(transition).get("outputs")):
                    output = cls._as_dict(output_item).get("output")
                    if output is not None:
                        outputs.add(output)
        return sorted(outputs, key=repr) or ["no_response"]

    @classmethod
    def _observed_transition_count(cls, tree_snapshot: Mapping[str, Any]) -> int:
        count = 0
        for node in cls._nodes(tree_snapshot):
            transitions = cls._as_dict(node.get("transitions"))
            for transition in transitions.values():
                if cls._as_dict(transition).get("sample_count", 0):
                    count += 1
        return count

    @classmethod
    def _apart_node_pairs(
        cls,
        modeler_context: Mapping[str, Any],
    ) -> list[Any]:
        explicit_pairs = modeler_context.get("apart_node_pairs")
        if explicit_pairs is None:
            explicit_pairs = modeler_context.get("apartness_pairs")
        if explicit_pairs is not None:
            return cls._as_list(explicit_pairs)

        pairs = []
        for item in cls._as_list(modeler_context.get("apartness_witness")):
            witness = cls._as_dict(item)
            left = witness.get("state_id_1")
            right = witness.get("state_id_2")
            if left is None or right is None:
                pair = cls._as_list(witness.get("pair"))
                if len(pair) >= 2:
                    left, right = pair[0], pair[1]
            if left is None or right is None:
                continue
            pairs.append({
                "node_id_1": left,
                "node_id_2": right,
            })
        return pairs

    # ------------------------------------------------------------------
    # JSON schema
    # ------------------------------------------------------------------
    @staticmethod
    def response_schema() -> dict[str, Any]:
        """
        Return the strict JSON schema expected from the LLM.

        The future LLMModeler can pass this schema to a JSON-schema-capable
        provider and can also reuse it for local validation before constructing
        an AALpy StochasticMealyMachine.
        """

        state_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "id": {"type": "string"},
                "source_basis_node": {"type": ["integer", "null"]},
            },
            "required": ["id", "source_basis_node"],
        }

        distribution_item_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "output": {"type": "string"},
                "probability": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "to": {"type": "string"},
            },
            "required": ["output", "probability", "to"],
        }

        transition_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "from": {"type": "string"},
                "input": {"type": "string"},
                "distribution": {
                    "type": "array",
                    "minItems": 1,
                    "items": distribution_item_schema,
                },
            },
            "required": ["from", "input", "distribution"],
        }

        mapping_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "tree_node_id": {"type": "integer"},
                "mapped_state": {"type": "string"},
                "basis_justification": {"type": "string"},
            },
            "required": ["tree_node_id", "mapped_state", "basis_justification"],
        }

        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "states": {
                    "type": "array",
                    "minItems": 1,
                    "items": state_schema,
                },
                "initial_state": {"type": "string"},
                "transitions": {
                    "type": "array",
                    "items": transition_schema,
                },
                "state_mapping_explanation": {
                    "type": "array",
                    "items": mapping_schema,
                },
            },
            "required": ["states", "initial_state", "transitions", "state_mapping_explanation"],
        }

    # ------------------------------------------------------------------
    # Prompt view and prompt construction
    # ------------------------------------------------------------------
    def build_prompt_view(
        self,
        tree_snapshot: Mapping[str, Any],
        modeler_context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        modeler_context = self._as_dict(modeler_context)
        constraints = self.constraints.to_dict()
        tree_meta = self._as_dict(tree_snapshot.get("meta"))
        context_meta = self._as_dict(modeler_context.get("meta"))

        prompt_view: dict[str, Any] = {
            "meta": {
                "automaton_type": tree_meta.get("automaton_type"),
                "input_alphabet": self._alphabet(tree_snapshot, modeler_context),
                "output_alphabet": self._output_alphabet(tree_snapshot, modeler_context),
                "root_id": tree_meta.get("root_id") or context_meta.get("root_id"),
                "active_n_c": tree_meta.get("active_n_c") or context_meta.get("active_n_c"),
                "active_apartness_n_c": tree_meta.get("active_apartness_n_c")
                or context_meta.get("active_apartness_n_c"),
                "basis_size": tree_meta.get("basis_size"),
                "frontier_size": tree_meta.get("frontier_size"),
                "serialized_node_count": tree_meta.get("serialized_node_count"),
                "role_counts": tree_meta.get("role_counts"),
                "observed_transition_count": self._observed_transition_count(tree_snapshot),
            },
            "constraints": constraints,
            "modeling_context": {
                "apart_node_pairs": self._apart_node_pairs(modeler_context),
            },
        }

        if self.constraints.include_full_nodes:
            prompt_view["observation_tree"] = {
                "learning_parameters": tree_snapshot.get("learning_parameters", {}),
                "nodes": self._nodes(tree_snapshot),
            }

        return prompt_view

    def build_prompt(
        self,
        tree_snapshot: Mapping[str, Any],
        modeler_context: Mapping[str, Any] | None = None,
    ) -> str:
        prompt_view = self.build_prompt_view(tree_snapshot, modeler_context)
        constraints = self.constraints
        schema = self.response_schema()
        alphabet = prompt_view["meta"]["input_alphabet"]
        output_alphabet = prompt_view["meta"]["output_alphabet"]

        state_limit_instruction = (
            f"- Use at most {constraints.max_states} states."
            if constraints.max_states is not None
            else "- Use the smallest conservative state set justified by basis/frontier/remain roles, apart node pairs, and deep tree evidence."
        )

        sections = [
            "You are a Hypothesis Builder for stochastic active learning of a black-box stochastic Mealy machine.",
            "Your role is to synthesize one complete candidate stochastic Mealy machine from the accumulated observation-tree evidence.",
            "",
            "Modeling objective:",
            "1. Construct a candidate SMM with states, initial_state, and stochastic transitions.",
            "2. Preserve all verified observations and respect apartness constraints.",
            "3. Use the full observation tree, including remain nodes, to make conservative structural inferences.",
            "4. Keep the model probabilistically well formed and as simple as the evidence permits.",
            "",
            "Hard constraints:",
            f"- Input symbols must come only from: {alphabet!r}.",
            f"- Output symbols must come only from observed outputs: {output_alphabet!r}.",
            "- If no concrete output has been observed yet, use the provided fallback output symbol 'no_response'.",
            "- Every returned state id must be unique.",
            "- initial_state must reference one returned state.",
            state_limit_instruction,
            "- Preserve current basis states explicitly: each basis node should correspond to a returned state source_basis_node.",
            "- Do not merge nodes listed in modeling_context.apart_node_pairs.",
            "- Do not contradict observed input/output paths or empirical transition evidence.",
            "- For every returned state and every input symbol, include exactly one transition object.",
            "- Each transition distribution must be non-empty, use probabilities in [0, 1], and sum to 1.",
            "- Use conservative completion for unresolved parts; unsupported completion should use low-complexity behavior justified by compatible evidence.",
            "- state_mapping_explanation should explain important non-basis mappings, especially unresolved frontier and uncertain remain nodes.",
            "- Return explanations only inside state_mapping_explanation. Do not add text outside the JSON object.",
            "",
            "Required output JSON schema:",
            self._json_dumps(schema),
            "",
            "Learner state for candidate-model synthesis:",
            self._json_dumps(prompt_view),
            "",
            "Return JSON only.",
        ]
        return "\n".join(sections)

    def build_prompt_package(
        self,
        tree_snapshot: Mapping[str, Any],
        modeler_context: Mapping[str, Any] | None = None,
        *,
        save_prompt: bool = True,
        prompt_path: str | Path | None = None,
    ) -> ModelerPromptPackage:
        prompt_view = self.build_prompt_view(tree_snapshot, modeler_context)
        prompt = self.build_prompt(tree_snapshot, modeler_context)

        resolved_prompt_path: str | None = None
        if save_prompt:
            path = Path(prompt_path).resolve() if prompt_path is not None else self.default_prompt_path()
            resolved_prompt_path = str(self.save_text(prompt, path))

        return ModelerPromptPackage(
            prompt=prompt,
            constraints=self.constraints.to_dict(),
            prompt_view=prompt_view,
            response_schema=self.response_schema(),
            prompt_path=resolved_prompt_path,
        )

    def build_prompt_package_from_files(
        self,
        *,
        tree_path: str | Path | None = None,
        context_path: str | Path | None = None,
        save_prompt: bool = True,
        prompt_path: str | Path | None = None,
    ) -> ModelerPromptPackage:
        tree_snapshot = self.load_json(tree_path or self.default_tree_path())

        resolved_context_path = Path(context_path).resolve() if context_path is not None else self.default_context_path()
        if resolved_context_path.exists():
            modeler_context = self.load_json(resolved_context_path)
        elif context_path is None and self.fallback_context_path().exists():
            modeler_context = self.load_json(self.fallback_context_path())
        else:
            modeler_context = {}

        return self.build_prompt_package(
            tree_snapshot,
            modeler_context,
            save_prompt=save_prompt,
            prompt_path=prompt_path,
        )


__all__ = [
    "ModelerPromptConstraints",
    "ModelerPromptPackage",
    "LLMModelerPromptBuilder",
]


if __name__ == "__main__":
    builder = LLMModelerPromptBuilder()
    package = builder.build_prompt_package_from_files(save_prompt=True)
    print(package.prompt_path)
