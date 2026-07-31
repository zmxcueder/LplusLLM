from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

try:  # package import
    from .config import DEEPSEEK_MODEL, LLMClientConfig, LLM_PROVIDER, OPENAI_MODEL, OPENROUTER_MODEL
    from .LLMModelerPromptBuilder import LLMModelerPromptBuilder, ModelerPromptPackage
except ImportError:  # pragma: no cover - standalone / local execution
    from config import DEEPSEEK_MODEL, LLMClientConfig, LLM_PROVIDER, OPENAI_MODEL, OPENROUTER_MODEL  # type: ignore
    from LLMModelerPromptBuilder import LLMModelerPromptBuilder, ModelerPromptPackage  # type: ignore


class BuiltinLLMModelerApiClient:
    """Internal OpenAI-compatible API client used by LLMModeler."""

    def __init__(
        self,
        config: LLMClientConfig | None = None,
        prompt_builder: LLMModelerPromptBuilder | None = None,
    ) -> None:
        self.config = config or LLMClientConfig()
        self.prompt_builder = prompt_builder or LLMModelerPromptBuilder()

    def response_schema(self) -> dict[str, Any]:
        return self.prompt_builder.response_schema()

    def complete(self, prompt: str) -> str:
        api_key = self.config.resolved_api_key()
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("Install the OpenAI Python SDK first: pip install openai") from exc

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if self.config.base_url:
            client_kwargs["base_url"] = self.config.base_url
        client = OpenAI(**client_kwargs)

        if self.config.provider in {"deepseek", "openrouter"}:
            return self._chat_completions_json_object(client, prompt)

        if self.config.use_json_schema_for_openai:
            try:
                return self._responses_json_schema(client, prompt)
            except Exception:
                return self._chat_completions_json_schema(client, prompt)
        return self._chat_completions_json_object(client, prompt)

    def _messages(self, prompt: str) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You synthesize complete stochastic Mealy machine candidates for stochastic active model learning. "
                    "Return only a valid JSON object matching the requested SMM schema."
                ),
            },
            {"role": "user", "content": prompt},
        ]

    def _responses_json_schema(self, client: Any, prompt: str) -> str:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "input": self._messages(prompt),
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "llm_modeler_candidate_smm",
                    "strict": True,
                    "schema": self.response_schema(),
                }
            },
        }
        if self.config.max_output_tokens is not None:
            kwargs["max_output_tokens"] = self.config.max_output_tokens
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature
        response = client.responses.create(**kwargs)
        return self._response_to_text(response)

    def _chat_completions_json_schema(self, client: Any, prompt: str) -> str:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": self._messages(prompt),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "llm_modeler_candidate_smm",
                    "strict": True,
                    "schema": self.response_schema(),
                },
            },
        }
        if self.config.max_output_tokens is not None:
            kwargs["max_tokens"] = self.config.max_output_tokens
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature
        response = client.chat.completions.create(**kwargs)
        return self._choice_content(response)

    def _chat_completions_json_object(self, client: Any, prompt: str) -> str:
        json_prompt = prompt + "\n\nReturn a valid JSON object only."
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": self._messages(json_prompt),
            "response_format": {"type": "json_object"},
        }
        if self.config.max_output_tokens is not None:
            kwargs["max_tokens"] = self.config.max_output_tokens
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature
        response = client.chat.completions.create(**kwargs)
        return self._choice_content(response)

    @staticmethod
    def _choice_content(response: Any) -> str:
        if response is None:
            raise RuntimeError("LLM provider returned no response object.")
        choices = getattr(response, "choices", None)
        if not choices:
            raise RuntimeError(f"LLM provider returned no choices. Response: {response!r}")
        choice = choices[0]
        if choice is None:
            raise RuntimeError(f"LLM provider returned an empty first choice. Response: {response!r}")
        finish_reason = getattr(choice, "finish_reason", None)
        message = getattr(choice, "message", None)
        content = getattr(message, "content", None)
        if content is None and isinstance(message, Mapping):
            content = message.get("content")
        content = content or ""
        if finish_reason == "length":
            raise RuntimeError(
                "LLM response was truncated by the provider. "
                "Reduce prompt size or set LLM_MAX_OUTPUT_TOKENS to a larger integer."
            )
        if not content.strip():
            raise RuntimeError("LLM returned an empty response content.")
        return content

    @staticmethod
    def _response_to_text(response: Any) -> str:
        if response is None:
            raise RuntimeError("LLM provider returned no response object.")
        text = getattr(response, "output_text", None)
        if isinstance(text, str):
            return text
        if isinstance(response, str):
            return response
        if isinstance(response, Mapping):
            for key in ("output_text", "text", "content", "message", "output"):
                value = response.get(key)
                if isinstance(value, str):
                    return value
            return json.dumps(response, ensure_ascii=False)
        return str(response)


@dataclass(slots=True)
class LLMModelerConfig:
    """Runtime behavior for LLMModeler."""

    max_retries: int = 1
    save_artifacts: bool = True
    artifact_dir: str | Path | None = None
    artifact_prefix: str = "modeler_round"
    artifact_mode: str = "minimal"  # minimal | full
    raise_on_failure: bool = False
    api_config: LLMClientConfig = field(default_factory=LLMClientConfig)

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative.")
        if self.artifact_mode not in {"minimal", "full"}:
            raise ValueError("artifact_mode must be 'minimal' or 'full'.")
        if not self.artifact_prefix:
            raise ValueError("artifact_prefix must not be empty.")


@dataclass(slots=True)
class ValidationReport:
    accepted: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked_constraints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "checked_constraints": list(self.checked_constraints),
        }


@dataclass(slots=True)
class ModelerRoundResult:
    prompt_package: ModelerPromptPackage | None
    raw_llm_output: str | None
    parsed_candidate: dict[str, Any] | None
    validation_report: ValidationReport
    hypothesis: Any | None = None
    accepted: bool = False
    artifacts: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    elapsed_time_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_package": self.prompt_package.to_dict() if self.prompt_package is not None else None,
            "raw_llm_output": self.raw_llm_output,
            "parsed_candidate": self.parsed_candidate,
            "validation_report": self.validation_report.to_dict(),
            "accepted": self.accepted,
            "artifacts": dict(self.artifacts),
            "error": self.error,
            "elapsed_time_sec": self.elapsed_time_sec,
            "hypothesis_states": len(getattr(self.hypothesis, "states", []) or []) if self.hypothesis is not None else 0,
        }


class LLMModeler:
    """LLM-based candidate hypothesis synthesizer for stochastic L+LLM."""

    def __init__(
        self,
        prompt_builder: LLMModelerPromptBuilder | None = None,
        config: LLMModelerConfig | None = None,
        llm_client: Any | None = None,
    ) -> None:
        self.prompt_builder = prompt_builder or LLMModelerPromptBuilder()
        self.config = config or LLMModelerConfig()
        self.llm_client = llm_client or BuiltinLLMModelerApiClient(self.config.api_config, self.prompt_builder)
        self.stats: dict[str, int] = {
            "rounds": 0,
            "llm_calls": 0,
            "parse_failures": 0,
            "validation_failures": 0,
            "accepted_candidates": 0,
            "rejected_candidates": 0,
            "candidate_states": 0,
            "candidate_transitions": 0,
        }

    # ------------------------------------------------------------------
    # Prompting and LLM I/O
    # ------------------------------------------------------------------
    def build_prompt_package(
        self,
        tree_snapshot: Mapping[str, Any],
        modeler_context: Mapping[str, Any] | None = None,
        *,
        save_prompt: bool | None = None,
        prompt_path: str | Path | None = None,
    ) -> ModelerPromptPackage:
        return self.prompt_builder.build_prompt_package(
            tree_snapshot,
            modeler_context,
            save_prompt=self.config.save_artifacts if save_prompt is None else save_prompt,
            prompt_path=prompt_path,
        )

    def call_llm(self, prompt: str) -> str:
        self.stats["llm_calls"] += 1
        client = self.llm_client
        if callable(client):
            response = client(prompt)
        elif hasattr(client, "complete"):
            response = client.complete(prompt)
        elif hasattr(client, "generate"):
            response = client.generate(prompt)
        elif hasattr(client, "invoke"):
            response = client.invoke(prompt)
        elif hasattr(client, "create"):
            response = client.create(prompt=prompt)
        else:
            raise TypeError("Unsupported llm_client. Provide complete/generate/invoke/create or a callable.")
        return self._response_to_text(response)

    @staticmethod
    def _response_to_text(response: Any) -> str:
        if response is None:
            return ""
        if isinstance(response, str):
            return response
        if isinstance(response, Mapping):
            for key in ("text", "content", "message", "output"):
                value = response.get(key)
                if isinstance(value, str):
                    return value
            return json.dumps(response, ensure_ascii=False)
        for attr in ("text", "content", "message", "output"):
            value = getattr(response, attr, None)
            if isinstance(value, str):
                return value
        return str(response)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------
    def parse_candidate(self, raw_text: str) -> dict[str, Any] | None:
        payload = self._extract_json_payload(raw_text)
        if not isinstance(payload, Mapping):
            self.stats["parse_failures"] += 1
            return None
        candidate = dict(payload)
        if not any(key in candidate for key in ("states", "initial_state", "state_mapping_explanation", "transitions")):
            self.stats["parse_failures"] += 1
            return None
        return candidate

    @classmethod
    def _extract_json_payload(cls, raw_text: str) -> Any | None:
        if not raw_text or not str(raw_text).strip():
            return None
        text = str(raw_text).strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            try:
                return json.loads(fenced.group(1).strip())
            except json.JSONDecodeError:
                pass
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None

    # ------------------------------------------------------------------
    # Candidate materialization
    # ------------------------------------------------------------------
    def materialize_candidate(
        self,
        candidate: Mapping[str, Any] | None,
        *,
        tree_snapshot: Mapping[str, Any],
        modeler_context: Mapping[str, Any] | None = None,
        prompt_package: ModelerPromptPackage | None = None,
    ) -> dict[str, Any] | None:
        if candidate is None or not isinstance(candidate, Mapping):
            return None

        prompt_view = prompt_package.prompt_view if prompt_package is not None else self.prompt_builder.build_prompt_view(
            tree_snapshot,
            modeler_context,
        )
        meta = self._as_dict(prompt_view.get("meta"))
        constraints = dict(self._as_dict(prompt_view.get("constraints")))
        if self._as_dict(modeler_context).get("budget_finalization"):
            constraints["preserve_basis"] = False
            constraints["strict_apartness"] = True
        input_alphabet = self._as_list(meta.get("input_alphabet"))
        output_alphabet = self._as_list(meta.get("output_alphabet"))
        tree_nodes = self._node_index(tree_snapshot)
        if not tree_nodes:
            return dict(candidate)

        root_id = meta.get("root_id")
        try:
            root_id = int(root_id)
        except (TypeError, ValueError):
            root_id = self._as_dict(tree_snapshot.get("meta")).get("root_id")
        try:
            root_id = int(root_id)
        except (TypeError, ValueError):
            root_id = min(tree_nodes)

        states_by_id: dict[str, dict[str, Any]] = {}
        basis_state_by_node: dict[int, str] = {}

        def unique_state_id(base: str) -> str:
            candidate_id = base or "S"
            if candidate_id not in states_by_id:
                return candidate_id
            suffix = 1
            while f"{candidate_id}_{suffix}" in states_by_id:
                suffix += 1
            return f"{candidate_id}_{suffix}"

        def add_state(
            state_id: str,
            *,
            source_basis_node: int | None = None,
            source_tree_nodes: Sequence[Any] | None = None,
            role: str = "basis",
        ) -> str:
            sid = unique_state_id(str(state_id))
            normalized_nodes = []
            for raw_node_id in self._as_list(source_tree_nodes):
                node_id = self._coerce_int(raw_node_id)
                if node_id is not None and node_id in tree_nodes and node_id not in normalized_nodes:
                    normalized_nodes.append(node_id)
            source_basis_id = self._coerce_int(source_basis_node)
            if source_basis_id is not None and source_basis_id in tree_nodes and source_basis_id not in normalized_nodes:
                normalized_nodes.insert(0, source_basis_id)
            states_by_id[sid] = {
                "id": sid,
                "source_basis_node": source_basis_id if source_basis_id is not None and source_basis_id in tree_nodes else None,
                "source_tree_nodes": normalized_nodes,
                "role": role if role in {"basis", "inferred"} else "basis",
            }
            if source_basis_id is not None and source_basis_id in tree_nodes:
                basis_state_by_node[source_basis_id] = sid
            return sid

        for raw_state in self._as_list(candidate.get("states")):
            state = self._as_dict(raw_state)
            state_id = state.get("id")
            if not isinstance(state_id, str) or not state_id:
                continue
            add_state(
                state_id,
                source_basis_node=state.get("source_basis_node"),
                source_tree_nodes=state.get("source_tree_nodes"),
                role=state.get("role", "basis"),
            )

        basis_node_ids = sorted(node_id for node_id, node in tree_nodes.items() if node.get("role") == "basis")
        if constraints.get("preserve_basis", True):
            for node_id in basis_node_ids:
                if node_id not in basis_state_by_node:
                    add_state(f"S{node_id}", source_basis_node=node_id, source_tree_nodes=[node_id], role="basis")

        if not states_by_id:
            add_state(f"S{root_id}", source_basis_node=root_id if root_id in tree_nodes else None, source_tree_nodes=[root_id], role="basis")

        if not basis_state_by_node and root_id in tree_nodes:
            first_state_id = next(iter(states_by_id))
            first_state = states_by_id[first_state_id]
            first_state["source_basis_node"] = root_id
            if root_id not in first_state["source_tree_nodes"]:
                first_state["source_tree_nodes"].insert(0, root_id)
            basis_state_by_node[root_id] = first_state_id

        node_to_state: dict[int, str] = {}
        for state_id, state in states_by_id.items():
            for node_id in self._as_list(state.get("source_tree_nodes")):
                if isinstance(node_id, int) and node_id in tree_nodes:
                    node_to_state.setdefault(node_id, state_id)
            source_basis_node = state.get("source_basis_node")
            if isinstance(source_basis_node, int) and source_basis_node in tree_nodes:
                node_to_state[source_basis_node] = state_id

        for raw_mapping in self._as_list(candidate.get("state_mapping_explanation")):
            mapping = self._as_dict(raw_mapping)
            node_id = self._coerce_int(mapping.get("tree_node_id"))
            state_id = mapping.get("mapped_state")
            if isinstance(node_id, int) and node_id in tree_nodes and isinstance(state_id, str) and state_id in states_by_id:
                node_to_state.setdefault(node_id, state_id)

        apart_pairs = set(tuple(sorted(pair)) for pair in self._apart_node_pairs(prompt_view, modeler_context))
        for node_id, node in sorted(tree_nodes.items()):
            if node_id in node_to_state:
                continue
            node_to_state[node_id] = self._choose_state_for_unmapped_node(
                node_id,
                node,
                tree_nodes=tree_nodes,
                basis_state_by_node=basis_state_by_node,
                apart_pairs=apart_pairs,
            )

        for state in states_by_id.values():
            state["source_tree_nodes"] = sorted(
                node_id for node_id, mapped_state in node_to_state.items() if mapped_state == state["id"]
            )

        initial_state = node_to_state.get(root_id)
        if initial_state is None or initial_state not in states_by_id:
            raw_initial = candidate.get("initial_state")
            initial_state = raw_initial if isinstance(raw_initial, str) and raw_initial in states_by_id else next(iter(states_by_id))

        explanations = self._materialized_mapping_explanations(candidate, node_to_state, tree_nodes)
        candidate_transitions = self._as_list(candidate.get("transitions"))
        if candidate_transitions:
            materialized_transitions = [dict(self._as_dict(item)) for item in candidate_transitions]
        else:
            materialized_transitions = self._construct_transitions_from_observation_tree(
                tree_nodes=tree_nodes,
                node_to_state=node_to_state,
                state_ids=set(states_by_id),
                input_alphabet=input_alphabet,
                output_alphabet=output_alphabet,
                initial_state=initial_state,
            )

        return {
            "states": list(states_by_id.values()),
            "initial_state": initial_state,
            "transitions": materialized_transitions,
            "state_mapping_explanation": explanations,
        }

    def _choose_state_for_unmapped_node(
        self,
        node_id: int,
        node: Mapping[str, Any],
        *,
        tree_nodes: Mapping[int, Mapping[str, Any]],
        basis_state_by_node: Mapping[int, str],
        apart_pairs: set[tuple[int, int]],
    ) -> str:
        candidate_basis = [
            raw_id
            for raw_id in self._as_list(node.get("frontier_candidate_basis_ids"))
            if isinstance(raw_id, int) and raw_id in basis_state_by_node
        ]
        basis_ids = candidate_basis or sorted(basis_state_by_node)
        if not basis_ids:
            raise ValueError("Cannot map observation-tree node without any available basis state.")

        best_basis = None
        best_score = None
        for basis_id in basis_ids:
            if tuple(sorted((node_id, basis_id))) in apart_pairs:
                continue
            score = self._transition_signature_score(node, tree_nodes.get(basis_id, {}))
            sort_key = (score, -abs(int(node.get("depth", 0) or 0) - int(self._as_dict(tree_nodes.get(basis_id)).get("depth", 0) or 0)), -basis_id)
            if best_score is None or sort_key > best_score:
                best_score = sort_key
                best_basis = basis_id
        if best_basis is None:
            best_basis = basis_ids[0]
        return basis_state_by_node[best_basis]

    def _transition_signature_score(self, node: Mapping[str, Any], basis_node: Mapping[str, Any]) -> int:
        score = 0
        node_transitions = self._as_dict(node.get("transitions"))
        basis_transitions = self._as_dict(basis_node.get("transitions"))
        for inp, raw_transition in node_transitions.items():
            outputs = {
                self._as_dict(item).get("output")
                for item in self._as_list(self._as_dict(raw_transition).get("outputs"))
                if self._as_dict(item).get("output") is not None
            }
            basis_outputs = {
                self._as_dict(item).get("output")
                for item in self._as_list(self._as_dict(basis_transitions.get(inp)).get("outputs"))
                if self._as_dict(item).get("output") is not None
            }
            score += 4 * len(outputs & basis_outputs)
            score -= 2 * len(outputs - basis_outputs)
            score -= len(basis_outputs - outputs)
        return score

    def _construct_transitions_from_observation_tree(
        self,
        *,
        tree_nodes: Mapping[int, Mapping[str, Any]],
        node_to_state: Mapping[int, str],
        state_ids: set[str],
        input_alphabet: Sequence[Any],
        output_alphabet: Sequence[Any],
        initial_state: str,
    ) -> list[dict[str, Any]]:
        buckets: dict[tuple[str, Any], dict[tuple[Any, str], dict[str, Any]]] = {}
        global_buckets: dict[Any, dict[tuple[Any, str], int]] = {}

        for node_id, node in tree_nodes.items():
            source_state = node_to_state.get(node_id)
            if source_state not in state_ids:
                continue
            for inp, raw_transition in self._as_dict(node.get("transitions")).items():
                transition = self._as_dict(raw_transition)
                outputs = self._as_list(transition.get("outputs"))
                if not outputs:
                    continue
                key = (source_state, inp)
                state_bucket = buckets.setdefault(key, {})
                global_bucket = global_buckets.setdefault(inp, {})
                for raw_output in outputs:
                    item = self._as_dict(raw_output)
                    output = item.get("output")
                    if output is None:
                        continue
                    try:
                        count = int(item.get("count", 1) or 1)
                    except (TypeError, ValueError):
                        count = 1
                    successor_id = item.get("successor_id")
                    target_state = node_to_state.get(successor_id) if isinstance(successor_id, int) else None
                    if target_state not in state_ids:
                        target_state = source_state
                    dist_key = (output, target_state)
                    entry = state_bucket.setdefault(
                        dist_key,
                        {
                            "count": 0,
                            "evidence_node_id": node_id,
                            "evidence_sample_count": 0,
                        },
                    )
                    entry["count"] += max(1, count)
                    entry["evidence_sample_count"] += max(1, count)
                    global_bucket[dist_key] = global_bucket.get(dist_key, 0) + max(1, count)

        transitions: list[dict[str, Any]] = []
        fallback_output = output_alphabet[0] if output_alphabet else None
        for state_id in sorted(state_ids):
            for inp in input_alphabet:
                state_bucket = buckets.get((state_id, inp))
                if state_bucket:
                    transitions.append(self._bucket_to_transition(state_id, inp, state_bucket))
                    continue
                global_bucket = global_buckets.get(inp)
                if global_bucket:
                    converted = {
                        dist_key: {
                            "count": count,
                            "evidence_node_id": None,
                            "evidence_sample_count": None,
                        }
                        for dist_key, count in global_bucket.items()
                    }
                    transitions.append(self._bucket_to_transition(state_id, inp, converted))
                    continue
                transitions.append(
                    {
                        "from": state_id,
                        "input": inp,
                        "distribution": [
                            {
                                "output": fallback_output,
                                "probability": 1.0,
                                "to": initial_state,
                                "evidence_node_id": None,
                                "evidence_sample_count": None,
                            }
                        ],
                    }
                )
        return transitions

    @staticmethod
    def _bucket_to_transition(state_id: str, inp: Any, bucket: Mapping[tuple[Any, str], Mapping[str, Any]]) -> dict[str, Any]:
        total = sum(max(0, int(entry.get("count", 0) or 0)) for entry in bucket.values())
        if total <= 0:
            total = 1
        distribution = []
        for (output, target_state), entry in sorted(bucket.items(), key=lambda item: (repr(item[0][0]), item[0][1])):
            count = max(0, int(entry.get("count", 0) or 0))
            distribution.append(
                {
                    "output": output,
                    "probability": count / total,
                    "to": target_state,
                    "evidence_node_id": entry.get("evidence_node_id"),
                    "evidence_sample_count": entry.get("evidence_sample_count"),
                }
            )
        return {"from": state_id, "input": inp, "distribution": distribution}

    def _materialized_mapping_explanations(
        self,
        candidate: Mapping[str, Any],
        node_to_state: Mapping[int, str],
        tree_nodes: Mapping[int, Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        explanations: dict[int, dict[str, Any]] = {}
        for raw_mapping in self._as_list(candidate.get("state_mapping_explanation")):
            mapping = self._as_dict(raw_mapping)
            node_id = self._coerce_int(mapping.get("tree_node_id"))
            mapped_state = mapping.get("mapped_state")
            if isinstance(node_id, int) and node_id in tree_nodes and isinstance(mapped_state, str):
                explanations[node_id] = {
                    "tree_node_id": node_id,
                    "mapped_state": mapped_state,
                    "basis_justification": str(mapping.get("basis_justification", "")),
                    "confidence": mapping.get("confidence") if isinstance(mapping.get("confidence"), (int, float)) else 0.5,
                }
        for node_id, mapped_state in node_to_state.items():
            if node_id in explanations:
                continue
            role = self._as_dict(tree_nodes.get(node_id)).get("role")
            if role == "basis":
                justification = "Basis node state materialized locally."
                confidence = 1.0
            else:
                justification = "Mapping materialized locally from LLM state partition and observation-tree compatibility."
                confidence = 0.5
            explanations[node_id] = {
                "tree_node_id": node_id,
                "mapped_state": mapped_state,
                "basis_justification": justification,
                "confidence": confidence,
            }
        return [explanations[node_id] for node_id in sorted(explanations)]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate_candidate(
        self,
        candidate: Mapping[str, Any] | None,
        *,
        tree_snapshot: Mapping[str, Any],
        modeler_context: Mapping[str, Any] | None = None,
        prompt_package: ModelerPromptPackage | None = None,
    ) -> ValidationReport:
        errors: list[str] = []
        warnings: list[str] = []
        checked: list[str] = []

        if candidate is None:
            return ValidationReport(False, ["candidate_is_none"], [], ["parseable_json_candidate"])
        if not isinstance(candidate, Mapping):
            return ValidationReport(False, ["candidate_must_be_json_object"], [], ["json_object"])

        prompt_view = prompt_package.prompt_view if prompt_package is not None else self.prompt_builder.build_prompt_view(
            tree_snapshot,
            modeler_context,
        )
        constraints = dict(prompt_view.get("constraints", {}) if isinstance(prompt_view, Mapping) else {})
        if self._as_dict(modeler_context).get("budget_finalization"):
            constraints["preserve_basis"] = False
            constraints["strict_apartness"] = True
        meta = prompt_view.get("meta", {}) if isinstance(prompt_view, Mapping) else {}
        input_alphabet = set(self._as_list(meta.get("input_alphabet")))
        output_alphabet = set(self._as_list(meta.get("output_alphabet")))
        tree_nodes = self._node_index(tree_snapshot)
        basis_node_ids = {
            node_id
            for node_id, node in tree_nodes.items()
            if node.get("role") == "basis"
        }
        root_id = meta.get("root_id")
        apart_node_pairs = self._apart_node_pairs(prompt_view, modeler_context)

        states = candidate.get("states")
        initial_state = candidate.get("initial_state")
        transitions = candidate.get("transitions")
        state_mapping_explanation = candidate.get("state_mapping_explanation")

        checked.append("top_level_schema")
        self._reject_extra_keys(
            warnings,
            candidate,
            {"states", "initial_state", "transitions", "state_mapping_explanation"},
            "candidate",
        )
        if not isinstance(states, list) or not states:
            errors.append("states_must_be_non_empty_list")
            states = []
        if not isinstance(initial_state, str) or not initial_state:
            errors.append("initial_state_must_be_non_empty_string")
        if not isinstance(transitions, list):
            errors.append("transitions_must_be_list")
            transitions = []
        if not isinstance(state_mapping_explanation, list):
            warnings.append("state_mapping_explanation_must_be_list")
            state_mapping_explanation = []

        state_ids: set[str] = set()
        state_to_tree_nodes: dict[str, set[int]] = {}
        tree_node_to_state: dict[int, str] = {}

        checked.append("state_schema")
        for idx, raw_state in enumerate(states):
            if not isinstance(raw_state, Mapping):
                errors.append(f"state[{idx}]_must_be_object")
                continue
            self._reject_extra_keys(
                warnings,
                raw_state,
                {"id", "source_basis_node", "source_tree_nodes", "role"},
                f"state[{idx}]",
            )
            state_id = raw_state.get("id")
            if not isinstance(state_id, str) or not state_id:
                errors.append(f"state[{idx}].id_must_be_non_empty_string")
                continue
            if state_id in state_ids:
                errors.append(f"duplicate_state_id:{state_id}")
            state_ids.add(state_id)

            role = raw_state.get("role")
            if role not in {"basis", "inferred"}:
                warnings.append(f"state[{state_id}].role_must_be_basis_or_inferred")

            source_basis_node = raw_state.get("source_basis_node")
            if source_basis_node is not None:
                if not isinstance(source_basis_node, int):
                    warnings.append(f"state[{state_id}].source_basis_node_must_be_integer_or_null")
                elif source_basis_node not in tree_nodes:
                    warnings.append(f"state[{state_id}].source_basis_node_not_in_tree:{source_basis_node}")
                elif tree_nodes[source_basis_node].get("role") != "basis":
                    warnings.append(f"state[{state_id}].source_basis_node_not_basis:{source_basis_node}")

            source_tree_nodes = raw_state.get("source_tree_nodes")
            if not isinstance(source_tree_nodes, list):
                warnings.append(f"state[{state_id}].source_tree_nodes_must_be_list")
                source_tree_nodes = []
            normalized_tree_nodes: set[int] = set()
            for raw_node_id in source_tree_nodes:
                if not isinstance(raw_node_id, int):
                    warnings.append(f"state[{state_id}].source_tree_node_must_be_integer:{raw_node_id!r}")
                    continue
                if raw_node_id not in tree_nodes:
                    warnings.append(f"state[{state_id}].source_tree_node_not_in_tree:{raw_node_id}")
                    continue
                normalized_tree_nodes.add(raw_node_id)
                previous = tree_node_to_state.get(raw_node_id)
                if previous is not None and previous != state_id:
                    warnings.append(f"tree_node_mapped_to_multiple_states:{raw_node_id}:{previous}:{state_id}")
                tree_node_to_state[raw_node_id] = state_id
            if isinstance(source_basis_node, int):
                if source_basis_node in tree_nodes:
                    normalized_tree_nodes.add(source_basis_node)
                    previous = tree_node_to_state.get(source_basis_node)
                    if previous is not None and previous != state_id:
                        warnings.append(f"basis_node_mapped_to_multiple_states:{source_basis_node}:{previous}:{state_id}")
                    tree_node_to_state[source_basis_node] = state_id
            state_to_tree_nodes[state_id] = normalized_tree_nodes

        max_states = constraints.get("max_states")
        if max_states is not None and isinstance(max_states, int) and len(state_ids) > max_states:
            errors.append(f"too_many_states:{len(state_ids)}>{max_states}")

        checked.append("initial_state")
        if isinstance(initial_state, str) and initial_state not in state_ids:
            errors.append(f"initial_state_not_defined:{initial_state}")
        if root_id is not None and isinstance(initial_state, str) and initial_state in state_to_tree_nodes:
            try:
                root_int = int(root_id)
            except (TypeError, ValueError):
                root_int = None
            if root_int is not None and root_int not in state_to_tree_nodes.get(initial_state, set()):
                warnings.append(f"root_node_not_mapped_to_initial_state:{root_int}:{initial_state}")

        checked.append("basis_preservation")
        if constraints.get("preserve_basis", True):
            missing_basis = sorted(node_id for node_id in basis_node_ids if node_id not in tree_node_to_state)
            if missing_basis:
                warnings.append(f"missing_basis_nodes:{missing_basis}")

        checked.append("apartness_consistency")
        if constraints.get("enforce_apartness", True):
            strict_apartness = bool(constraints.get("strict_apartness", False))
            for left, right in apart_node_pairs:
                left_state = tree_node_to_state.get(left)
                right_state = tree_node_to_state.get(right)
                if left_state is not None and left_state == right_state:
                    message = f"apart_nodes_merged:{left}:{right}:state={left_state}"
                    if strict_apartness:
                        errors.append(message)
                    else:
                        warnings.append(message)
                elif left_state is None or right_state is None:
                    message = f"apart_pair_not_fully_mapped:{left}:{right}"
                    if strict_apartness:
                        errors.append(message)
                    else:
                        warnings.append(message)

        checked.append("transition_schema")
        transition_groups: dict[tuple[str, Any], list[Mapping[str, Any]]] = {}
        for idx, raw_transition in enumerate(transitions):
            if not isinstance(raw_transition, Mapping):
                errors.append(f"transition[{idx}]_must_be_object")
                continue
            self._reject_extra_keys(
                warnings,
                raw_transition,
                {"from", "input", "distribution"},
                f"transition[{idx}]",
            )
            src = raw_transition.get("from")
            inp = raw_transition.get("input")
            dist = raw_transition.get("distribution")
            if not isinstance(src, str) or src not in state_ids:
                errors.append(f"transition[{idx}].from_not_defined:{src!r}")
            if inp not in input_alphabet:
                errors.append(f"transition[{idx}].input_out_of_alphabet:{inp!r}")
            if not isinstance(dist, list) or not dist:
                errors.append(f"transition[{idx}].distribution_must_be_non_empty_list")
                dist = []
            if isinstance(src, str) and inp in input_alphabet:
                key = (src, inp)
                if key in transition_groups:
                    errors.append(f"duplicate_transition_for_state_input:{src}:{inp!r}")
                transition_groups[key] = list(dist)

            prob_sum = 0.0
            for dist_idx, raw_item in enumerate(dist):
                if not isinstance(raw_item, Mapping):
                    errors.append(f"transition[{idx}].distribution[{dist_idx}]_must_be_object")
                    continue
                self._reject_extra_keys(
                    warnings,
                    raw_item,
                    {"output", "probability", "to", "evidence_node_id", "evidence_sample_count"},
                    f"transition[{idx}].distribution[{dist_idx}]",
                )
                output = raw_item.get("output")
                probability = raw_item.get("probability")
                target = raw_item.get("to")
                evidence_node_id = raw_item.get("evidence_node_id")
                evidence_sample_count = raw_item.get("evidence_sample_count")

                if output not in output_alphabet:
                    errors.append(f"transition[{idx}].distribution[{dist_idx}].output_out_of_alphabet:{output!r}")
                if not isinstance(probability, (int, float)) or isinstance(probability, bool):
                    errors.append(f"transition[{idx}].distribution[{dist_idx}].probability_must_be_number")
                elif probability < 0 or probability > 1:
                    errors.append(f"transition[{idx}].distribution[{dist_idx}].probability_out_of_range:{probability}")
                else:
                    prob_sum += float(probability)
                if not isinstance(target, str) or target not in state_ids:
                    errors.append(f"transition[{idx}].distribution[{dist_idx}].target_not_defined:{target!r}")
                if evidence_node_id is not None:
                    if not isinstance(evidence_node_id, int):
                        warnings.append(f"transition[{idx}].distribution[{dist_idx}].evidence_node_id_must_be_integer_or_null")
                    elif evidence_node_id not in tree_nodes:
                        warnings.append(f"transition[{idx}].distribution[{dist_idx}].evidence_node_id_not_in_tree:{evidence_node_id}")
                if evidence_sample_count is not None:
                    if not isinstance(evidence_sample_count, int) or evidence_sample_count < 0:
                        warnings.append(
                            f"transition[{idx}].distribution[{dist_idx}].evidence_sample_count_must_be_non_negative_integer_or_null"
                        )

            tolerance = float(constraints.get("probability_sum_tolerance", 0.02) or 0.0)
            if dist and abs(prob_sum - 1.0) > tolerance:
                errors.append(f"transition[{idx}].probability_sum_not_one:{prob_sum:.6f}")

        checked.append("state_mapping_explanation_schema")
        if isinstance(state_mapping_explanation, list):
            for idx, raw_mapping in enumerate(state_mapping_explanation):
                if not isinstance(raw_mapping, Mapping):
                    warnings.append(f"state_mapping_explanation[{idx}]_must_be_object")
                    continue
                self._reject_extra_keys(
                    warnings,
                    raw_mapping,
                    {"tree_node_id", "mapped_state", "basis_justification", "confidence"},
                    f"state_mapping_explanation[{idx}]",
                )
                tree_node_id = raw_mapping.get("tree_node_id")
                mapped_state = raw_mapping.get("mapped_state")
                confidence = raw_mapping.get("confidence")
                if not isinstance(tree_node_id, int) or tree_node_id not in tree_nodes:
                    warnings.append(f"state_mapping_explanation[{idx}].tree_node_id_not_in_tree:{tree_node_id!r}")
                if not isinstance(mapped_state, str) or mapped_state not in state_ids:
                    warnings.append(f"state_mapping_explanation[{idx}].mapped_state_not_defined:{mapped_state!r}")
                if not isinstance(raw_mapping.get("basis_justification"), str):
                    warnings.append(f"state_mapping_explanation[{idx}].basis_justification_must_be_string")
                if confidence is not None and (
                    not isinstance(confidence, (int, float))
                    or isinstance(confidence, bool)
                    or confidence < 0
                    or confidence > 1
                ):
                    warnings.append(f"state_mapping_explanation[{idx}].confidence_out_of_range:{confidence!r}")

        checked.append("complete_input_transitions")
        if constraints.get("require_complete_input_transitions", True):
            for state_id in sorted(state_ids):
                for inp in input_alphabet:
                    if (state_id, inp) not in transition_groups:
                        errors.append(f"missing_transition_for_state_input:{state_id}:{inp!r}")

        checked.append("observation_tree_path_consistency")
        self._validate_observation_tree_paths(
            errors,
            warnings,
            tree_nodes=tree_nodes,
            initial_state=initial_state if isinstance(initial_state, str) else None,
            transition_groups=transition_groups,
        )

        accepted = not errors
        if not accepted:
            self.stats["validation_failures"] += 1
        return ValidationReport(accepted=accepted, errors=errors, warnings=warnings, checked_constraints=checked)

    def _validate_observation_tree_paths(
        self,
        errors: list[str],
        warnings: list[str],
        *,
        tree_nodes: Mapping[int, Mapping[str, Any]],
        initial_state: str | None,
        transition_groups: Mapping[tuple[str, Any], Sequence[Mapping[str, Any]]],
    ) -> None:
        if not tree_nodes:
            warnings.append("no_observation_tree_nodes_available_for_path_consistency")
            return
        if initial_state is None:
            warnings.append("path_consistency_skipped_without_valid_initial_state")
            return

        for node_id, node in sorted(tree_nodes.items()):
            possible_states = self._follow_observed_io_trace(
                errors,
                warnings,
                node_id=node_id,
                io_trace=self._as_list(node.get("io_trace")),
                initial_state=initial_state,
                transition_groups=transition_groups,
            )
            if not possible_states:
                continue

            transitions = self._as_dict(node.get("transitions"))
            for inp, raw_transition in transitions.items():
                tree_transition = self._as_dict(raw_transition)
                observed_outputs = self._as_list(tree_transition.get("outputs"))
                if not observed_outputs:
                    continue
                for raw_observed in observed_outputs:
                    observed = self._as_dict(raw_observed)
                    output = observed.get("output")
                    if self._any_candidate_state_supports_output(
                        possible_states,
                        inp,
                        output,
                        transition_groups,
                    ):
                        continue
                    errors.append(
                        "observed_path_output_not_supported:"
                        f"node={node_id}:candidate_states={sorted(possible_states)!r}:"
                        f"input={inp!r}:output={output!r}"
                    )

    def _follow_observed_io_trace(
        self,
        errors: list[str],
        warnings: list[str],
        *,
        node_id: int,
        io_trace: Sequence[Any],
        initial_state: str,
        transition_groups: Mapping[tuple[str, Any], Sequence[Mapping[str, Any]]],
    ) -> set[str]:
        if len(io_trace) % 2 != 0:
            warnings.append(f"node[{node_id}].io_trace_has_odd_length:{list(io_trace)!r}")
            return set()

        possible_states = {initial_state}
        for step_index, (inp, output) in enumerate(zip(io_trace[0::2], io_trace[1::2]), start=1):
            next_states: set[str] = set()
            for state_id in possible_states:
                next_states.update(
                    self._candidate_targets_for_output(
                        state_id,
                        inp,
                        output,
                        transition_groups,
                    )
                )
            if not next_states:
                errors.append(
                    "observed_path_prefix_not_supported:"
                    f"node={node_id}:step={step_index}:candidate_states={sorted(possible_states)!r}:"
                    f"input={inp!r}:output={output!r}"
                )
                return set()
            possible_states = next_states
        return possible_states

    def _any_candidate_state_supports_output(
        self,
        state_ids: set[str],
        inp: Any,
        output: Any,
        transition_groups: Mapping[tuple[str, Any], Sequence[Mapping[str, Any]]],
    ) -> bool:
        return any(
            self._candidate_targets_for_output(state_id, inp, output, transition_groups)
            for state_id in state_ids
        )

    @staticmethod
    def _candidate_targets_for_output(
        state_id: str,
        inp: Any,
        output: Any,
        transition_groups: Mapping[tuple[str, Any], Sequence[Mapping[str, Any]]],
    ) -> set[str]:
        targets: set[str] = set()
        for item in transition_groups.get((state_id, inp), []):
            if not isinstance(item, Mapping):
                continue
            probability = item.get("probability")
            if not isinstance(probability, (int, float)) or isinstance(probability, bool) or probability <= 0:
                continue
            if item.get("output") != output:
                continue
            target = item.get("to")
            if isinstance(target, str):
                targets.add(target)
        return targets

    # ------------------------------------------------------------------
    # Candidate to AALpy SMM
    # ------------------------------------------------------------------
    def candidate_to_smm(
        self,
        candidate: Mapping[str, Any],
        *,
        state_cls: Any | None = None,
        machine_cls: Any | None = None,
    ) -> Any:
        if state_cls is None or machine_cls is None:
            try:
                from aalpy.automata import StochasticMealyMachine, StochasticMealyState
            except ImportError:
                from aalpy.automata.StochasticMealyMachine import StochasticMealyMachine, StochasticMealyState
            state_cls = state_cls or StochasticMealyState
            machine_cls = machine_cls or StochasticMealyMachine

        states_payload = self._as_list(candidate.get("states"))
        states_by_id = {str(raw["id"]): state_cls(str(raw["id"])) for raw in states_payload if isinstance(raw, Mapping)}

        for raw_transition in self._as_list(candidate.get("transitions")):
            transition = self._as_dict(raw_transition)
            source = states_by_id[str(transition.get("from"))]
            inp = transition.get("input")
            for raw_item in self._as_list(transition.get("distribution")):
                item = self._as_dict(raw_item)
                target = states_by_id[str(item.get("to"))]
                source.transitions[inp].append((target, item.get("output"), float(item.get("probability"))))

        initial_state = states_by_id[str(candidate.get("initial_state"))]
        hypothesis = machine_cls(initial_state, list(states_by_id.values()))
        if hasattr(hypothesis, "compute_prefixes"):
            hypothesis.compute_prefixes()
        return hypothesis

    # ------------------------------------------------------------------
    # Validation feedback retry
    # ------------------------------------------------------------------
    def _build_validation_feedback_prompt(
        self,
        *,
        base_prompt: str,
        report: ValidationReport,
        raw_llm_output: str | None,
        parsed_candidate: Mapping[str, Any] | None,
        materialized_candidate: Mapping[str, Any] | None,
        attempt_index: int,
    ) -> str:
        feedback: dict[str, Any] = {
            "previous_attempt_index": attempt_index,
            "accepted": report.accepted,
            "validation_errors": report.errors[:80],
            "validation_warnings": report.warnings[:80],
            "repair_instructions": [
                "Return the same complete-SMM JSON schema as requested in the original prompt.",
                "Include states, initial_state, transitions, and state_mapping_explanation.",
                "For every returned state and input symbol, include exactly one stochastic transition distribution.",
                "Use validation errors to repair only the candidate SMM while staying grounded in observation_tree.nodes.",
            ],
        }
        if parsed_candidate is None:
            feedback["parse_failure"] = True
            feedback["previous_raw_excerpt"] = str(raw_llm_output or "")[:4000]
        else:
            feedback["parsed_mapping_summary"] = self._candidate_mapping_summary(parsed_candidate)
        if materialized_candidate is not None:
            feedback["materialized_candidate_summary"] = self._candidate_mapping_summary(materialized_candidate)

        return "\n\n".join(
            [
                base_prompt,
                "Validation feedback from the previous attempt:",
                self._json_dumps(feedback),
                "Return corrected JSON only.",
            ]
        )

    @classmethod
    def _candidate_mapping_summary(cls, candidate: Mapping[str, Any]) -> dict[str, Any]:
        states = cls._as_list(candidate.get("states"))
        mappings = cls._as_list(candidate.get("state_mapping_explanation"))
        transitions = cls._as_list(candidate.get("transitions"))
        state_ids = []
        mapped_node_count = 0
        for raw_state in states:
            state = cls._as_dict(raw_state)
            state_id = state.get("id")
            if isinstance(state_id, str):
                state_ids.append(state_id)
            mapped_node_count += len(
                [node_id for node_id in cls._as_list(state.get("source_tree_nodes")) if isinstance(node_id, int)]
            )
        return {
            "state_count": len(states),
            "state_ids": state_ids[:80],
            "state_source_tree_node_count": mapped_node_count,
            "mapping_explanation_count": len(mappings),
            "transition_count": len(transitions),
            "top_level_keys": sorted(str(key) for key in candidate.keys()),
        }

    @staticmethod
    def _json_dumps(payload: Any) -> str:
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    # ------------------------------------------------------------------
    # End-to-end synthesis
    # ------------------------------------------------------------------
    def synthesize_candidate(
        self,
        *,
        tree_snapshot: Mapping[str, Any],
        modeler_context: Mapping[str, Any] | None = None,
        prompt_package: ModelerPromptPackage | None = None,
        state_cls: Any | None = None,
        machine_cls: Any | None = None,
    ) -> ModelerRoundResult:
        self.stats["rounds"] += 1
        started = perf_counter()
        prompt_package = prompt_package or self.build_prompt_package(tree_snapshot, modeler_context)
        current_prompt = prompt_package.prompt

        last_result: ModelerRoundResult | None = None
        attempts = self.config.max_retries + 1
        for attempt_index in range(attempts):
            raw_llm_output = self.call_llm(current_prompt)
            parsed_candidate = self.parse_candidate(raw_llm_output)
            candidate = None
            materialization_error = None
            if parsed_candidate is None:
                report = self.validate_candidate(
                    None,
                    tree_snapshot=tree_snapshot,
                    modeler_context=modeler_context,
                    prompt_package=prompt_package,
                )
            else:
                try:
                    candidate = self.materialize_candidate(
                        parsed_candidate,
                        tree_snapshot=tree_snapshot,
                        modeler_context=modeler_context,
                        prompt_package=prompt_package,
                    )
                    report = self.validate_candidate(
                        candidate,
                        tree_snapshot=tree_snapshot,
                        modeler_context=modeler_context,
                        prompt_package=prompt_package,
                    )
                except Exception as exc:
                    materialization_error = f"materialization_failed:{exc}"
                    report = ValidationReport(
                        accepted=False,
                        errors=[materialization_error],
                        warnings=[],
                        checked_constraints=["mapping_candidate_materialization"],
                    )
            hypothesis = None
            error = materialization_error
            accepted = False
            if candidate is not None and report.accepted:
                try:
                    hypothesis = self.candidate_to_smm(candidate, state_cls=state_cls, machine_cls=machine_cls)
                    accepted = True
                    self.stats["accepted_candidates"] += 1
                    self.stats["candidate_states"] = len(candidate.get("states", []) or [])
                    self.stats["candidate_transitions"] = len(candidate.get("transitions", []) or [])
                except Exception as exc:  # pragma: no cover - defensive conversion path
                    error = f"candidate_to_smm_failed:{exc}"
                    report.accepted = False
                    report.errors.append(error)
                    self.stats["validation_failures"] += 1

            if not accepted:
                self.stats["rejected_candidates"] += 1
                error = error or ("validation_failed" if candidate is not None else "parse_failed")

            result = ModelerRoundResult(
                prompt_package=prompt_package,
                raw_llm_output=raw_llm_output,
                parsed_candidate=candidate,
                validation_report=report,
                hypothesis=hypothesis,
                accepted=accepted,
                error=error,
                elapsed_time_sec=perf_counter() - started,
            )
            result.artifacts = self.save_round_artifacts(result) if self.config.save_artifacts else {}
            last_result = result
            if accepted:
                return result
            if attempt_index + 1 < attempts:
                current_prompt = self._build_validation_feedback_prompt(
                    base_prompt=prompt_package.prompt,
                    report=report,
                    raw_llm_output=raw_llm_output,
                    parsed_candidate=parsed_candidate,
                    materialized_candidate=candidate,
                    attempt_index=attempt_index,
                )

        assert last_result is not None
        if self.config.raise_on_failure:
            raw_path = last_result.artifacts.get("raw_llm_output") if last_result.artifacts else None
            raise RuntimeError(
                "LLMModeler failed to synthesize an accepted candidate. "
                f"Error: {last_result.error}. Raw output artifact: {raw_path or '<not saved>'}"
            )
        return last_result

    # ------------------------------------------------------------------
    # Frontier-resolution synthesis
    # ------------------------------------------------------------------
    @staticmethod
    def frontier_resolution_response_schema() -> dict[str, Any]:
        merge_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "frontier_node_id": {"type": "integer"},
                "basis_node_id": {"type": "integer"},
                "evidence_node_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
                "evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "risks": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["frontier_node_id", "basis_node_id", "evidence_node_ids", "evidence", "risks"],
        }
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "frontier_merges": {
                    "type": "array",
                    "items": merge_schema,
                }
            },
            "required": ["frontier_merges"],
        }

    def build_frontier_resolution_prompt(
        self,
        *,
        tree_snapshot: Mapping[str, Any],
        modeler_context: Mapping[str, Any],
        feedback: Mapping[str, Any] | None = None,
    ) -> str:
        payload = {
            "observation_tree": tree_snapshot,
            "identified_aliases_for_reasoning_only": self._identified_aliases_for_prompt(modeler_context),
            "unresolved_frontier_constraints": self._as_list(modeler_context.get("unresolved_frontier_constraints")),
            "apart_node_pairs": self._as_list(modeler_context.get("apart_node_pairs")),
            "feedback": dict(feedback or {}),
        }
        sections = [
            "You are a Frontier Resolver for stochastic L+ active learning.",
            "Your only task is to propose state-merge mappings for unresolved frontier nodes after the symbolic learner reaches its sampling budget.",
            "",
            "Decision objective:",
            "1. Use the complete observation tree as evidence.",
            "2. Treat identified frontier-basis pairs as current equivalence aliases when comparing successor observations; symbolic code will freeze them.",
            "3. Return a merge only when the tree evidence supports one legal candidate basis for an unresolved frontier.",
            "4. Omit any unresolved frontier when no legal candidate basis is sufficiently supported; symbolic code will route omitted frontiers to chaos.",
            "",
            "Hard constraints:",
            "- Return JSON with frontier_merges only; frontier_merges may be empty.",
            "- Include only unresolved frontier nodes that should be merged.",
            "- Never copy identified_aliases_for_reasoning_only into frontier_merges; they are already handled by symbolic code.",
            "- Do not output chaos decisions, action fields, states, transitions, probabilities, or a complete model.",
            "- basis_node_id must be one of that frontier's candidate_basis_ids.",
            "- If candidate_basis_ids is empty, omit that frontier.",
            "- Never merge nodes that appear in apart_node_pairs.",
            "- Do not invent node ids, basis ids, inputs, outputs, or observations.",
            "- Evidence must cite node ids from the observation tree.",
            "- Return JSON only.",
            "",
            "Required output JSON schema:",
            self._json_dumps(self.frontier_resolution_response_schema()),
            "",
            "Resolver evidence package:",
            self._json_dumps(payload),
        ]
        return "\n".join(sections)

    def parse_frontier_resolutions(self, raw_text: str) -> list[dict[str, Any]] | None:
        payload = self._extract_json_payload(raw_text)
        if not isinstance(payload, Mapping):
            self.stats["parse_failures"] += 1
            return None
        merges = payload.get("frontier_merges")
        if isinstance(merges, list):
            return [dict(item) for item in merges if isinstance(item, Mapping)]
        # Backward-compatible parsing for older tests/artifacts: accept legacy
        # frontier_resolutions only when the entries are merge mappings.
        resolutions = payload.get("frontier_resolutions")
        if isinstance(resolutions, list):
            return [dict(item) for item in resolutions if isinstance(item, Mapping)]
        if not isinstance(merges, list):
            self.stats["parse_failures"] += 1
            return None
        return None

    @staticmethod
    def _normalized_apart_pairs(modeler_context: Mapping[str, Any]) -> set[tuple[int, int]]:
        pairs = set()
        for raw_pair in LLMModeler._as_list(modeler_context.get("apart_node_pairs")):
            item = LLMModeler._as_dict(raw_pair)
            left = item.get("node_id_1")
            right = item.get("node_id_2")
            if left is None or right is None:
                values = LLMModeler._as_list(raw_pair)
                if len(values) >= 2:
                    left, right = values[0], values[1]
            try:
                pairs.add(tuple(sorted((int(left), int(right)))))
            except (TypeError, ValueError):
                continue
        return pairs

    @classmethod
    def _identified_aliases_for_prompt(cls, modeler_context: Mapping[str, Any]) -> list[dict[str, Any]]:
        aliases: list[dict[str, Any]] = []
        for item in cls._as_list(modeler_context.get("identified_frontier_basis_map")):
            entry = cls._as_dict(item)
            frontier_id = cls._coerce_int(entry.get("frontier_node_id"))
            basis_id = cls._coerce_int(entry.get("basis_node_id"))
            if frontier_id is None or basis_id is None:
                continue
            aliases.append({
                "tree_node_id": frontier_id,
                "equivalent_basis_node_id": basis_id,
                "role": "reasoning_alias_only_do_not_output",
            })
        return aliases

    @classmethod
    def _identified_alias_map(cls, modeler_context: Mapping[str, Any]) -> dict[int, int]:
        aliases: dict[int, int] = {}
        for item in cls._as_list(modeler_context.get("identified_frontier_basis_map")):
            entry = cls._as_dict(item)
            frontier_id = cls._coerce_int(entry.get("frontier_node_id"))
            basis_id = cls._coerce_int(entry.get("basis_node_id"))
            if frontier_id is None or basis_id is None:
                continue
            aliases[frontier_id] = basis_id
        return aliases

    def _filter_symbolic_alias_merges(
        self,
        resolutions: Sequence[Mapping[str, Any]] | None,
        *,
        modeler_context: Mapping[str, Any],
    ) -> tuple[list[dict[str, Any]] | None, list[str]]:
        if resolutions is None:
            return None, []
        unresolved_ids = {
            node_id
            for item in self._as_list(modeler_context.get("unresolved_frontier_constraints"))
            if (node_id := self._coerce_int(self._as_dict(item).get("frontier_node_id"))) is not None
        }
        identified_aliases = self._identified_alias_map(modeler_context)
        filtered: list[dict[str, Any]] = []
        warnings: list[str] = []
        for idx, raw_resolution in enumerate(resolutions):
            resolution = self._as_dict(raw_resolution)
            frontier_id = self._coerce_int(resolution.get("frontier_node_id"))
            basis_id = self._coerce_int(resolution.get("basis_node_id"))
            if (
                frontier_id is not None
                and frontier_id not in unresolved_ids
                and identified_aliases.get(frontier_id) == basis_id
            ):
                warnings.append(f"merge[{idx}].ignored_symbolic_identified_mapping:{frontier_id}:{basis_id}")
                continue
            filtered.append(dict(resolution))
        return filtered, warnings

    def validate_frontier_resolutions(
        self,
        resolutions: Sequence[Mapping[str, Any]] | None,
        *,
        modeler_context: Mapping[str, Any],
    ) -> ValidationReport:
        errors: list[str] = []
        warnings: list[str] = []
        checked = ["frontier_merge_schema", "legal_candidate_basis", "apartness_constraints"]

        if resolutions is None:
            return ValidationReport(False, ["frontier_merges_parse_failed"], [], checked)

        constraints = {}
        for item in self._as_list(modeler_context.get("unresolved_frontier_constraints")):
            entry = self._as_dict(item)
            node_id = self._coerce_int(entry.get("frontier_node_id"))
            if node_id is not None:
                constraints[node_id] = {
                    "candidate_basis_ids": {
                        int(candidate)
                        for candidate in self._as_list(entry.get("candidate_basis_ids"))
                        if self._coerce_int(candidate) is not None
                    },
                    "forbidden_basis_ids": {
                        int(candidate)
                        for candidate in self._as_list(entry.get("forbidden_basis_ids"))
                        if self._coerce_int(candidate) is not None
                    },
                }
        expected = set(constraints)
        apart_pairs = self._normalized_apart_pairs(modeler_context)
        seen = set()

        for idx, raw_resolution in enumerate(resolutions):
            resolution = self._as_dict(raw_resolution)
            frontier_id = self._coerce_int(resolution.get("frontier_node_id"))
            if frontier_id is None or frontier_id not in constraints:
                errors.append(f"merge[{idx}].unknown_frontier_node_id:{frontier_id!r}")
                continue
            if frontier_id in seen:
                errors.append(f"duplicate_frontier_merge:{frontier_id}")
            seen.add(frontier_id)
            candidate_ids = constraints[frontier_id]["candidate_basis_ids"]
            forbidden_ids = constraints[frontier_id]["forbidden_basis_ids"]
            action = resolution.get("action")
            if action is not None and str(action).lower() != "merge":
                errors.append(f"merge[{idx}].llm_must_not_output_chaos:{frontier_id}")
                continue
            basis_id = self._coerce_int(resolution.get("basis_node_id"))
            if basis_id is None:
                errors.append(f"merge[{idx}].requires_basis_node_id:{frontier_id}")
                continue
            if basis_id not in candidate_ids:
                errors.append(f"merge[{idx}].basis_not_in_candidate_set:{frontier_id}:{basis_id}")
            if basis_id in forbidden_ids or tuple(sorted((frontier_id, basis_id))) in apart_pairs:
                errors.append(f"merge[{idx}].violates_apartness:{frontier_id}:{basis_id}")
            if not candidate_ids:
                errors.append(f"merge[{idx}].empty_candidate_set_must_be_omitted:{frontier_id}")

        return ValidationReport(accepted=not errors, errors=errors, warnings=warnings, checked_constraints=checked)

    def materialize_frontier_merge_mappings(
        self,
        resolutions: Sequence[Mapping[str, Any]] | None,
        *,
        modeler_context: Mapping[str, Any],
    ) -> list[dict[str, Any]] | None:
        if resolutions is None:
            return None
        completed: list[dict[str, Any]] = []
        seen: set[int] = set()

        for item in self._as_list(modeler_context.get("identified_frontier_basis_map")):
            entry = self._as_dict(item)
            frontier_id = self._coerce_int(entry.get("frontier_node_id"))
            basis_id = self._coerce_int(entry.get("basis_node_id"))
            if frontier_id is None or basis_id is None:
                continue
            completed.append({
                "frontier_node_id": frontier_id,
                "action": "merge",
                "basis_node_id": basis_id,
                "evidence_node_ids": [frontier_id, basis_id],
                "evidence": ["symbolic identified frontier-basis mapping frozen for candidate construction"],
                "risks": ["symbolic_identified_mapping"],
            })

        for item in self._as_list(resolutions):
            entry = self._as_dict(item)
            frontier_id = self._coerce_int(entry.get("frontier_node_id"))
            basis_id = self._coerce_int(entry.get("basis_node_id"))
            if frontier_id is None or basis_id is None:
                continue
            seen.add(frontier_id)
            completed.append({
                "frontier_node_id": frontier_id,
                "action": "merge",
                "basis_node_id": basis_id,
                "evidence_node_ids": self._as_list(entry.get("evidence_node_ids")),
                "evidence": self._as_list(entry.get("evidence")),
                "risks": self._as_list(entry.get("risks")),
            })

        for item in self._as_list(modeler_context.get("unresolved_frontier_constraints")):
            entry = self._as_dict(item)
            frontier_id = self._coerce_int(entry.get("frontier_node_id"))
            if frontier_id is None or frontier_id in seen:
                continue
            completed.append({
                "frontier_node_id": frontier_id,
                "action": "chaos",
                "basis_node_id": None,
                "evidence_node_ids": [],
                "evidence": ["LLM omitted this unresolved frontier; defaulting to chaos because no merge evidence was provided."],
                "risks": ["defaulted_missing_resolution"],
            })
        return completed

    def complete_missing_frontier_resolutions_as_chaos(
        self,
        resolutions: Sequence[Mapping[str, Any]] | None,
        *,
        modeler_context: Mapping[str, Any],
    ) -> list[dict[str, Any]] | None:
        return self.materialize_frontier_merge_mappings(resolutions, modeler_context=modeler_context)

    @staticmethod
    def _observed_output_labels(tree: Any) -> set[str]:
        labels: set[str] = set()
        iterator = getattr(tree, "iter_nodes_with_roles", None)
        if not callable(iterator):
            return labels
        for node, _, _ in iterator(max_depth=None, max_nodes=None):
            successors = getattr(node, "successors", {}) or {}
            for out_map in successors.values():
                for output in getattr(out_map, "keys", lambda: [])():
                    labels.add(str(output))
        return labels

    def validate_hypothesis_io_labels(self, hypothesis: Any, tree: Any) -> list[str]:
        """Check only whether constructed hypothesis uses legal input/output labels."""
        errors: list[str] = []
        allowed_inputs = {str(inp) for inp in getattr(tree, "alphabet", []) or []}
        allowed_outputs = self._observed_output_labels(tree)
        allowed_outputs.add("chaos")
        states = list(getattr(hypothesis, "states", []) or [])
        for state in states:
            state_id = getattr(state, "state_id", getattr(state, "name", repr(state)))
            transitions = getattr(state, "transitions", {}) or {}
            for inp, entries in transitions.items():
                if str(inp) not in allowed_inputs:
                    errors.append(f"illegal_input_label:{state_id}:{inp!r}")
                for entry in entries or []:
                    try:
                        _, output, _ = entry
                    except (TypeError, ValueError):
                        errors.append(f"malformed_transition:{state_id}:{inp!r}:{entry!r}")
                        continue
                    if str(output) not in allowed_outputs:
                        errors.append(f"illegal_output_label:{state_id}:{inp!r}:{output!r}")
        return errors

    def synthesize_frontier_resolutions(
        self,
        *,
        tree: Any,
        tree_snapshot: Mapping[str, Any],
        modeler_context: Mapping[str, Any],
        state_cls: Any | None = None,
        machine_cls: Any | None = None,
        validate: bool = True,
    ) -> ModelerRoundResult:
        self.stats["rounds"] += 1
        started = perf_counter()
        feedback: dict[str, Any] | None = None
        last_result: ModelerRoundResult | None = None
        attempts = self.config.max_retries + 1 if validate else 1

        prompt_package = ModelerPromptPackage(
            prompt="",
            constraints={"frontier_merge_mapping": True},
            prompt_view=dict(modeler_context),
            response_schema=self.frontier_resolution_response_schema(),
            prompt_path=None,
        )

        for attempt_index in range(attempts):
            prompt = self.build_frontier_resolution_prompt(
                tree_snapshot=tree_snapshot,
                modeler_context=modeler_context,
                feedback=feedback,
            )
            prompt_package.prompt = prompt
            raw_llm_output = self.call_llm(prompt)
            merge_mappings = self.parse_frontier_resolutions(raw_llm_output)
            merge_mappings, filter_warnings = self._filter_symbolic_alias_merges(
                merge_mappings,
                modeler_context=modeler_context,
            )
            symbolic_resolutions = None
            hypothesis = None
            accepted = False
            if validate:
                report = self.validate_frontier_resolutions(merge_mappings, modeler_context=modeler_context)
                error = None if report.accepted else "frontier_merge_validation_failed"
            else:
                report = ValidationReport(
                    accepted=True,
                    errors=[],
                    warnings=["frontier_resolution_validation_skipped"],
                    checked_constraints=["llm_direct_frontier_merge_output"],
                )
                error = None
            if filter_warnings:
                report = ValidationReport(
                    accepted=report.accepted,
                    errors=report.errors,
                    warnings=report.warnings + filter_warnings,
                    checked_constraints=report.checked_constraints + ["symbolic_alias_output_filter"],
                )

            snapshot = None
            if report.accepted:
                symbolic_resolutions = self.materialize_frontier_merge_mappings(
                    merge_mappings,
                    modeler_context=modeler_context,
                )
                snapshot = tree.snapshot_frontier_resolutions()
                try:
                    tree.apply_frontier_resolutions(symbolic_resolutions)
                    hypothesis = tree.construct_hypothesis(state_cls=state_cls, machine_cls=machine_cls)
                    if not validate:
                        accepted = True
                        self.stats["accepted_candidates"] += 1
                        self.stats["candidate_states"] = len(getattr(hypothesis, "states", []) or [])
                    else:
                        io_label_errors = self.validate_hypothesis_io_labels(hypothesis, tree)
                        if not io_label_errors:
                            accepted = True
                            self.stats["accepted_candidates"] += 1
                            self.stats["candidate_states"] = len(getattr(hypothesis, "states", []) or [])
                        else:
                            validation_errors = io_label_errors
                            checked_constraints = report.checked_constraints + ["hypothesis_io_labels"]
                            report = ValidationReport(
                                accepted=False,
                                errors=validation_errors,
                                warnings=[],
                                checked_constraints=checked_constraints,
                            )
                            error = "hypothesis_io_label_validation_failed"
                            feedback = {
                                "previous_attempt_index": attempt_index,
                                "rejected_merges": merge_mappings,
                                "validation_errors": validation_errors[:80],
                                "repair_instructions": [
                                    "Revise only frontier_merges.",
                                    "Do not invent input or output labels.",
                                    "Omit a frontier when no legal candidate basis is supported; symbolic code handles chaos.",
                                ],
                            }
                            tree.restore_frontier_resolutions(snapshot)
                            hypothesis = None
                except Exception as exc:
                    report = ValidationReport(
                        accepted=False,
                        errors=[f"hypothesis_construction_failed:{exc}"],
                        warnings=[],
                        checked_constraints=report.checked_constraints + ["symbolic_hypothesis_construction"],
                    )
                    error = f"hypothesis_construction_failed:{exc}"
                    feedback = {
                        "previous_attempt_index": attempt_index,
                        "rejected_merges": merge_mappings,
                        "symbolic_error": str(exc),
                        "repair_instructions": [
                            "Revise only frontier_merges.",
                            "Omit unresolved frontiers that cannot be legally merged; symbolic code handles chaos.",
                        ],
                    }
                    if snapshot is not None:
                        tree.restore_frontier_resolutions(snapshot)

            if not accepted:
                self.stats["rejected_candidates"] += 1
                if feedback is None:
                    feedback = {
                        "previous_attempt_index": attempt_index,
                        "validation_errors": report.errors[:80],
                        "repair_instructions": [
                            "Return only legal merge mappings for unresolved frontiers.",
                            "merge basis_node_id must be in candidate_basis_ids.",
                            "Do not output chaos decisions; omit unsupported frontiers.",
                        ],
                    }

            result = ModelerRoundResult(
                prompt_package=prompt_package,
                raw_llm_output=raw_llm_output,
                parsed_candidate={
                    "frontier_merges": merge_mappings or [],
                    "symbolic_frontier_resolutions": symbolic_resolutions or [],
                },
                validation_report=report,
                hypothesis=hypothesis,
                accepted=accepted,
                error=error,
                elapsed_time_sec=perf_counter() - started,
            )
            result.artifacts = self.save_round_artifacts(result) if self.config.save_artifacts else {}
            last_result = result
            if accepted:
                return result

        assert last_result is not None
        if self.config.raise_on_failure:
            raw_path = last_result.artifacts.get("raw_llm_output") if last_result.artifacts else None
            raise RuntimeError(
                "LLMModeler failed to synthesize accepted frontier resolutions. "
                f"Error: {last_result.error}. Raw output artifact: {raw_path or '<not saved>'}"
            )
        return last_result

    def synthesize_from_tree(
        self,
        tree: Any,
        *,
        modeler_context: Mapping[str, Any] | None = None,
        state_cls: Any | None = None,
        machine_cls: Any | None = None,
    ) -> ModelerRoundResult:
        if hasattr(tree, "update_frontier"):
            tree.update_frontier()
        if not hasattr(tree, "serialize_for_llm"):
            raise TypeError("tree must provide serialize_for_llm(...).")
        tree_snapshot = tree.serialize_for_llm(max_depth=None, max_nodes=None)
        if modeler_context is None and hasattr(tree, "serialize_querier_context_for_llm"):
            modeler_context = tree.serialize_querier_context_for_llm(max_depth=None, max_nodes=None)
        return self.synthesize_candidate(
            tree_snapshot=tree_snapshot,
            modeler_context=modeler_context,
            state_cls=state_cls,
            machine_cls=machine_cls,
        )

    # ------------------------------------------------------------------
    # Artifacts and small helpers
    # ------------------------------------------------------------------
    def save_round_artifacts(self, result: ModelerRoundResult) -> dict[str, str]:
        artifact_dir = Path(self.config.artifact_dir or self._default_artifact_dir()).resolve()
        artifact_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{self.config.artifact_prefix}_{self.stats['rounds']:04d}"
        artifacts: dict[str, str] = {}

        def save_json(name: str, payload: Any) -> None:
            path = artifact_dir / f"{stem}_{name}.json"
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
            artifacts[name] = str(path)

        def save_text(name: str, text: str | None) -> None:
            path = artifact_dir / f"{stem}_{name}.txt"
            path.write_text(text or "", encoding="utf-8")
            artifacts[name] = str(path)

        save_text("raw_llm_output", result.raw_llm_output)
        save_json("validation_report", result.validation_report.to_dict())
        save_json("round_result", result.to_dict())
        if self.config.artifact_mode == "full":
            if result.parsed_candidate is not None:
                save_json("parsed_candidate", result.parsed_candidate)
            if result.prompt_package is not None:
                save_json("prompt_package", result.prompt_package.to_dict())
        return artifacts

    @staticmethod
    def _default_artifact_dir() -> Path:
        return Path(__file__).resolve().parent / "modeler_artifacts"

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
    def _coerce_int(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            text = value.strip()
            if text and text.lstrip("-").isdigit():
                return int(text)
        return None

    @classmethod
    def _node_index(cls, tree_snapshot: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
        index: dict[int, dict[str, Any]] = {}
        for raw_node in cls._as_list(tree_snapshot.get("nodes")):
            node = cls._as_dict(raw_node)
            raw_id = node.get("id")
            if isinstance(raw_id, int):
                index[raw_id] = node
        return index

    @classmethod
    def _apart_node_pairs(
        cls,
        prompt_view: Mapping[str, Any],
        modeler_context: Mapping[str, Any] | None,
    ) -> list[tuple[int, int]]:
        context = cls._as_dict(modeler_context)
        modeling_context = cls._as_dict(prompt_view.get("modeling_context"))
        raw_pairs = modeling_context.get("apart_node_pairs")
        if raw_pairs is None:
            raw_pairs = context.get("apart_node_pairs", context.get("apartness_pairs"))
        pairs: list[tuple[int, int]] = []
        for raw in cls._as_list(raw_pairs):
            if isinstance(raw, Mapping):
                left = raw.get("node_id_1", raw.get("state_id_1"))
                right = raw.get("node_id_2", raw.get("state_id_2"))
            else:
                pair = cls._as_list(raw)
                left = pair[0] if len(pair) >= 1 else None
                right = pair[1] if len(pair) >= 2 else None
            if isinstance(left, int) and isinstance(right, int):
                pairs.append((left, right))
        return pairs

    @staticmethod
    def _reject_extra_keys(
        errors: list[str],
        payload: Mapping[str, Any],
        allowed: set[str],
        location: str,
    ) -> None:
        extra = sorted(str(key) for key in payload.keys() if key not in allowed)
        if extra:
            errors.append(f"{location}.unexpected_keys:{extra}")


__all__ = [
    "LLMClientConfig",
    "BuiltinLLMModelerApiClient",
    "LLMModelerConfig",
    "ValidationReport",
    "ModelerRoundResult",
    "LLMModeler",
]
