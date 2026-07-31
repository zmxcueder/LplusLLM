from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

try:  # package import
    from .config import DEEPSEEK_MODEL, LLMClientConfig, LLM_PROVIDER, OPENAI_MODEL, OPENROUTER_MODEL
    from .LLMQuerierPromptBuilder import LLMQuerierPromptBuilder, QuerierPromptPackage
except ImportError:  # standalone / local execution
    from config import DEEPSEEK_MODEL, LLMClientConfig, LLM_PROVIDER, OPENAI_MODEL, OPENROUTER_MODEL  # type: ignore
    from LLMQuerierPromptBuilder import LLMQuerierPromptBuilder, QuerierPromptPackage  # type: ignore


SUPPORTED_PURPOSES = (
    "complete_branch",
    "distinguish_frontier",
    "refine_hypothesis",
)


class BuiltinLLMApiClient:
    """Internal OpenAI-compatible API client used by LLMQuerier."""

    def __init__(self, config: LLMClientConfig | None = None) -> None:
        self.config = config or LLMClientConfig()

    @staticmethod
    def response_schema() -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "ranked_queries": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "inputs": {"type": "array", "items": {"type": "string"}},
                            "purpose": {"type": "string", "enum": list(SUPPORTED_PURPOSES)},
                            "target_node_id": {"type": ["integer", "null"]},
                            "priority": {"type": "number"},
                            "reason": {"type": "string"},
                        },
                        "required": ["inputs", "purpose", "target_node_id", "priority", "reason"],
                    },
                }
            },
            "required": ["ranked_queries"],
        }

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
                    "You rank and, when requested, extend candidate query seeds for stochastic active model learning. "
                    "Return only a valid JSON object with a ranked_queries array."
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
                    "name": "llm_querier_ranked_queries",
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
                    "name": "llm_querier_ranked_queries",
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
        output = getattr(response, "output", None)
        if output is not None:
            chunks: list[str] = []
            try:
                for item in output:
                    for content in getattr(item, "content", []) or []:
                        value = getattr(content, "text", None)
                        if isinstance(value, str):
                            chunks.append(value)
            except Exception:
                pass
            if chunks:
                return "".join(chunks)
        if isinstance(response, Mapping):
            for key in ("output_text", "text", "content"):
                value = response.get(key)
                if isinstance(value, str):
                    return value
            return json.dumps(response, ensure_ascii=False)
        return str(response)


@dataclass(slots=True)
class LLMQuerierConfig:
    """Runtime behavior for LLMQuerier."""

    max_selected_queries: int | None = None
    per_query_default_samples: int = 1
    complete_branch_use_missing_as_samples: bool = False
    allow_fallback_symbolic: bool = False
    save_artifacts: bool = True
    artifact_mode: str = "minimal"  # input_sequences | minimal | full
    input_sequences_filename: str = "input_sequences.json"
    raise_on_unparseable_llm_output: bool = True
    artifact_dir: str | Path | None = None
    artifact_prefix: str = "querier_round"
    budget_unit: str = "episodes"  # episodes | steps
    respect_remaining_budget: bool = True
    enforce_candidate_pool_membership: bool = False
    allow_prefix_only_distinguish_frontier: bool = True
    allowed_purposes: tuple[str, ...] = SUPPORTED_PURPOSES
    api_config: LLMClientConfig = field(default_factory=LLMClientConfig)

    def __post_init__(self) -> None:
        if self.max_selected_queries is not None and self.max_selected_queries <= 0:
            raise ValueError("max_selected_queries must be positive when provided.")
        if self.per_query_default_samples <= 0:
            raise ValueError("per_query_default_samples must be positive.")
        if self.budget_unit not in {"steps", "episodes"}:
            raise ValueError("budget_unit must be 'steps' or 'episodes'.")
        if self.artifact_mode not in {"input_sequences", "minimal", "full"}:
            raise ValueError("artifact_mode must be 'input_sequences', 'minimal', or 'full'.")
        if not self.input_sequences_filename:
            raise ValueError("input_sequences_filename must not be empty.")
        self.allowed_purposes = tuple(self.allowed_purposes or SUPPORTED_PURPOSES)


@dataclass(slots=True)
class QueryProposal:
    inputs: tuple[Any, ...]
    purpose: str
    target_node_id: int | None = None
    reason: str = ""
    priority_score: float | None = None
    source: str = "llm"
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "inputs": list(self.inputs),
            "purpose": self.purpose,
            "target_node_id": self.target_node_id,
            "reason": self.reason,
            "priority_score": self.priority_score,
            "source": self.source,
            "raw": self.raw,
        }


@dataclass(slots=True)
class FilterDecision:
    accepted: bool
    rejection_reason: str | None = None
    matched_evidence: dict[str, Any] = field(default_factory=dict)
    num_samples: int = 1
    estimated_cost: int = 1

    @classmethod
    def accept(
        cls,
        *,
        matched_evidence: dict[str, Any] | None = None,
        num_samples: int = 1,
        estimated_cost: int = 1,
    ) -> "FilterDecision":
        return cls(
            accepted=True,
            matched_evidence=matched_evidence or {},
            num_samples=max(1, int(num_samples or 1)),
            estimated_cost=max(1, int(estimated_cost or 1)),
        )

    @classmethod
    def reject(cls, reason: str, *, matched_evidence: dict[str, Any] | None = None) -> "FilterDecision":
        return cls(accepted=False, rejection_reason=reason, matched_evidence=matched_evidence or {})


@dataclass(slots=True)
class QueryExecutionPlan:
    inputs: tuple[Any, ...]
    purpose: str
    target_node_id: int | None
    reason: str
    priority_score: float | None
    estimated_cost: int
    num_samples: int
    matched_evidence: dict[str, Any] = field(default_factory=dict)
    source: str = "llm"

    def to_dict(self) -> dict[str, Any]:
        return {
            "inputs": list(self.inputs),
            "purpose": self.purpose,
            "target_node_id": self.target_node_id,
            "reason": self.reason,
            "priority_score": self.priority_score,
            "estimated_cost": self.estimated_cost,
            "num_samples": self.num_samples,
            "matched_evidence": self.matched_evidence,
            "source": self.source,
        }


@dataclass(slots=True)
class QuerierRoundResult:
    prompt_package: QuerierPromptPackage | None
    raw_llm_output: str | None
    parsed_proposals: list[QueryProposal]
    accepted_plans: list[QueryExecutionPlan]
    selected_plans: list[QueryExecutionPlan]
    rejected_proposals: list[dict[str, Any]]
    executed_traces: list[tuple[Any, ...]]
    artifacts: dict[str, str] = field(default_factory=dict)
    fallback_used: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_package": self.prompt_package.to_dict() if self.prompt_package is not None else None,
            "raw_llm_output": self.raw_llm_output,
            "parsed_proposals": [p.to_dict() for p in self.parsed_proposals],
            "accepted_plans": [p.to_dict() for p in self.accepted_plans],
            "selected_plans": [p.to_dict() for p in self.selected_plans],
            "rejected_proposals": self.rejected_proposals,
            "executed_traces": [list(trace) for trace in self.executed_traces],
            "artifacts": self.artifacts,
            "fallback_used": self.fallback_used,
            "error": self.error,
        }


class LLMQuerier:
    """LLM-guided query planner/executor for stochastic L+LLM."""

    def __init__(
        self,
        prompt_builder: LLMQuerierPromptBuilder | None = None,
        config: LLMQuerierConfig | None = None,
        llm_client: Any | None = None,
    ) -> None:
        self.prompt_builder = prompt_builder or LLMQuerierPromptBuilder()
        self.config = config or LLMQuerierConfig()
        self.llm_client = llm_client or BuiltinLLMApiClient(self.config.api_config)
        self.stats: dict[str, int] = {
            "rounds": 0,
            "llm_calls": 0,
            "parse_failures": 0,
            "fallbacks": 0,
            "accepted_plans": 0,
            "rejected_proposals": 0,
            "executed_plans": 0,
            "executed_traces": 0,
        }

    def build_prompt_package(
        self,
        tree_snapshot: Mapping[str, Any],
        querier_context: Mapping[str, Any],
        *,
        remaining_sampling_budget: int | None = None,
        save_prompt: bool = True,
        prompt_path: str | Path | None = None,
    ) -> QuerierPromptPackage:
        if hasattr(self.prompt_builder, "constraints"):
            self.prompt_builder.constraints.remaining_sampling_budget = remaining_sampling_budget
            if hasattr(self.prompt_builder.constraints, "top_k"):
                self.prompt_builder.constraints.top_k = self.config.max_selected_queries
            # Backward-compatible learner integration: older StochasticLSharp
            # versions instantiate the PromptBuilder with query_mode='rank_only'.
            # For the current LLMQuerier design, candidate_pool entries are
            # seed prefixes that the LLM should extend heuristically.
            if getattr(self.prompt_builder.constraints, "query_mode", None) == "rank_only":
                self.prompt_builder.constraints.query_mode = "extend_and_rank"
        return self.prompt_builder.build_prompt_package(
            tree_snapshot,
            querier_context,
            save_prompt=save_prompt,
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

    def parse_llm_response(self, raw_text: str) -> list[QueryProposal]:
        payload = self._extract_json_payload(raw_text)
        if payload is None:
            self.stats["parse_failures"] += 1
            return []
        raw_queries = payload.get("ranked_queries") if isinstance(payload, Mapping) else None
        if raw_queries is None and isinstance(payload, list):
            raw_queries = payload
        if not isinstance(raw_queries, list):
            self.stats["parse_failures"] += 1
            return []
        proposals: list[QueryProposal] = []
        for raw in raw_queries:
            proposal = self._coerce_raw_query(raw, source="llm")
            if proposal is not None:
                proposals.append(proposal)
        return proposals

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
        expanded_text = cls._expand_javascript_array_fill(text)
        if expanded_text != text:
            try:
                return json.loads(expanded_text)
            except json.JSONDecodeError:
                pass
        for open_ch, close_ch in (("{", "}"), ("[", "]")):
            start = text.find(open_ch)
            end = text.rfind(close_ch)
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    continue
        return None

    @staticmethod
    def _expand_javascript_array_fill(text: str) -> str:
        pattern = re.compile(
            r'("inputs"\s*:\s*)'
            r'(\[[^\[\]]*?\])'
            r'\s*\+\s*Array\(\s*(\d+)\s*\)\.fill\(\s*("[^"]*"|\'[^\']*\')\s*\)'
        )

        def replace(match: re.Match) -> str:
            prefix, raw_items, raw_count, raw_fill = match.groups()
            try:
                count = int(raw_count)
                if count < 0 or count > 1000:
                    return match.group(0)
                items = json.loads(raw_items)
                fill_value = json.loads(raw_fill.replace("'", '"'))
                if not isinstance(items, list):
                    return match.group(0)
            except Exception:
                return match.group(0)
            return prefix + json.dumps(items + [fill_value] * count, ensure_ascii=False)

        return pattern.sub(replace, text)

    @staticmethod
    def _coerce_raw_query(raw: Any, *, source: str) -> QueryProposal | None:
        if not isinstance(raw, Mapping):
            return None
        raw_inputs = raw.get("inputs") or raw.get("input_sequence") or raw.get("query")
        if isinstance(raw_inputs, str):
            inputs = (raw_inputs,)
        elif isinstance(raw_inputs, Sequence):
            inputs = tuple(raw_inputs)
        else:
            return None
        purpose = str(raw.get("purpose") or "").strip()
        if not purpose:
            return None
        raw_target = raw.get("target_node_id", raw.get("node_id", None))
        try:
            target_node_id = int(raw_target) if raw_target not in (None, "") else None
        except (TypeError, ValueError):
            target_node_id = None
        raw_priority = raw.get("priority", raw.get("priority_score", None))
        try:
            priority_score = float(raw_priority) if raw_priority is not None else None
        except (TypeError, ValueError):
            priority_score = None
        return QueryProposal(
            inputs=inputs,
            purpose=purpose,
            target_node_id=target_node_id,
            reason=str(raw.get("reason") or ""),
            priority_score=priority_score,
            source=source,
            raw=dict(raw),
        )

    def filter_ranked_queries(
        self,
        proposals: Sequence[QueryProposal],
        *,
        tree_snapshot: Mapping[str, Any],
        querier_context: Mapping[str, Any],
        candidate_pool: Sequence[Mapping[str, Any]] | None = None,
    ) -> tuple[list[QueryExecutionPlan], list[dict[str, Any]]]:
        alphabet_list = self._alphabet(tree_snapshot, querier_context)
        alphabet = set(alphabet_list)
        max_query_len = self._max_query_len()
        candidate_pool = list(candidate_pool or [])
        accepted: list[QueryExecutionPlan] = []
        rejected: list[dict[str, Any]] = []
        seen: set[tuple[tuple[Any, ...], str, int | None]] = set()

        for raw_proposal in proposals:
            proposal = self._extend_proposal_to_limit(
                raw_proposal,
                alphabet=alphabet_list,
                max_query_len=max_query_len,
                candidate_pool=candidate_pool,
            )
            key = (proposal.inputs, proposal.purpose, proposal.target_node_id)
            if key in seen:
                rejected.append(self._rejection_record(proposal, "duplicate_proposal"))
                continue
            seen.add(key)

            common = self._validate_common(proposal, alphabet=alphabet, max_query_len=max_query_len)
            if not common.accepted:
                rejected.append(self._rejection_record(proposal, common.rejection_reason or "common_validation_failed"))
                continue

            if self.config.enforce_candidate_pool_membership and not self._in_candidate_pool(proposal, candidate_pool):
                rejected.append(self._rejection_record(proposal, "not_in_candidate_pool"))
                continue

            decision = self._validate_by_purpose(proposal, tree_snapshot=tree_snapshot, querier_context=querier_context)
            if not decision.accepted:
                rejected.append(self._rejection_record(proposal, decision.rejection_reason or "purpose_validation_failed"))
                continue

            accepted.append(
                QueryExecutionPlan(
                    inputs=proposal.inputs,
                    purpose=proposal.purpose,
                    target_node_id=proposal.target_node_id,
                    reason=proposal.reason,
                    priority_score=proposal.priority_score,
                    estimated_cost=decision.estimated_cost,
                    num_samples=decision.num_samples,
                    matched_evidence=decision.matched_evidence,
                    source=proposal.source,
                )
            )
        accepted.sort(key=self._plan_sort_key)
        self.stats["accepted_plans"] += len(accepted)
        self.stats["rejected_proposals"] += len(rejected)
        return accepted, rejected

    def _extend_proposal_to_limit(
        self,
        proposal: QueryProposal,
        *,
        alphabet: Sequence[Any],
        max_query_len: int,
        candidate_pool: Sequence[Mapping[str, Any]],
    ) -> QueryProposal:
        """Complete short LLM proposals to max_query_length.

        The LLM prompt asks for heuristic extension, but provider outputs can
        still be short. Treat the returned sequence as a required prefix and
        append alphabet symbols locally so learner-side execution honors the
        configured query length.
        """
        inputs = tuple(proposal.inputs)
        if not inputs or len(inputs) >= max_query_len or not alphabet:
            return proposal

        required_prefix = inputs
        for raw in candidate_pool:
            if raw.get("purpose") != proposal.purpose:
                continue
            raw_target = raw.get("target_node_id")
            try:
                raw_target = int(raw_target) if raw_target is not None else None
            except (TypeError, ValueError):
                raw_target = None
            if raw_target != proposal.target_node_id:
                continue
            candidate_prefix = tuple(self._as_list(raw.get("required_prefix") or raw.get("inputs")))
            if candidate_prefix and self._starts_with(candidate_prefix, inputs):
                required_prefix = candidate_prefix
                break

        extension = self._heuristic_extension(required_prefix, alphabet, max_query_len - len(required_prefix))
        extended = tuple(required_prefix) + tuple(extension)
        if extended == proposal.inputs:
            return proposal
        raw_payload = dict(proposal.raw or {})
        raw_payload.setdefault("original_inputs", list(proposal.inputs))
        raw_payload["inputs"] = list(extended)
        return QueryProposal(
            inputs=extended,
            purpose=proposal.purpose,
            target_node_id=proposal.target_node_id,
            reason=(proposal.reason + " [locally extended to max_query_length]").strip(),
            priority_score=proposal.priority_score,
            source=proposal.source,
            raw=raw_payload,
        )

    @staticmethod
    def _heuristic_extension(prefix: Sequence[Any], alphabet: Sequence[Any], length: int) -> list[Any]:
        if length <= 0 or not alphabet:
            return []
        ordered = list(alphabet)
        if prefix and prefix[-1] in ordered:
            start = (ordered.index(prefix[-1]) + 1) % len(ordered)
        else:
            start = 0
        return [ordered[(start + i) % len(ordered)] for i in range(length)]

    def _validate_common(self, proposal: QueryProposal, *, alphabet: set[Any], max_query_len: int) -> FilterDecision:
        if not proposal.inputs:
            return FilterDecision.reject("empty_inputs")
        if len(proposal.inputs) > max_query_len:
            return FilterDecision.reject("query_too_long")
        invalid = [sym for sym in proposal.inputs if sym not in alphabet]
        if invalid:
            return FilterDecision.reject(f"symbol_out_of_alphabet:{invalid!r}")
        if proposal.purpose not in self.config.allowed_purposes:
            return FilterDecision.reject(f"purpose_not_allowed:{proposal.purpose}")
        return FilterDecision.accept()

    def _validate_by_purpose(
        self,
        proposal: QueryProposal,
        *,
        tree_snapshot: Mapping[str, Any],
        querier_context: Mapping[str, Any],
    ) -> FilterDecision:
        if proposal.purpose == "complete_branch":
            return self._validate_complete_branch(proposal, tree_snapshot, querier_context)
        if proposal.purpose == "distinguish_frontier":
            return self._validate_distinguish_frontier(proposal, tree_snapshot, querier_context)
        if proposal.purpose == "refine_hypothesis":
            return self._validate_refine_hypothesis(proposal, tree_snapshot, querier_context)
        return FilterDecision.reject(f"unknown_purpose:{proposal.purpose}")

    def _validate_complete_branch(
        self,
        proposal: QueryProposal,
        tree_snapshot: Mapping[str, Any],
        querier_context: Mapping[str, Any],
    ) -> FilterDecision:
        node_by_id = self._node_index(tree_snapshot)
        matches: list[tuple[dict[str, Any], dict[str, Any], tuple[Any, ...]]] = []
        for item in self._iter_incomplete_items(querier_context):
            node_id = item.get("node_id")
            if proposal.target_node_id is not None and node_id != proposal.target_node_id:
                continue
            node = node_by_id.get(node_id)
            if node is None:
                continue
            access = tuple(self._as_list(node.get("access_sequence")))
            suffix = tuple(x for x in self._as_list(item.get("input_suffix") or [item.get("input")]) if x is not None)
            expected_prefix = access + suffix
            if expected_prefix and self._starts_with(proposal.inputs, expected_prefix):
                matches.append((item, node, expected_prefix))
        if not matches:
            return FilterDecision.reject("complete_branch_not_aligned_with_incomplete_continuation")
        item, node, expected_prefix = max(matches, key=lambda x: int(x[0].get("missing") or 0))
        missing = int(item.get("missing") or 0)
        num_samples = max(1, missing) if self.config.complete_branch_use_missing_as_samples else self.config.per_query_default_samples
        return FilterDecision.accept(
            matched_evidence={
                "kind": "incomplete",
                "node_id": item.get("node_id"),
                "expected_prefix": list(expected_prefix),
                "missing": missing,
                "access_sequence": self._as_list(node.get("access_sequence")),
            },
            num_samples=num_samples,
            estimated_cost=self._estimate_cost(proposal.inputs, num_samples),
        )

    def _validate_distinguish_frontier(
        self,
        proposal: QueryProposal,
        tree_snapshot: Mapping[str, Any],
        querier_context: Mapping[str, Any],
    ) -> FilterDecision:
        unresolved = self._unresolved_frontier_index(querier_context)
        frontier_nodes = self._frontier_node_index(tree_snapshot)
        candidate_items: list[dict[str, Any]] = []
        if proposal.target_node_id is not None:
            if proposal.target_node_id in unresolved:
                candidate_items = [unresolved[proposal.target_node_id]]
            elif proposal.target_node_id in frontier_nodes:
                node = frontier_nodes[proposal.target_node_id]
                candidate_items = [
                    {
                        "node_id": proposal.target_node_id,
                        "role": "frontier",
                        "access_sequence": self._as_list(node.get("access_sequence")),
                        "candidate_basis_ids": self._as_list(node.get("frontier_candidate_basis_ids")),
                        "status": "frontier",
                    }
                ]
            else:
                return FilterDecision.reject("distinguish_frontier_target_not_frontier")
        else:
            candidate_items = list(unresolved.values())
            for node_id, node in frontier_nodes.items():
                if node_id not in unresolved:
                    candidate_items.append(
                        {
                            "node_id": node_id,
                            "role": "frontier",
                            "access_sequence": self._as_list(node.get("access_sequence")),
                            "candidate_basis_ids": self._as_list(node.get("frontier_candidate_basis_ids")),
                            "status": "frontier",
                        }
                    )
        for item in candidate_items:
            node_id = item.get("node_id")
            prefix = tuple(self._as_list(item.get("access_sequence")))
            if not prefix:
                continue
            if not self._starts_with(proposal.inputs, prefix):
                continue
            if len(proposal.inputs) == len(prefix) and not self.config.allow_prefix_only_distinguish_frontier:
                return FilterDecision.reject("distinguish_frontier_missing_extension_after_frontier_prefix")
            suffix = proposal.inputs[len(prefix) :]
            witness_match = self._find_witness_suffix_match(suffix, item, querier_context)
            return FilterDecision.accept(
                matched_evidence={
                    "kind": "frontier_directed_query",
                    "node_id": node_id,
                    "required_prefix": list(prefix),
                    "extension": list(suffix),
                    "candidate_basis_ids": self._as_list(item.get("candidate_basis_ids")),
                    "frontier_status": item.get("status"),
                    "witness_match": witness_match,
                },
                num_samples=self.config.per_query_default_samples,
                estimated_cost=self._estimate_cost(proposal.inputs, self.config.per_query_default_samples),
            )
        return FilterDecision.reject("distinguish_frontier_missing_frontier_access_prefix")

    def _validate_refine_hypothesis(
        self,
        proposal: QueryProposal,
        tree_snapshot: Mapping[str, Any],
        querier_context: Mapping[str, Any],
    ) -> FilterDecision:
        matched: dict[str, Any] = {"kind": "hypothesis_refinement"}
        if proposal.target_node_id is not None:
            node = self._node_index(tree_snapshot).get(proposal.target_node_id)
            if node is None:
                return FilterDecision.reject("refine_hypothesis_target_node_not_found")
            matched.update(
                {
                    "node_id": proposal.target_node_id,
                    "node_role": node.get("role"),
                    "access_sequence": self._as_list(node.get("access_sequence")),
                }
            )
        return FilterDecision.accept(
            matched_evidence=matched,
            num_samples=self.config.per_query_default_samples,
            estimated_cost=self._estimate_cost(proposal.inputs, self.config.per_query_default_samples),
        )

    def select_execution_batch(
        self,
        accepted_plans: Sequence[QueryExecutionPlan],
        *,
        remaining_sampling_budget: int | None = None,
    ) -> list[QueryExecutionPlan]:
        """Select all accepted plans unless an explicit budget/batch cap is set.

        With the current learner-side semantics, one sampled input sequence
        consumes one episode of sampling budget. Therefore the default budget
        unit is ``episodes`` and ``max_selected_queries=None`` means
        that LLMQuerier does not truncate the ranked list by count.
        """
        selected: list[QueryExecutionPlan] = []
        consumed = 0
        for plan in accepted_plans:
            if self.config.respect_remaining_budget and remaining_sampling_budget is not None:
                if consumed + plan.estimated_cost > remaining_sampling_budget:
                    continue
            selected.append(plan)
            consumed += plan.estimated_cost
            if self.config.max_selected_queries is not None and len(selected) >= self.config.max_selected_queries:
                break
        return selected

    def execute_query_batch(self, tree: Any, plans: Sequence[QueryExecutionPlan]) -> list[tuple[Any, ...]]:
        traces: list[tuple[Any, ...]] = []
        for plan in plans:
            traces.extend(self.execute_plan(tree, plan))
            self.stats["executed_plans"] += 1
        self.stats["executed_traces"] += len(traces)
        return traces

    def execute_plan(self, tree: Any, plan: QueryExecutionPlan) -> list[tuple[Any, ...]]:
        if hasattr(tree, "execute_odq") and callable(tree.execute_odq):
            return self._normalize_executed_traces(tree.execute_odq(plan.inputs, plan.num_samples))
        if hasattr(tree, "sample_query") and callable(tree.sample_query):
            return self._normalize_executed_traces(tree.sample_query(plan.inputs, plan.num_samples))
        if hasattr(tree, "_sample_query_once") and callable(tree._sample_query_once):
            return [tuple(tree._sample_query_once(plan.inputs)) for _ in range(plan.num_samples)]
        return [self._execute_with_sul_and_insert(tree, plan.inputs) for _ in range(plan.num_samples)]

    @staticmethod
    def _normalize_executed_traces(result: Any) -> list[tuple[Any, ...]]:
        if result is None:
            return []
        if isinstance(result, tuple):
            return [result]
        if isinstance(result, list):
            traces = []
            for item in result:
                if isinstance(item, tuple):
                    traces.append(item)
                elif isinstance(item, list):
                    traces.append(tuple(item))
            return traces
        return []

    @staticmethod
    def _execute_with_sul_and_insert(tree: Any, inputs: Sequence[Any]) -> tuple[Any, ...]:
        sul = getattr(tree, "sul", None)
        if sul is None:
            raise ValueError("Cannot execute query: tree has no SUL and no executable query method.")
        observed: list[Any] = []
        sul.pre()
        try:
            for inp in inputs:
                out = sul.step(inp)
                observed.extend([inp, out])
        finally:
            sul.post()
        io_trace = tuple(observed)
        if io_trace and hasattr(tree, "insert_trace"):
            tree.insert_trace(io_trace)
        return io_trace

    def plan_next_queries(
        self,
        *,
        tree_snapshot: Mapping[str, Any],
        querier_context: Mapping[str, Any],
        remaining_sampling_budget: int | None = None,
        prompt_package: QuerierPromptPackage | None = None,
    ) -> QuerierRoundResult:
        self.stats["rounds"] += 1
        if prompt_package is None:
            prompt_package = self.build_prompt_package(
                tree_snapshot,
                querier_context,
                remaining_sampling_budget=remaining_sampling_budget,
                save_prompt=self.config.save_artifacts,
            )

        raw_llm_output = self.call_llm(prompt_package.prompt)
        proposals = self.parse_llm_response(raw_llm_output)
        fallback_used = False
        if not proposals and self.config.allow_fallback_symbolic:
            fallback_used = True
            self.stats["fallbacks"] += 1
            proposals = self.fallback_proposals_from_candidate_pool(prompt_package.candidate_pool)

        if not proposals and self.config.raise_on_unparseable_llm_output:
            result = QuerierRoundResult(
                prompt_package=prompt_package,
                raw_llm_output=raw_llm_output,
                parsed_proposals=[],
                accepted_plans=[],
                selected_plans=[],
                rejected_proposals=[],
                executed_traces=[],
                fallback_used=fallback_used,
                error="no_parseable_ranked_queries",
            )
            if self.config.save_artifacts:
                result.artifacts = self.save_round_artifacts(result)
            raw_path = result.artifacts.get("raw_llm_output") if result.artifacts else None
            raise RuntimeError(
                "LLM returned no parseable ranked_queries. "
                "The response is likely empty, non-JSON, or truncated. "
                f"Raw output artifact: {raw_path or '<not saved>'}"
            )

        accepted, rejected = self.filter_ranked_queries(
            proposals,
            tree_snapshot=tree_snapshot,
            querier_context=querier_context,
            candidate_pool=prompt_package.candidate_pool,
        )
        selected = self.select_execution_batch(accepted, remaining_sampling_budget=remaining_sampling_budget)
        result = QuerierRoundResult(
            prompt_package=prompt_package,
            raw_llm_output=raw_llm_output,
            parsed_proposals=proposals,
            accepted_plans=accepted,
            selected_plans=selected,
            rejected_proposals=rejected,
            executed_traces=[],
            fallback_used=fallback_used,
            error=None,
        )
        result.artifacts = self.save_round_artifacts(result) if self.config.save_artifacts else {}
        return result

    def run_one_round(
        self,
        *,
        tree: Any,
        tree_snapshot: Mapping[str, Any],
        querier_context: Mapping[str, Any],
        remaining_sampling_budget: int | None = None,
        prompt_package: QuerierPromptPackage | None = None,
        execute: bool = True,
    ) -> QuerierRoundResult:
        result = self.plan_next_queries(
            tree_snapshot=tree_snapshot,
            querier_context=querier_context,
            remaining_sampling_budget=remaining_sampling_budget,
            prompt_package=prompt_package,
        )
        if execute and result.selected_plans:
            result.executed_traces = self.execute_query_batch(tree, result.selected_plans)
            if self.config.save_artifacts:
                result.artifacts.update(self.save_round_artifacts(result))
        return result

    def fallback_proposals_from_candidate_pool(self, candidate_pool: Sequence[Mapping[str, Any]]) -> list[QueryProposal]:
        proposals: list[QueryProposal] = []
        fallback_pool = (
            list(candidate_pool)
            if self.config.max_selected_queries is None
            else list(candidate_pool[: self.config.max_selected_queries * 3])
        )
        for index, raw in enumerate(fallback_pool):
            proposal = self._coerce_raw_query(raw, source="symbolic_fallback")
            if proposal is None:
                continue
            if proposal.priority_score is None:
                proposal.priority_score = max(0.0, 1.0 - index * 0.01)
            proposals.append(proposal)
        return proposals

    def build_input_sequences_payload(self, result: QuerierRoundResult) -> dict[str, Any]:
        """Return only input sequences for direct learner-side sampling."""
        return {"input_sequences": [list(plan.inputs) for plan in result.selected_plans]}

    def save_round_artifacts(self, result: QuerierRoundResult) -> dict[str, str]:
        artifact_dir = Path(self.config.artifact_dir or self._default_artifact_dir()).resolve()
        artifact_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{self.config.artifact_prefix}_{self.stats['rounds']:04d}"
        artifacts: dict[str, str] = {}

        def save_json_path(key: str, path: Path, payload: Any) -> None:
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
            artifacts[key] = str(path)

        def save_json(name: str, payload: Any) -> None:
            save_json_path(name, artifact_dir / f"{stem}_{name}.json", payload)

        def save_text(name: str, text: str | None) -> None:
            path = artifact_dir / f"{stem}_{name}.txt"
            path.write_text(text or "", encoding="utf-8")
            artifacts[name] = str(path)

        # Learner-facing final query table. Prefer the run-specific artifact
        # directory so parallel LLMQuerier runs do not overwrite each other.
        # Fall back to the historical module-local path when no artifact
        # directory is configured.
        if self.config.artifact_dir is not None:
            input_sequences_path = artifact_dir / self.config.input_sequences_filename
        else:
            input_sequences_path = Path(__file__).resolve().parent / self.config.input_sequences_filename
        save_json_path(
            "input_sequences",
            input_sequences_path,
            self.build_input_sequences_payload(result),
        )

        if self.config.artifact_mode in {"minimal", "full"}:
            save_text("raw_llm_output", result.raw_llm_output)
            save_json("round_result", result.to_dict())

        if self.config.artifact_mode == "full":
            save_json("parsed_proposals", [p.to_dict() for p in result.parsed_proposals])
            save_json("accepted_plans", [p.to_dict() for p in result.accepted_plans])
            save_json("selected_plans", [p.to_dict() for p in result.selected_plans])
            save_json("rejected_proposals", result.rejected_proposals)
            save_json("executed_traces", [list(t) for t in result.executed_traces])

        return artifacts

    @staticmethod
    def _default_artifact_dir() -> Path:
        return Path(__file__).resolve().parent / "querier_artifacts"

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

    @classmethod
    def _alphabet(cls, tree_snapshot: Mapping[str, Any], querier_context: Mapping[str, Any]) -> list[Any]:
        tree_meta = cls._as_dict(tree_snapshot.get("meta"))
        context_meta = cls._as_dict(querier_context.get("meta"))
        return list(cls._as_list(tree_meta.get("alphabet") or context_meta.get("alphabet") or []))

    @classmethod
    def _node_index(cls, tree_snapshot: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
        result: dict[int, dict[str, Any]] = {}
        for raw in cls._as_list(tree_snapshot.get("nodes")):
            node = cls._as_dict(raw)
            try:
                result[int(node.get("id"))] = node
            except (TypeError, ValueError):
                continue
        return result

    @classmethod
    def _frontier_node_index(cls, tree_snapshot: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
        return {node_id: node for node_id, node in cls._node_index(tree_snapshot).items() if node.get("role") == "frontier"}

    @classmethod
    def _unresolved_frontier_index(cls, querier_context: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
        result: dict[int, dict[str, Any]] = {}
        for raw in cls._as_list(querier_context.get("unresolved_frontier")):
            item = cls._as_dict(raw)
            try:
                result[int(item.get("node_id"))] = dict(item)
            except (TypeError, ValueError):
                continue
        return result

    @classmethod
    def _iter_incomplete_items(cls, querier_context: Mapping[str, Any]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for raw in cls._as_list(querier_context.get("incomplete")):
            item = cls._as_dict(raw)
            try:
                copied = dict(item)
                copied["node_id"] = int(copied.get("node_id"))
            except (TypeError, ValueError):
                continue
            result.append(copied)
        return result

    def _max_query_len(self) -> int:
        constraints = getattr(self.prompt_builder, "constraints", None)
        value = getattr(constraints, "max_query_length", None)
        try:
            return int(value)
        except (TypeError, ValueError):
            return 1

    @staticmethod
    def _starts_with(seq: Sequence[Any], prefix: Sequence[Any]) -> bool:
        if len(seq) < len(prefix):
            return False
        return tuple(seq[: len(prefix)]) == tuple(prefix)

    def _estimate_cost(self, inputs: Sequence[Any], num_samples: int) -> int:
        if self.config.budget_unit == "episodes":
            return max(1, int(num_samples or 1))
        return max(1, int(num_samples or 1) * len(tuple(inputs)))

    def _plan_sort_key(self, plan: QueryExecutionPlan) -> tuple[Any, ...]:
        purpose_rank = {"distinguish_frontier": 0, "complete_branch": 1, "refine_hypothesis": 2}.get(plan.purpose, 9)
        priority = plan.priority_score if plan.priority_score is not None else 0.0
        return (purpose_rank, -priority, -len(plan.inputs), plan.estimated_cost, repr(plan.inputs))

    @staticmethod
    def _rejection_record(proposal: QueryProposal, reason: str) -> dict[str, Any]:
        data = proposal.to_dict()
        data["rejection_reason"] = reason
        return data

    @staticmethod
    def _in_candidate_pool(proposal: QueryProposal, candidate_pool: Sequence[Mapping[str, Any]]) -> bool:
        p_inputs = tuple(proposal.inputs)
        for raw in candidate_pool:
            c_inputs = tuple(raw.get("inputs") or [])
            c_purpose = raw.get("purpose")
            c_target = raw.get("target_node_id")
            try:
                c_target = int(c_target) if c_target is not None else None
            except (TypeError, ValueError):
                c_target = None
            if p_inputs == c_inputs and proposal.purpose == c_purpose and proposal.target_node_id == c_target:
                return True
        return False

    def _find_witness_suffix_match(
        self,
        suffix: Sequence[Any],
        frontier_item: Mapping[str, Any],
        querier_context: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        candidate_ids = []
        for raw_id in self._as_list(frontier_item.get("candidate_basis_ids")):
            try:
                candidate_ids.append(int(raw_id))
            except (TypeError, ValueError):
                pass
        suffix_tuple = tuple(suffix)
        if not candidate_ids or not suffix_tuple:
            return None
        for raw in self._as_list(querier_context.get("apartness_witness")):
            witness = self._as_dict(raw)
            try:
                s1 = int(witness.get("state_id_1"))
                s2 = int(witness.get("state_id_2"))
            except (TypeError, ValueError):
                continue
            if s1 not in candidate_ids or s2 not in candidate_ids:
                continue
            w = tuple(self._as_list(witness.get("witness")))
            if w and self._starts_with(suffix_tuple, w):
                return {"state_id_1": s1, "state_id_2": s2, "witness": list(w)}
        return None

    def snapshot_stats(self) -> dict[str, int]:
        return dict(self.stats)


__all__ = [
    "LLM_PROVIDER",
    "OPENAI_MODEL",
    "DEEPSEEK_MODEL",
    "OPENROUTER_MODEL",
    "LLMClientConfig",
    "BuiltinLLMApiClient",
    "LLMQuerierConfig",
    "QueryProposal",
    "FilterDecision",
    "QueryExecutionPlan",
    "QuerierRoundResult",
    "LLMQuerier",
]
