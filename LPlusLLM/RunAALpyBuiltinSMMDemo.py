from __future__ import annotations

from pathlib import Path
from random import seed

try:
    from .config import LLMRunConfig
    from .StochasticLSharp import (
        CountingSUL,
        StatisticalEqOracle,
        StochasticLSharpInternalConfig,
        run_stochastic_Lsharp,
    )
except ImportError:  # pragma: no cover
    from config import LLMRunConfig
    from StochasticLSharp import (
        CountingSUL,
        StatisticalEqOracle,
        StochasticLSharpInternalConfig,
        run_stochastic_Lsharp,
    )


BENCHMARK_EXAMPLES = {
    'first_grid',
    'second_grid',
    'shared_coin',
    'slot_machine',
    'faulty_car_alarm',
    'mqtt',
    'tcp',
    'bluetooth',
}


def _candidate_dot_paths(example: str):
    import aalpy

    here = Path(__file__).resolve()
    candidates = [
        here.parent / f'{example}.dot',
        Path.cwd() / f'{example}.dot',
        here.parents[3] / 'DotModels' / 'MDPs' / f'{example}.dot' if len(here.parents) >= 4 else None,
        Path(aalpy.__file__).resolve().parents[1] / 'DotModels' / 'MDPs' / f'{example}.dot',
    ]
    return [p for p in candidates if p is not None]


def _load_builtin_target(example: str = 'faulty_coffee'):
    from aalpy.utils import get_faulty_coffee_machine_MDP, load_automaton_from_file

    if example == 'faulty_coffee':
        target = get_faulty_coffee_machine_MDP()
        return target, 'get_faulty_coffee_machine_MDP()'

    if example not in BENCHMARK_EXAMPLES:
        raise ValueError(
            f"Unknown example {example!r}. Choose one of: 'faulty_coffee', "
            f"{', '.join(sorted(BENCHMARK_EXAMPLES))}."
        )

    for dot_path in _candidate_dot_paths(example):
        if dot_path.exists():
            target = load_automaton_from_file(str(dot_path), automaton_type='mdp')
            return target, str(dot_path)

    searched = '\n'.join(str(p) for p in _candidate_dot_paths(example))
    raise FileNotFoundError(
        f'Could not find built-in benchmark model for {example!r}. Searched:\n{searched}'
    )


def _print_metrics(info):
    timings = info.get('timings', {}) or {}
    query_counts = info.get('query_counts', {}) or {}
    sampling_counts = info.get('sampling_counts', {}) or {}
    learner_stats = info.get('learner_stats', {}) or {}
    phase_sul_stats = info.get('phase_sul_stats', {}) or {}
    eq_oracle_stats = info.get('eq_oracle_stats', {}) or {}
    tree_stats = info.get('tree_stats', {}) or {}

    print('\n===== Learning Metrics =====')
    print(f"Total learning time          : {timings.get('total_learning_time_sec', 0.0):.3f}s")
    print(f"Hypothesis build time        : {timings.get('hypothesis_build_time_sec', 0.0):.3f}s")
    print(f"EQ time                      : {timings.get('eq_time_sec', 0.0):.3f}s")
    print(f"Counterexample processing    : {timings.get('counterexample_processing_time_sec', 0.0):.3f}s")

    print('\n===== Query Counts =====')
    print(f"Learning rounds              : {query_counts.get('learning_rounds', learner_stats.get('nr_learning_rounds_started', 0))}")
    print(f"Hypothesis builds            : {query_counts.get('hypothesis_builds', learner_stats.get('nr_hypothesis_builds', 0))}")
    print(f"EQ calls                     : {query_counts.get('eq_calls', learner_stats.get('nr_eq_calls', 0))}")
    print(f"Counterexamples              : {query_counts.get('counterexamples', 0)}")
    print(f"Candidate sequences checked  : {query_counts.get('candidate_sequences_checked', eq_oracle_stats.get('candidate_sequences_checked', 0))}")
    print(f"LLM Querier calls            : {query_counts.get('llm_querier_calls', learner_stats.get('nr_llm_querier_calls', 0))}")
    print(f"LLM input sequences          : {query_counts.get('llm_input_sequences', learner_stats.get('nr_llm_input_sequences', 0))}")
    print(f"LLM executed traces          : {query_counts.get('llm_executed_traces', learner_stats.get('nr_llm_executed_traces', 0))}")
    print(f"LLM Modeler calls            : {query_counts.get('llm_modeler_calls', learner_stats.get('nr_llm_modeler_calls', 0))}")
    print(f"LLM Modeler accepted         : {query_counts.get('llm_modeler_accepted', learner_stats.get('nr_llm_modeler_accepted', 0))}")
    print(f"LLM Modeler rejected         : {query_counts.get('llm_modeler_rejected', learner_stats.get('nr_llm_modeler_rejected', 0))}")
    print(f"LLM Modeler budget skips     : {query_counts.get('llm_modeler_budget_skips', learner_stats.get('nr_llm_modeler_budget_skips', 0))}")
    print(f"Symbolic budget interrupts   : {query_counts.get('symbolic_budget_interrupts', learner_stats.get('nr_symbolic_budget_interrupts', 0))}")
    print(f"Total SUL episodes           : {query_counts.get('total_sul_episodes', 0)}")

    print('\n===== Sampling Counts =====')
    print(f"Total SUL steps              : {sampling_counts.get('total_sul_steps', 0)}")
    print(f"EQ sampled traces            : {sampling_counts.get('eq_sampled_traces', 0)}")
    print(f"EQ sampled steps             : {sampling_counts.get('eq_sampled_steps', 0)}")
    print(f"Completion attempts          : {sampling_counts.get('completion_attempts', 0)}")
    print(f"Completion matched samples   : {sampling_counts.get('completion_successful_samples', 0)}")
    print(f"Completion failed replays    : {sampling_counts.get('completion_failed_replays', 0)}")
    print(f"Identify frontier samples    : {sampling_counts.get('identify_frontier_samples', 0)}")
    print(f"Guided completion queries    : {sampling_counts.get('guided_completion_queries', tree_stats.get('guided_completion_queries', 0))}")
    print(f"Guided frontier queries      : {sampling_counts.get('guided_frontier_queries', tree_stats.get('guided_frontier_queries', 0))}")
    print(f"LLM executed traces          : {sampling_counts.get('llm_executed_traces', learner_stats.get('nr_llm_executed_traces', 0))}")

    print('\n===== Tree Stats =====')
    print(f"ensure_complete calls        : {tree_stats.get('ensure_complete_calls', 0)}")
    print(f"identify_frontier calls      : {tree_stats.get('identify_frontier_calls', 0)}")
    print(f"guided completion queries    : {tree_stats.get('guided_completion_queries', 0)}")
    print(f"guided frontier queries      : {tree_stats.get('guided_frontier_queries', 0)}")
    print(f"binary search calls          : {tree_stats.get('binary_search_calls', 0)}")

    print('\n===== Phase-wise SUL Usage =====')
    for phase in ('llm_querier', 'build_hypothesis', 'eq', 'counterexample_processing', 'total'):
        stats = phase_sul_stats.get(phase)
        if not stats:
            continue
        print(f"{phase:>24}: episodes={stats.get('episodes', 0)}, steps={stats.get('steps', 0)}")


def _convert_smm_to_mdp(model):
    """
    Convert an AALpy StochasticMealyMachine to an MDP using the library's
    built-in conversion helper.
    """
    if hasattr(model, 'to_mdp'):
        return model.to_mdp()

    try:
        from aalpy.automata.StochasticMealyMachine import smm_to_mdp_conversion
        return smm_to_mdp_conversion(model)
    except Exception as exc:
        raise RuntimeError(
            'Could not convert learned SMM to MDP. The installed AALpy version '
            'does not expose StochasticMealyMachine.to_mdp() or '
            'smm_to_mdp_conversion(...).'
        ) from exc


def learn_builtin_stochastic_model(
    example: str = 'faulty_coffee',
    strategy: str = 'normal',
    eq_num_samples: int = 1,
    eq_max_cex_len: int = 5,
    eq_random_search: bool = True,
    eq_random_traces: int = 150,
    eq_trace_len: int | None = 100,
    n_c: int = 2,
    n_resample: int | None = None,
    alpha: float = 0.05,
    max_rounds: int = 500,
    cex_processing: str | None = None,
    print_level: int = 2,
    random_seed: int = 7,
    output_path: str | Path = 'learned_aalpy_builtin_smm',
    debug: bool = True,
    *,
    internal_config: dict | StochasticLSharpInternalConfig | None = None,
    query_strategy: str = 'symbolic',
    model_strategy: str = 'symbolic',
    llm_config: LLMRunConfig | None = None,
):
    """
    Demo wrapper with a compact, AALpy-like parameter surface.

    Redundant aliases from the old demo are removed:
      - basis_n_c -> n_c
      - apartness_alpha / eq_alpha -> alpha
    Advanced low-level learner knobs can still be supplied through
    `internal_config`.
    """
    llm_config = llm_config or LLMRunConfig(sampling_budget_per_round=10)
    seed(random_seed)

    from aalpy.SULs import AutomatonSUL
    from aalpy.utils import save_automaton_to_file

    target, source = _load_builtin_target(example)
    alphabet = tuple(target.get_input_alphabet())

    sul = CountingSUL(AutomatonSUL(target), name='learning_sul')
    eq_oracle = StatisticalEqOracle(
        alphabet=alphabet,
        sul=sul,
        num_samples=eq_num_samples,
        max_cex_len=eq_max_cex_len,
        alpha=alpha,
        random_search=eq_random_search,
        random_trace_count=eq_random_traces,
        random_trace_len=eq_trace_len if eq_trace_len is not None else eq_max_cex_len,
        random_seed=random_seed,
        insert_samples_into_tree=True,
        debug=debug,
    )

    learned_model, info = run_stochastic_Lsharp(
        input_alphabet=alphabet,
        sul=sul,
        eq_oracle=eq_oracle,
        automaton_type='smm',
        strategy=strategy,
        cex_processing=cex_processing,
        return_data=True,
        n_c=n_c,
        n_resample=n_resample,
        print_level=print_level,
        alpha=alpha,
        max_rounds=max_rounds,
        internal_config=internal_config,
        debug=debug,
        query_strategy=query_strategy,
        model_strategy=model_strategy,
        llm_config=llm_config,
    )

    output_path = Path(output_path)
    smm_output_path = output_path
    mdp_output_path = output_path.with_name(f'{output_path.name}_as_mdp')

    save_automaton_to_file(learned_model, path=str(smm_output_path), file_type='dot')

    learned_mdp_model = _convert_smm_to_mdp(learned_model)
    save_automaton_to_file(learned_mdp_model, path=str(mdp_output_path), file_type='dot')

    api_alignment = info.get('api_alignment', {}) or {}
    resolved_internal_config = api_alignment.get('internal_config', {}) or {}
    resolved_n_resample = api_alignment.get('n_resample_as_refinement_hits', n_resample)

    print(f'Target source                : {source}')
    print(f'Input alphabet               : {alphabet}')
    print(f'Learning rounds              : {info["learning_rounds"]}')
    print(f'Counterexamples              : {info["nr_counterexamples"]}')
    print(f'Learned SMM states           : {len(getattr(learned_model, "states", []))}')
    print(f'Learned MDP states           : {len(getattr(learned_mdp_model, "states", []))}')
    print(f'strategy                     : {strategy}')
    print(f'query_strategy               : {query_strategy}')
    print(f'model_strategy               : {model_strategy}')
    print(f'llm_config                   : {llm_config}')
    print(f'n_c                          : {n_c}')
    print(f'n_resample                   : {resolved_n_resample}')
    print(f'max_rounds                   : {max_rounds}')
    print(f'cex_processing               : {cex_processing}')
    print(f'print_level                  : {print_level}')
    print(f'alpha                        : {alpha}')
    print(f'eq_num_samples               : {eq_num_samples}')
    print(f'eq_max_cex_len               : {eq_max_cex_len}')
    print(f'eq_random_search             : {eq_random_search}')
    print(f'eq_random_traces             : {eq_random_traces}')
    print(f'eq_trace_len                 : {eq_trace_len if eq_trace_len is not None else eq_max_cex_len}')
    print(f'internal_config              : {resolved_internal_config}')
    print(f'SMM DOT output               : {smm_output_path}.dot')
    print(f'MDP DOT output               : {mdp_output_path}.dot')

    _print_metrics(info)
    print('\n===== Learned SMM model (DOT) =====')
    print(learned_model)
    print('\n===== Learned MDP model (DOT) =====')
    print(learned_mdp_model)

    return learned_model, learned_mdp_model, info, target


if __name__ == '__main__':
    learn_builtin_stochastic_model()
