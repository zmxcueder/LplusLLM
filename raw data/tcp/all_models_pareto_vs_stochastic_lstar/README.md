# all_models_pareto_vs_stochastic_lstar: tcp

帕累托图展示四个 L+LLM 模型的前沿、被支配点与 Stochastic L* 基线。

## 文件说明

- `figure.png`、`figure.pdf`、`figure.svg`：该单图的三种导出格式。
- `raw_lplusllm_successful_runs.csv`：图中 L+LLM 的运行级成功记录，160 行。
- Stochastic L* 运行级数据：`raw_stochastic_lstar_selected_runs.csv`，10 行。
  规则：从每个任务的 20 次成功运行中按运行编号固定选取前 10 次。
- `plotted_summary.csv`：绘图实际使用的分组汇总或归一化结果，20 行。

## 数据来源

- L+LLM 原始来源：`Evaluation/results/four_models_experiment_summary_20260807_min_error_fixed.xlsx` 的 `Raw Runs` 工作表。
- 仅保留状态为 `ok` 且属于当前任务的记录；每个模型—预算条件均为 10 次成功运行。
