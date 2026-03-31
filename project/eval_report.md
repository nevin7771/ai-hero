# Evaluation Report

## Scope

This report summarizes Day 5 and Day 7 evaluation for the docs assistant in `project/`.

## Metrics Used

- Retrieval quality:
  - Hit@5
  - MRR
- Answer quality proxy:
  - token overlap F1 between generated answer and top retrieved chunk

## Latest Results

- Best retrieval mode: `hybrid`
- Hit@5: `0.95`
- MRR: `0.8494`
- Answer-vs-context token F1: `0.3778`
- Benchmark questions: `5`

## Pass/Fail Targets

- Hit@5 >= 0.90 -> PASS
- MRR >= 0.75 -> PASS
- Answer-vs-context token F1 >= 0.30 -> PASS

## Artifacts

- `day7_benchmark_*.csv`
- `day7_summary_*.json`
- `day7_final_share_*.json`

## Notes

- Hybrid retrieval performs best for this dataset.
- Function calling improves traceability by explicitly exposing retrieved sources.
