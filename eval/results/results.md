# Evaluation results (2026-10-26)

Dataset: FLEURS test split (Google). Speech recognition: English `mlx:large-v3-turbo:fast`, Hindi `indicconformer:rnnt`, Marathi `indicconformer:rnnt`, Bengali `google`, Punjabi `indicconformer:ctc`. Stage 2 model: `llama3.2`, intensity 50.

## Word error rate (lower is better)

| Language | Clips | Raw ASR | + Stage 1 | Stage 2 only | Full pipeline |
|---|---|---|---|---|---|
| English | 30 | 5.7% | 5.7% | 7.4% | 7.4% |
| Hindi | 30 | 11.1% | 11.1% | 11.7% | 11.7% |
| Marathi | 30 | 15.5% | 15.5% | 15.6% | 15.6% |
| Bengali | 30 | 16.7% | 16.7% | 17.1% | 16.7% |
| Punjabi | 30 | 15.9% | 15.8% | 16.3% | 15.8% |

## Character error rate

| Language | Raw ASR | + Stage 1 | Stage 2 only | Full pipeline |
|---|---|---|---|---|
| English | 3.1% | 3.1% | 4.1% | 4.1% |
| Hindi | 4.8% | 4.8% | 5.0% | 5.0% |
| Marathi | 3.5% | 3.5% | 3.6% | 3.6% |
| Bengali | 4.0% | 4.0% | 4.1% | 4.0% |
| Punjabi | 5.7% | 5.6% | 5.9% | 5.6% |

## Stage 2 outcomes and latency

| Language | Stage 2 outcomes | LLM edits kept / reverted | Clips with ASR alternatives | Median ASR ms | Median Stage 2 ms |
|---|---|---|---|---|---|
| English | applied 18, rejected 1, unchanged 11 | 14 / 11 | 0/30 | 984 | 737 |
| Hindi | applied 20, unchanged 10 | 7 / 8 | 0/30 | 386 | 1624 |
| Marathi | applied 23, unchanged 7 | 2 / 12 | 0/30 | 396 | 1883 |
| Bengali | skipped 30 | 0 / 0 | 26/30 | 3716 | 0 |
| Punjabi | skipped 30 | 0 / 0 | 0/30 | 264 | 0 |
