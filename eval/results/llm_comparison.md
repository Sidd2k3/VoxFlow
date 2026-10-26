# Stage 2 model comparison (development data)

Full pipeline GLEU, higher is better. English: JFLEG validation (150 sentences); Hindi/Bengali: IndicGEC 2025 dev (107/101).

| Stage 2 model | English | Hindi | Bengali | Median time per sentence |
|---|---|---|---|---|
| none (no correction) | 32.98 | 88.78 | 95.25 | – |
| llama3.2 | 47.58 | 88.13 | 93.40 | 1128 ms |
| gemma3:12b | 49.80 | 88.86 | 93.96 | 7306 ms |
