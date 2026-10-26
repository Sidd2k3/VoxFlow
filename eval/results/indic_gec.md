# Grammar correction in Indian languages (IndicGEC 2025 test sets)

Shared-task GLEU, higher is better. Stage 2 model `llama3.2`, intensity 50.

| System | Hindi (236) | Bengali (330) |
|---|---|---|
| No correction (source) | 80.74 | 95.79 |
| Stage 1 rules only | 81.29 | 95.85 |
| Stage 2 LLM + filter only | 82.17 | 94.21 |
| Full pipeline (as in the app) | 82.22 | 95.85 |
| Rules + unfiltered LLM | 80.62 | 90.82 |
