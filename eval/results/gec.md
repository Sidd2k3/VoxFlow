# Grammar correction (JFLEG test, 747 sentences)

GLEU, higher is better. Stage 2 model `llama3.2`. The edit filter was designed on the dev sentences; the held-out score is the unbiased one.

| System | Held-out (301-747) | Dev (1-300) | All |
|---|---|---|---|
| No correction (source) | 44.04 | 38.48 | 41.78 |
| Stage 1 rules only | 44.55 | 39.21 | 42.38 |
| Stage 2 LLM + filter only | 58.44 | 54.64 | 56.88 |
| Full pipeline (rules + LLM + filter) | 58.52 | 55.02 | 57.08 |
| Rules + unfiltered LLM | 61.79 | 58.70 | 60.51 |
