# Speech recogniser comparison (FLEURS test clips, 30 per language)

Word error rate (lower is better); character error rate in brackets.

| Engine | English | Hindi | Marathi | Bengali | Punjabi | Median time per clip |
|---|---|---|---|---|---|---|
| google | 34.6% (22.4%) | 17.5% (7.6%) | 27.0% (8.2%) | 16.7% (4.0%) | 23.7% (8.9%) | 3591 ms |
| whisper:small | 8.5% (3.8%) | 57.7% (28.8%) | 108.5% (48.5%) | 119.0% (103.1%) | 102.5% (102.6%) | 4204 ms |
| whisper:large-v3-turbo | 6.0% (2.9%) | 29.8% (11.3%) | 79.6% (25.6%) | 84.6% (39.6%) | 87.9% (46.5%) | 8898 ms |
