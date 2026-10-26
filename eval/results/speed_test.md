# Google (Chrome's speech engine) vs VoxFlow, same recordings

Median time from the end of the recording to the text; word error rate (lower is better). FLEURS test clips. VoxFlow = IndicConformer as the app runs it: Hindi indicconformer:rnnt, Marathi indicconformer:rnnt, Punjabi indicconformer:ctc.

| Language | Clips | Google time | VoxFlow time | VoxFlow faster on | Google WER | VoxFlow WER | + grammar rules | + LLM |
|---|---|---|---|---|---|---|---|---|
| Hindi | 10 | 1600 ms | 368 ms | 10/10 | 19.5% | 9.0% | 1 ms | 1966 ms |
| Marathi | 10 | 1726 ms | 351 ms | 10/10 | 25.8% | 13.7% | 0 ms | 1840 ms |
| Punjabi | 10 | 1532 ms | 220 ms | 10/10 | 25.7% | 15.2% | 0 ms | – |
