# VoxFlow

Speech recognition with grammar correction for **Hindi, Marathi, Bengali, Punjabi and English**. Pick a language, speak, and VoxFlow writes down what you said, fixes the grammar, lets you accept or reject every change, and reads the corrected text back to you.

**Why:** Google's speech recognition (the engine behind Chrome's microphone) makes many mistakes on Indian languages and doesn't fix grammar. On the same recordings, VoxFlow recognises Hindi, Marathi and Punjabi **4–7× faster than Google with 33–43% fewer errors**, recognises Indian-accented English faster with less than half the errors, and then corrects the grammar in all five languages, including several mistakes across two sentences (**18–19 of 20** such cases fixed, against 6 of 20 with rules alone; Results 12):

| Same recordings | Google: time | VoxFlow: time | Google: word errors | VoxFlow: word errors |
|---|---|---|---|---|
| Hindi | 1.6 s | **0.37 s** | 17.5% | **11.1%** |
| Marathi | 1.7 s | **0.35 s** | 27.0% | **15.5%** |
| Punjabi | 1.5 s | **0.22 s** | 23.7% | **15.9%** |
| English (Indian accent) | 1.3 s | **0.95 s** | 13.0% | **4.9%** |
| Bengali | same engine (Google, see below) | | 16.7% | 16.7% |

Time = from the end of the recording to the text (median, M4 MacBook Air). Word errors on 30 FLEURS clips per language (60 Indian-accented clips for English). Hindi, Marathi, Punjabi and English are recognised on your Mac, without internet. Every number below was measured with the scripts in `eval/`.

## How it works

```
microphone ─► WAV (16 kHz mono) ─► speech recognition ─► Stage 1: spelling + grammar rules ─► Stage 2: large LLM ─► corrections ─► Speak Back
  (you pick the language)          (on the Mac)            (shown at once, ~0 ms)           (Groq → Gemini, free)
                                      │
                      nothing heard? ─┘─► noise reduction + VAD ─► recognise again
```

| Step | What happens | Tool |
|---|---|---|
| Speech recognition | Each language uses the recogniser that measured best for it. **Hindi, Marathi → AI4Bharat IndicConformer, RNNT decoding; Punjabi → IndicConformer, CTC decoding** (600M parameters, trained on Indian-language speech, MIT licence, on your Mac). **English → Whisper large-v3-turbo** on the Mac's GPU. **Bengali → Google**: IndicConformer was more accurate on the FLEURS clips but misheard short spoken Bengali sentences in the live test, which Google heard correctly (Results 10). If IndicConformer can't load, the app falls back to Google and the Dashboard says why. | `transformers` + ONNX Runtime, `mlx-whisper`, Google Web Speech API |
| Retry | If nothing is recognised, the audio is cleaned and tried once more. | `noisereduce` (spectral gating), `webrtcvad` (voice activity detection) |
| Stage 1 | **(a) Recogniser spellings:** two vowel signs in a row (an IndicConformer glitch, "খাইছেি") are cleaned; a rare word becomes the much more common word one sound-alike letter away (বারি → বাড়ি, पानि → पानी; Bengali and Hindi, which have word-frequency lists); known recogniser spellings of key words are restored (Marathi कल → काल, शालेत → शाळेत; Punjabi ਅਸੀਨ → ਅਸੀਂ; Bengali গোটকাল → গতকাল). **(b) Grammar rules:** subject-verb agreement for pronoun and noun subjects (Bengali through a 20-verb conjugation table) (she know → she knows, ती जातो → ती जाते, আমি খেয়েছে → খেয়েছি, ਅਸੀਂ ਜਾਂਦਾ ਹੈ → ਜਾਂਦੇ ਹਾਂ), gender agreement (लड़की जाता → जाती), base form after do/does/did and modals, tense with a time word ("yesterday", "कल", "গতকাল", "ਕੱਲ੍ਹ"), the ergative marker (वह → उसने), "could of" → "could have", a/an, stutters, the full stop. Rules only fire in safe positions, so correct sentences are left alone. Takes ~0 ms, so its result is on screen right after recognition. | Python token rules, `wordfreq` |
| Stage 2 | A **large LLM** fixes what the rules can't, in all five languages: every grammar mistake in every sentence (tense, agreement, ergative ने/ਨੇ, helper verbs, postpositions), words the recogniser clearly misheard or ran together (मजाभाऊ → माझा भाऊ, ਬਹਿਣ → ਭੈਣ), and a present-tense verb that should follow the time of the sentence before it. Free tiers, asked in order: **Groq** (`gpt-oss-120b`, ~0.6 s) → **Gemini** (`gemini-3.5-flash-lite`, ~2 s) when Groq hits its limit → the local `llama3.2` (English only) without internet. Only the recognised **text** is sent, never audio. The answer is checked: wrong script / translation, or more than 80% of the words changed → rejected; the speaker's own gender (मैं जाती / मैं जाता) is never flipped. Keys go in `.env` (Setup). Without keys the app runs fully offline with Stage 1 (and the local LLM for English, behind a strict word-by-word edit filter). | Groq API, Gemini API (`app/cloud_llm.py`), Ollama |
| Corrections | Word-level diff of what was heard → final text. Each correction says whether a rule or the LLM made it; rejecting one puts the original words back. | `difflib` |
| Speak Back | Corrected text → speech in a native voice per language (Google's voice for Punjabi, where Microsoft's free endpoint has none). | `edge-tts`, `gTTS` |

**Correction intensity** (Settings): 0–33 = Stage 1 only; 34–100 (default 50) = rules + the LLM. (With the local fallback model, 80–100 also lets it replace words the recogniser was unsure it heard.)

**No Auto-detect, on purpose.** An earlier version guessed the spoken language from the audio (Whisper + Meta's MMS language ID, 98.7% right on 210 test clips). Live, the remaining mistakes all happened on short sentences, where Marathi, Hindi and Punjabi sound alike, and a wrong language means a wrong transcript. Picking the language is one click and is always right, so the guessing was removed, along with ~4 GB of models and about a second of waiting per recording.

## Website

Dashboard (live component status, benchmark results, recent sessions) · Speech Input (record with the button or `R`, language picker, typed-text fallback) · Text Output (heard vs corrected, accept/reject, copy, JSON export) · Speak Back (player with 0.5x–2x speed) · Settings · Insights panel (language, recogniser confidence, benchmark WER, time per step, correction counts).

While you wait, the status line shows the time and the rule-corrected text as soon as recognition finishes (`/api/transcribe`); the LLM's version replaces it when it's ready (`/api/text`). The backend is Python; the website is three plain files (`index.html`, `styles.css`, `app.js`) served by the same Python server. Sessions are saved in the browser.

## Results

All measured on an M4 MacBook Air with the scripts in `eval/`; output files are in `eval/results/`. Results 4–11 measure the grammar step with the earlier **local** model (`llama3.2`, 3B) or rules only; Result 12 measures the large free cloud models the app now uses. The large models were not run on the full JFLEG/IndicGEC/1000-sentence sets: those need thousands of calls, more than the free tiers allow in a day.

**1. Google vs VoxFlow, head to head** (`eval/speed_test.py`, 10 FLEURS test clips per language). Every clip goes to both, live (nothing cached), taking turns which goes first. Google is the free Google Web Speech endpoint, the same service as Chrome's microphone, with the finished clip in one request. Time is from the end of the recording to the text.

| Language | Google | VoxFlow | VoxFlow faster on | Google WER | VoxFlow WER |
|---|---|---|---|---|---|
| Hindi (IndicConformer RNNT) | 1600 ms | **368 ms** | 10/10 clips | 19.5% | **9.0%** |
| Marathi (IndicConformer RNNT) | 1726 ms | **351 ms** | 10/10 | 25.8% | **13.7%** |
| Punjabi (IndicConformer CTC) | 1532 ms | **220 ms** | 10/10 | 25.7% | **15.2%** |

Stage 1 adds ~0 ms, and it is the only grammar step for the Indian languages, so the times above are the whole wait. The LLM runs only for English (about 0.3 s on a short spoken sentence). Chrome shows words while you are still talking, which this test doesn't measure; it measures the wait after you stop.

**2. Speech recognition accuracy, Indian languages** (`eval/compare_asr.py`, 30 FLEURS test clips per language). Word error rate, character error rate in brackets, lower is better:

| Engine | Hindi | Marathi | Bengali | Punjabi |
|---|---|---|---|---|
| Google Web Speech | 17.5% (7.6%) | 27.0% (8.2%) | 16.7% (4.0%) | 23.7% (8.9%) |
| IndicConformer, CTC | 11.7% (4.5%) | 17.9% (3.8%) | **13.1% (3.3%)** | **15.9% (5.7%)** |
| IndicConformer, RNNT | **11.1% (4.8%)** | **15.5% (3.5%)** | 13.4% (3.1%) | 16.3% (5.7%) |
| Whisper large-v3-turbo (tested earlier) | 29.8% | 79.6% | 84.6% | 87.9% |

IndicConformer beats Google in every language on these clips; each language uses its better decoding mode. Whisper is not usable for these languages (above 100% would mean more wrong words than the sentence has). For Bengali, see Results 10.

**3. Indian-accented English** (`eval/compare_english.py`, 60 clips from AI4Bharat's [Svarah](https://huggingface.co/datasets/ai4bharat/Svarah), speakers of 19 native languages; "missed words" = spoken words absent from the transcript):

| Engine | WER | CER | Missed words | Median time per clip |
|---|---|---|---|---|
| Google, en-US | 13.0% | 7.3% | 3.7% | 1438 ms |
| Google, en-IN | 13.0% | 7.3% | 3.7% | 1337 ms |
| Whisper large-v3-turbo, CPU | 5.7% | 3.0% | 0.2% | 5446 ms |
| Whisper large-v3, GPU | **4.1%** | **2.3%** | 0.4% | 3247 ms |
| Whisper large-v3-turbo, GPU | 4.6% | 2.5% | 0.2% | 1842 ms |
| **Whisper large-v3-turbo, GPU, no per-word timing (VoxFlow)** | 4.9% | 2.6% | 0.2% | 946 ms |
| NVIDIA Parakeet TDT 0.6B v2, GPU | 10.4% | 6.4% | 0.4% | **191 ms** |
| NVIDIA Parakeet TDT 0.6B v3, GPU | 9.4% | 6.0% | 0.2% | 204 ms |

The app uses turbo without per-word timing: faster than Google with less than half its errors. Parakeet is 5× faster still but makes twice as many errors, and for a grammar app a misheard word can't be corrected. Full large-v3 is slightly more accurate but 3.4× slower. A "verbatim" prompt (asking Whisper to keep grammar mistakes) made Whisper slightly worse, so it is off.

**4. English grammar correction** (`eval/run_gec.py`, [JFLEG](https://huggingface.co/datasets/jhu-clsp/jfleg) test set: 747 learner sentences, each with 4 human corrections; GLEU, higher is better):

| System | Held-out (sentences 301–747) | Dev (1–300) |
|---|---|---|
| No correction | 44.0 | 38.5 |
| Stage 1 rules only | 44.5 | 39.2 |
| Stage 2 LLM + edit filter only | 58.4 | 54.6 |
| **Full pipeline** | **58.5** | **55.0** |
| Rules + unfiltered LLM | 61.9 | 58.7 |

+14.5 GLEU on sentences never looked at while building the filter. The 3.3-point gap to the unfiltered LLM is the price of never letting it swap correct words for synonyms or rewrite content.

**5. Hindi and Bengali grammar correction** (`eval/run_indic_gec.py`, [IndicGEC 2025](https://github.com/BHASHA-Workshop/IndicGEC2025) shared-task test sets: real sentences with mistakes, each corrected by a person; the shared task's GLEU, higher is better; Marathi and Punjabi have no public test set):

| System | Hindi (236) | Bengali (330) |
|---|---|---|
| No correction | 80.74 | 95.79 |
| Stage 1 only (spelling + rules) | 81.29 | 95.85 |
| Stage 2 LLM + edit filter only | 82.17 | 94.21 |
| Rules + unfiltered LLM | 80.62 | 90.82 |
| **The app** (rules only) | **81.29** | **95.85** |

The unfiltered LLM makes both languages worse. With the edit filter it gained 0.9 for Hindi on this test set, but lost 0.65 on the Hindi dev set (Result 7) and damaged correct sentences in Result 11, so the app runs Stage 1 only for Hindi, as for Bengali. For scale: the best Hindi system in the shared task (GPT-4.1 mini with 10 hand-picked examples) scored 85.7.

**6. The spelling repair** (`eval/spelling_check.py`): Stage 1 with and without the sound-alike repair, on the grammar test sets (GLEU): Hindi 81.12 → **81.29**, Bengali 95.79 → **95.85**. It changed 24 words and every change was a real misspelling (বেসি → বেশি, ধ্বংশ → ধ্বংস, दुर → दूर, पानि → पानी). A broader idea, dropping an "extra" vowel from unknown words (খাইচ্ছে → খাচ্ছে), was measured too and rejected: it made wrong changes (किसम → कसम) and lowered Hindi to 81.12. Cleaning doubled vowel signs lowered IndicConformer's errors on the FLEURS clips (Bengali 13.05% → 12.58%, Hindi 11.72% → 11.60%).

**7. Choosing the LLM** (`eval/compare_llm.py`, development data only: JFLEG validation 150 sentences, IndicGEC dev 107 Hindi / 101 Bengali):

| Stage 2 model | English | Hindi | Bengali | Time per sentence |
|---|---|---|---|---|
| none | 33.0 | 88.8 | 95.3 | – |
| **llama3.2 (3B)** | 47.6 | 88.1 | 93.4 | **1.1 s** |
| gemma3:12b | **49.8** | **88.9** | 93.96 | 7.3 s |

Gemma 3 12B is slightly better but 6.5× slower, too slow for a live app. Punjabi has no grammar test set; on FLEURS speech the LLM didn't improve Punjabi (15.8% → 16.3% WER on its own) and it is the slowest language for llama3.2 (Gurmukhi becomes many tokens: 5.4 s per sentence), so it is off for Punjabi. Marathi has no grammar test set either; on Result 11 the LLM turned correct Marathi into wrong forms (आहात → आहेत), so it is off for Marathi too.

**8. The whole app on speech** (`eval/run_eval.py`, 150 FLEURS clips, the app's exact settings). Word error rate against what was said, lower is better:

| Language | Recognition | + Stage 1 | Stage 2 only | The app |
|---|---|---|---|---|
| English | 5.7% | 5.7% | 7.4% | 7.4% |
| Hindi | 11.1% | 11.1% | 11.7% | 11.7% |
| Marathi | 15.5% | 15.5% | 15.6% | 15.6% |
| Bengali (Google) | 16.7% | 16.7% | 17.1% | **16.7%** (Stage 1 only) |
| Punjabi | 15.9% | 15.8% | 16.3% | **15.8%** (Stage 1 only) |

FLEURS sentences are grammatical and read aloud, and WER measures faithfulness to what was said, so any grammar "fix" can only cost points here: the LLM adds commas and small words the speaker didn't say (English +1.7 points, Hindi +0.6, Marathi +0.1). Its value shows where there are real mistakes (JFLEG +14.5). 30 clips per language is small; differences under about 1 point are within noise.

**9. Everyday mistakes** (`eval/check_languages.py`, typed sentences with typical spoken mistakes, the whole app): **27/27 corrected**: English 5/5, Hindi 6/6, Marathi 5/5, Bengali 5/5, Punjabi 6/6 (e.g. मैं कल बाज़ार गया है → गया था, ती घरी जातो → जाते, সে গতকাল বাড়ি যায় → গেল, ਕੁੜੀ ਸਕੂਲ ਜਾਂਦਾ ਹੈ → ਜਾਂਦੀ ਹੈ).

**10. Live test, spoken** (`eval/live_check.py`; recordings saved with `SAVE_RECORDINGS=1`). 15 short sentences with a grammar mistake each, 3 per language, spoken into the app by the author (not a native speaker of every language), then every recording run through each recogniser:

| Language | Final result | What the live test changed |
|---|---|---|
| English | 3/3 | – |
| Hindi | 3/3 | – |
| Marathi | 3/3 | the recogniser wrote कल for काल and शालेत/सालेत for शाळेत: Stage 1 now restores them |
| Punjabi | 3/3 | the recogniser wrote ਅਸੀਨ for ਅਸੀਂ: Stage 1 now restores it |
| Bengali | 3/3 | IndicConformer misheard 2 of 3 short sentences (গতকাল → কতকাল, বই পড়ে → বোয়ে পরে); Google heard both correctly, so Bengali uses Google |

On these recordings Google was better than IndicConformer for Bengali and worse for Punjabi (it heard "ਅਸੀਂ ਘਰ" as "ਬੱਸ ਨ ਘਰ"); for Hindi all recognisers heard the same words. Every live sentence is now a test case in `tests/test_languages.py`. A rule that picked Google's transcript whenever IndicConformer's contained an unknown word was also measured, and rejected: it made Bengali and Hindi worse on FLEURS (12.6% → 16.2%, 11.1% → 15.0%).

**11. 1000 generated mistakes** (`eval/synthetic_gec.py`). Public test sets with real mistakes exist only for English and Hindi/Bengali, so 200 sentences per language are built from grammar tables (subjects with person/number/gender, verbs in four tenses, places, time words) with one typical mistake each: person, number or gender agreement, a tense that doesn't fit the time word ("yesterday", "last week", "tomorrow"), a missing ने/ਨੇ, the progressive form, and for English -s, didn't + past, can + -s, have/has, was/were, a/an, "could of". The correct version of each sentence is run too, to count false alarms. Stage 1 rules only:

| | First run (seed 1) | After adding rules, on a new set (seed 2) | Wrong changes | False alarms on 1000 correct sentences |
|---|---|---|---|---|
| English | 68% | 98% | 0 | 0 |
| Hindi | 46% | 96% | 0 | 0 |
| Marathi | 53% | 97% | 0 | 0 |
| Bengali | 35% | 99% | 0 | 0 |
| Punjabi | 38% | 100% | 0 | 0 |
| **All** | **48%** | **98%** | **0** | **0** |

The first run showed which kinds of mistakes the rules missed (tense with "last week"/"tomorrow", plural noun subjects, ने/ਨੇ with noun subjects, Bengali verb forms for "you"/"they"); general rules were added for them, including a 20-verb Bengali conjugation table, and the result was measured on a fresh set of 1000. Every added rule was also checked on the real test sets above: it changed 4 JFLEG sentences (all real fixes) and nothing in IndicGEC or the FLEURS transcripts. **Read the 98% for what it is:** the new set has new sentences but the same kinds of patterns and words, so it shows that these mistake types are now handled reliably, not that 98% of real-world mistakes are fixed. Real-world accuracy is what Results 4, 5 and 10 measure.

**The whole app on the same 1000** (seed 2, with the LLM where it was on at the time) showed the LLM doing harm in Hindi and Marathi:

| | Rules only | Rules + LLM | False alarms with the LLM |
|---|---|---|---|
| Hindi | 96% | 55% | 93/200 |
| Marathi | 97% | 90% | 17/200 |
| English | 98% | 98% | 2/200 |

The LLM rewrote correct sentences (गया → गया था, बैठे → बैठेंगे, आहात → आहेत) and undid correct rule fixes (पापा … टहलती → टहलते was put back to टहलती). So the small local LLM is not used for Hindi and Marathi (Result 12 shows what a large model does instead); with rules only the 1000-sentence score is the rules-only one: **98%, 0 wrong changes, 0 false alarms in Hindi, Marathi, Bengali and Punjabi**; English 98% with 2 false alarms (the LLM changed "has liked" → "likes").

**12. Several mistakes in two sentences, large models** (`eval/check_languages.py --set hard`). What people actually say: 20 cases, 4 per language, each two sentences with 2–3 mistakes (e.g. मेरी बहन कल बाजार गया था। उसने वहाँ से सब्जी खरीदा। → गई थी … खरीदी; ਮੇਰੀ ਭੈਣ ਕੱਲ੍ਹ ਦਿੱਲੀ ਗਿਆ ਸੀ। ਉਹ ਉੱਥੇ ਆਪਣੀ ਸਹੇਲੀ ਨੂੰ ਮਿਲਿਆ। → ਗਈ … ਮਿਲੀ):

| Grammar step | English | Hindi | Marathi | Bengali | Punjabi | All | Time per answer |
|---|---|---|---|---|---|---|---|
| Rules only (Stage 1) | 2/4 | 0/4 | 1/4 | 2/4 | 1/4 | **6/20** | ~0 ms |
| Rules + Gemini 3.5 Flash-Lite | 4/4 | 4/4 | 3/4 | 4/4 | 3/4 | **18/20** | ~2.3 s |
| Rules + Groq gpt-oss-120b (the app), 3 runs | 4/4 | 4/4 | 3–4/4 | 3–4/4 | 3–4/4 | **18–19/20** | ~0.6 s |

The model's misses change from run to run (one or two per run): a tense that should follow the previous sentence (বাড়ি গেল … গিয়েছিল → যাবে), once a verb form (जाते → जाणे instead of जातो), and before its examples included a similar case, Punjabi past-tense agreement with the object (ਮੈਂ ਰੋਟੀ ਖਾਧਾ → ਖਾਧੀ). The prompt tells the model to follow the previous sentence's time only for a present-tense verb: a broader version of that rule changed a correct future into the past (राहणार आहे → राहिला), so it was narrowed. 20 cases is small; read it as "about 9 in 10", not as an exact rate.

**Live, spoken by the author** (`SAVE_RECORDINGS=1`), two sentences with two mistakes each:

| Language | What the recogniser heard | Final text | |
|---|---|---|---|
| Hindi | मेरी बहन कल बाजार गया था उसने वहां से सब्जी खरीदा | मेरी बहन कल बाजार **गई थी**, उसने वहां से सब्जी **खरीदी**। | ✅ |
| Marathi | मजाभाऊ काल मुंबईला गेली तो दोन दिवस राहणार आहेत | **माझा भाऊ** काल मुंबईला **गेला**, तो दोन दिवस राहणार **आहे**। | ✅ (तिथे was not heard) |
| Punjabi | ਮੇਰੀ ਬਹਿਣ ਕੱਲ੍ਹ ਦਿੱਲੀ ਗਿਆ ਸੀ ਉਹ ਆਪਣੀ ਸਹੇਲੀ ਨੂੰ ਮਿਲਿਆ | ਮੇਰੀ **ਭੈਣ** ਕੱਲ੍ਹ ਦਿੱਲੀ **ਗਈ** ਸੀ, ਉਹ ਆਪਣੀ ਸਹੇਲੀ ਨੂੰ **ਮਿਲੀ**। | ✅ (ਉੱਥੇ was not heard) |
| Bengali | আমি কাল বাজারে যাই সেখানে মাছ কিনি | আমি কাল বাজারে **গিয়ে** সেখানে মাছ কিনি। | ½ |

The whole wait (recognition + grammar) was 1.4–2.0 s. The Bengali case is genuinely ambiguous: কাল means both "yesterday" and "tomorrow", and the model avoided choosing; with গতকাল ("yesterday") the same mistakes are fixed (the hard set above). Words the recogniser doesn't hear at all (तिथे, ਉੱਥੇ) can't be restored by any grammar step.

## Setup (macOS, Apple Silicon)

You need Python 3.12, [Ollama](https://ollama.com) and a free [Hugging Face](https://huggingface.co) account. IndicConformer is free (MIT) but gated: open [its page](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual) and accept the terms once.

```bash
cd VoxFlow
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
hf auth login            # paste a "Read" token from huggingface.co/settings/tokens
ollama pull llama3.2      # optional: offline fallback for English
```

**Grammar models (free, no credit card):** create a key at [Groq](https://console.groq.com/keys) and one at [Google AI Studio](https://aistudio.google.com/apikey), then store them in `.env` without them appearing on screen (`.env` is never uploaded, see `.gitignore`):

```bash
read -s "K?Groq key: " && echo "GROQ_API_KEY=$K" >> .env && echo " saved"
read -s "K?Gemini key: " && echo "GEMINI_API_KEY=$K" >> .env && echo " saved"
python -m eval.check_keys     # checks both keys with one Hindi sentence
```

The first start downloads IndicConformer (~2.5 GB) and Whisper large-v3-turbo (~1.6 GB).

## Run

```bash
source .venv/bin/activate
python -m app
```

Open http://127.0.0.1:8010, pick a language and press record. Speech recognition runs on the Mac; the grammar models need internet (without it: rules, plus the local LLM for English if Ollama is running); only Speak Back needs internet. The first recording after a start is slower (the models warm up).

## Evaluate

```bash
python -m eval.fetch_fleurs --n 30   # 30 FLEURS test clips per language (a few MB)
python -m eval.speed_test            # Google vs VoxFlow, live, same clips (languages not on Google) -> eval/results/speed_test.md
python -m eval.compare_asr           # recogniser accuracy, Indian languages -> eval/results/asr_comparison_indian.md
python -m eval.fetch_svarah --n 60   # Indian-accented English clips (needs the Hugging Face login)
python -m eval.compare_english       # recogniser comparison, Indian English -> eval/results/indian_english.md
python -m eval.run_eval              # the whole app on speech (WER/CER per stage) -> eval/results/results.md
python -m eval.run_gec               # English grammar correction on JFLEG -> eval/results/gec.md
python -m eval.run_indic_gec         # Hindi + Bengali grammar correction on IndicGEC 2025 -> eval/results/indic_gec.md
python -m eval.spelling_check        # the spelling repair, on vs off -> eval/results/spelling_check.md
python -m eval.compare_llm           # which Ollama model for Stage 2 (development data) -> eval/results/llm_comparison.md
python -m eval.check_languages       # everyday mistakes in all five languages -> eval/results/language_check.md
python -m eval.synthetic_gec --seed 2   # 1000 generated mistakes through the whole app -> eval/results/synthetic_seed2.md
python -m eval.try_recording my-recording.wav --lang bn-IN   # one recording through every engine for that language
SAVE_RECORDINGS=1 python -m app      # keep every recording you make in eval/live/ (stays on your computer)
python -m eval.live_check            # all saved recordings through every engine, side by side
```

WER and CER are computed in `eval/metrics.py` (edit distance, written from scratch) after removing case and punctuation, so an added full stop isn't counted as an error. Recognition and LLM answers are cached, so re-runs are fast (`speed_test` never uses the cache).

## Tests

```bash
pytest
```

355 tests: everyday mistakes in all five languages (fixed, and the correct versions left alone), every Stage 1 rule and the spelling repair, the 15 live-test sentences exactly as the app heard them, the LLM guard and edit filter (with cases taken from real LLM mistakes), the cloud models (model choice, fallback on rate limits, request-setting fallback, the answer check and the speaker-gender guard), accept/reject rebuilding the text, WAV decoding (PCM, float, 24-bit), resampling and VAD, every recogniser (IndicConformer, Whisper on GPU/CPU, Parakeet, Google) and the fallback to Google, the text-to-speech fallback, all API endpoints and error codes, and the WER/CER maths. Models, Google, Groq, Gemini, Ollama and edge-tts are replaced by fakes, so tests run offline in seconds.

## Project structure

```
app/
  main.py          FastAPI app: API endpoints + serves the website
  pipeline.py      one correction strategy per language (strategy pattern) and the router
  asr.py           speech recognition (IndicConformer, Whisper on GPU/CPU, Parakeet, Google), per-language routing, fallback, retry
  audio.py         WAV reading, resampling to 16 kHz, noise reduction, VAD
  stage1.py        spelling repair + rule-based grammar correction for all five languages
  stage2.py        grammar step: prompts, answer checks (light for large models, word-by-word filter for the local one)
  cloud_llm.py     free cloud LLMs: Groq -> Gemini -> local Ollama, with rate-limit fallback
  diff.py          corrections list and text segments for accept/reject
  tts.py           Speak Back with edge-tts
  languages.py     the five languages: codes, scripts, voices
  static/          the website (index.html, styles.css, app.js, recorder-worklet.js)
data/fewshot/      few-shot correction examples per language (local model)
data/fewshot_strong/  examples for the large models (several mistakes, misheard and run-together words)
eval/              test-data downloads, WER/CER/GLEU metrics, benchmarks, results
tests/             pytest suite
```

## Configuration

`.env` holds the grammar-model keys (Setup). See `.env.example` for the rest: which cloud model is asked first (`STAGE2_ORDER`), a fixed model name, the local Ollama model, the speech recognisers, the port.

## Data credits

- [JFLEG](https://huggingface.co/datasets/jhu-clsp/jfleg) (JHU), CC BY-NC-SA 4.0: English grammar correction test sentences.
- [IndicGEC 2025](https://github.com/BHASHA-Workshop/IndicGEC2025) (BHASHA workshop shared task): Hindi and Bengali grammar correction test sentences.
- [FLEURS](https://huggingface.co/datasets/google/fleurs) (Google), CC BY 4.0: multilingual test recordings.
- [Svarah](https://huggingface.co/datasets/ai4bharat/Svarah) (AI4Bharat), CC BY 4.0: Indian-accented English test recordings.

Test audio is downloaded by the scripts in `eval/` and is not stored in this repository.
