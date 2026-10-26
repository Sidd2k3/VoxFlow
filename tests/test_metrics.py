from eval.metrics import corpus_cer, corpus_wer, normalize_for_scoring


def test_normalization_ignores_case_and_punctuation():
    assert normalize_for_scoring("Hello, World!") == "hello world"
    assert normalize_for_scoring("मैं जाता हूँ।") == "मैं जाता हूँ"


def test_wer():
    assert corpus_wer(["the cat sat"], ["the cat sat"]) == 0
    assert corpus_wer(["the cat sat"], ["the bat sat down"]) == 2 / 3
    assert corpus_wer(["a b", "c d"], ["a b", "c"]) == 1 / 4


def test_cer():
    assert corpus_cer(["abc"], ["abd"]) == 1 / 3


def test_gleu_rewards_corrections_and_penalises_leaving_errors():
    from eval.run_gec import gleu
    src = ["she go to school every days ."]
    ref = ["she goes to school every day ."]
    perfect = gleu(src, ref, ref)
    unchanged = gleu(src, src, ref)
    partial = gleu(src, ["she goes to school every days ."], ref)
    assert perfect > partial > unchanged and abs(perfect - 100) < 1e-6


def test_missed_words_counts_only_dropped_words():
    from eval.metrics import corpus_missed_words, edit_ops
    assert edit_ops("she was looking at me like she knew".split(), "she was looking at me she knew".split()) == \
        {"sub": 0, "del": 1, "ins": 0}
    assert edit_ops("a b c".split(), "a x c d".split()) == {"sub": 1, "del": 0, "ins": 1}
    assert abs(corpus_missed_words(["one two three four"], ["one three four"]) - 0.25) < 1e-9


def test_indicgec_gleu_matches_its_definition():
    from eval.run_indic_gec import shared_task_gleu
    assert shared_task_gleu(["राम घर गया।"], ["राम घर गया।"]) == 100.0
    assert shared_task_gleu(["राम घर गया।"], ["राम घर जाता।"]) < 100.0
    # invisible encoding differences don't count: precomposed য় vs য + nukta
    word = "\u0986\u09ae\u09bf \u09ac\u09be\u09df\u09bf \u09af\u09be\u0987 \u098f\u09ac\u0982 \u0986\u09b8\u09bf"   # আমি বাড়ি... with precomposed য়
    assert shared_task_gleu([word], [word.replace("\u09df", "\u09af\u09bc")]) == 100.0
