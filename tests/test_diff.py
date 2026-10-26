from app import diff
from app.text_utils import join_tokens, tokenize


def rebuild(segments, corrections, rejected=()):
    tokens = []
    for seg in segments:
        if seg["kind"] == "same":
            tokens += seg["tokens"]
        else:
            c = next(c for c in corrections if c["id"] == seg["id"])
            tokens += c["original"] if c["id"] in rejected else c["corrected"]
    return join_tokens(tokens)


def test_tokenize_splits_punctuation_and_keeps_apostrophes():
    assert tokenize("She doesn't, (really).") == ["She", "doesn't", ",", "(", "really", ")", "."]
    assert join_tokens(tokenize("She doesn't, (really).")) == "She doesn't, (really)."


def test_segments_rebuild_final_text_and_rejecting_restores_original():
    raw, s1, final = "i go to park yesterday", "I went to the park yesterday.", "I went to the park yesterday."
    segments, corrections = diff.build(raw, s1, final)
    assert rebuild(segments, corrections) == final
    assert rebuild(segments, corrections, rejected={c["id"] for c in corrections}) == raw
    the = next(c for c in corrections if c["corrected"] == ["the"])
    assert the["type"] == "insertion" and the["source"] == "rule"


def test_corrections_made_by_llm_are_labelled_llm():
    raw = "there is many problem"
    s1 = "There are many problem."
    final = "There are many problems."
    _, corrections = diff.build(raw, s1, final)
    by_word = {w: c["source"] for c in corrections for w in c["corrected"]}
    assert by_word["problems"] == "llm"
    assert by_word["are"] == "rule"


def test_no_changes_means_no_corrections():
    segments, corrections = diff.build("Hello there.", "Hello there.", "Hello there.")
    assert corrections == [] and rebuild(segments, corrections) == "Hello there."


def test_doubled_vowel_signs_from_the_recogniser_are_cleaned():
    from app.text_utils import normalize_unicode
    from app import stage1
    assert normalize_unicode("খাইছেি") == "খাইছে"                      # live IndicConformer output
    assert stage1.correct("আমি ভাত খাইছেি", "bn-IN") == "আমি ভাত খাইছি।"
    for word in ["কৌতূহল", "গাঁ", "বাংলা", "हिंदी", "कौन", "ਪੰਜਾਬੀ", "ਕੁੜੀ"]:   # real combinations stay
        assert normalize_unicode(word) == word
