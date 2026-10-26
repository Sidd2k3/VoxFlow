import httpx

from app import stage2
from tests.conftest import FakeLLM


def test_intensity_controls_stage2():
    assert stage2.correction_level(0) == 0 and stage2.correction_level(33) == 0      # rules only
    assert stage2.correction_level(34) == 1 and stage2.correction_level(79) == 1     # grammar + spelling
    assert stage2.correction_level(80) == 2 and stage2.correction_level(100) == 2    # + unsure real words


def test_low_intensity_skips_llm():
    llm = FakeLLM("anything")
    result = stage2.correct("There are many problem.", "en-US", intensity=10, client=llm)
    assert result.status == "skipped" and llm.calls == []


def test_small_fix_is_applied():
    result = stage2.correct("There are many problem.", "en-US", 50, FakeLLM("There are many problems."))
    assert result.status == "applied" and result.text == "There are many problems."


def test_same_answer_is_unchanged():
    result = stage2.correct("I went home.", "en-US", 50, FakeLLM("I went home"))
    assert result.status == "unchanged" and result.text == "I went home."


def test_translation_is_rejected():
    result = stage2.correct("मैं घर जाता हूँ।", "hi-IN", 100, FakeLLM("I go home."))
    assert result.status == "rejected" and "script" in result.reason
    assert result.text == "मैं घर जाता हूँ।"


def test_full_rewrite_is_reverted():
    source = "I went to the market today to buy some fresh vegetables."
    result = stage2.correct(source, "en-US", 50,
                            FakeLLM("Today I visited a shop where I purchased vegetables that were fresh."))
    assert result.text == source and result.reverted_edits > 0


def test_ollama_down_is_reported():
    result = stage2.correct("Hello.", "en-US", 50, FakeLLM(error=httpx.ConnectError("refused")))
    assert result.status == "unavailable" and result.text == "Hello."


def test_prompt_uses_language_and_fewshot_examples():
    messages = stage2.build_messages("मैं जाता है", "hi-IN")
    assert "Hindi" in messages[0]["content"] and "Devanagari" in messages[0]["content"]
    assert messages[-1] == {"role": "user", "content": "Transcript: मैं जाता है"}
    assert len(messages) == 1 + 2 * len(stage2.load_examples("hi-IN")) + 1


def test_word_edit_distance():
    assert stage2.word_edit_distance(["a", "b", "c"], ["a", "x", "c", "d"]) == 2


# Edit filter: cases taken from real llama3.2 answers on FLEURS clips.

def test_invented_words_are_reverted():
    text, kept, reverted = stage2.filter_edits(
        "The northern marianas emergency office said so.",
        "The Northern Mariana Islands Emergency Office said so.", "en-US")
    assert "Islands" not in text and reverted == 1


def test_dropped_numbers_are_reverted():
    src = "आपको एक अलग वीजा की आवश्यकता नहीं होगी 2009 तक।"
    text, _, reverted = stage2.filter_edits(src, "आपको एक अलग वीजा की आवश्यकता नहीं होगी।", "hi-IN")
    assert "2009" in text and reverted == 1


def test_synonym_swaps_are_reverted_but_inflection_fixes_kept():
    text, kept, reverted = stage2.filter_edits(
        "ਦੋਵਾਂ ਤੱਤਾਂ ਦਾ ਸੁਮੇਲ ਕਰਦਾ ਹੈ।", "ਦੋਵਾਂ ਤੱਤਾਂ ਦਾ ਸੰਗਮ ਕਰਦਾ ਹੈ।", "pa-IN")
    assert "ਸੁਮੇਲ" in text and reverted == 1
    text, kept, _ = stage2.filter_edits("There are many problem.", "There are many problems.", "en-US")
    assert text == "There are many problems." and kept == 1


def test_grammar_words_can_be_added_and_heard_words_used():
    text, kept, _ = stage2.filter_edits("She go to school.", "She goes to the school.", "en-US")
    assert text == "She goes to the school." and kept == 2
    text, kept, _ = stage2.filter_edits("lion price act like wolves", "lion prides act like wolves",
                                        "en-US", heard={"prides"})
    assert "prides" in text and kept == 1


def test_alternatives_are_given_to_the_llm():
    messages = stage2.build_messages("lion price act", "en-US", ["lion prides act", "lion price act"])
    assert "Other hypotheses from the recogniser: lion prides act" in messages[-1]["content"]


def test_copied_separator_and_word_splits_are_reverted():
    text, _, reverted = stage2.filter_edits("সেইসব কথা বলেছে।", "সেই সব কথা বলেছে। |", "bn-IN", heard=set())
    assert text == "সেইসব কথা বলেছে।" and reverted >= 1


def test_real_word_spelling_variants_are_not_swapped():
    # "ऑनलाइन" and "ऑनलाईन" are both real words: not a grammar fix, not a misspelling -> reverted
    for heard in (set(), None):
        text, _, reverted = stage2.filter_edits("ऑनलाइन बुकिंग करें।", "ऑनलाईन बुकिंग करें।", "hi-IN", heard=heard)
        assert "ऑनलाइन" in text and reverted == 1


def test_priority_grammar_then_misspelling_then_nothing_else():
    cases = [
        ("She doesn't knows it.", "She doesn't know it.", "en-US", True),               # grammar: verb form
        ("she knowed me.", "she knew me.", "en-US", True),                              # grammar: over-regularised
        ("I went their yesterday.", "I went there yesterday.", "en-US", True),          # grammar: confused word
        ("He is a honest man and it are good.", "He is an honest man and it is good.", "en-US", True),
        ("यह पता प्रदान करेगी।", "यह पता प्रदान करेगा।", "hi-IN", True),                    # grammar: gender agreement
        ("I recieved the parcel.", "I received the parcel.", "en-US", True),            # non-word -> real word
        ("He bought a car.", "He purchased a car.", "en-US", False),                    # synonym swap
        ("ਤੱਤਾਂ ਦਾ ਸੁਮੇਲ ਕਰਦਾ ਹੈ।", "ਤੱਤਾਂ ਦਾ ਸੰਗਮ ਕਰਦਾ ਹੈ।", "pa-IN", False),             # synonym swap
    ]
    for src, answer, lang, accepted in cases:
        text, _, _ = stage2.filter_edits(src, answer, lang, heard=set())
        assert (text == answer) is accepted, (src, text)


def test_inserted_content_words_are_reverted_but_grammar_words_kept():
    text, kept, reverted = stage2.filter_edits("She went to market.", "She went to the big market.", "en-US", heard=set())
    assert text == "She went to the market." and kept == 1 and reverted == 1


def test_finalize_reports_reverted_edits():
    r = stage2.finalize("The northern marianas office said so.", "The Northern Mariana Islands office said so.",
                        "en-US", 50, "fake", alternatives=[])
    assert r.reverted_edits == 1 and "reverted" in r.reason and "Islands" not in r.text



def test_english_inflection_fix_is_kept_for_speech_but_word_swaps_are_not():
    # real case: Whisper heard "She doesn't knows ...", the LLM fixed it, the old filter reverted it
    text, kept, _ = stage2.filter_edits("She doesn't knows what I was talking about.",
                                        "She doesn't know what I was talking about.", "en-US", heard=set())
    assert text == "She doesn't know what I was talking about." and kept == 1
    text, _, reverted = stage2.filter_edits("He bought a car.", "He purchased a car.", "en-US", heard=set())
    assert text == "He bought a car." and reverted == 1



def test_only_low_confidence_words_may_be_replaced():
    # real case: Whisper wrote "She" as "C" with low word probability
    fixed, kept, _ = stage2.filter_edits("C doesn't know what I was talking about.",
                                         "She doesn't know what I was talking about.", "en-US",
                                         heard=set(), uncertain={"C"})
    assert fixed.startswith("She") and kept == 1
    same, _, reverted = stage2.filter_edits("C doesn't know.", "She doesn't know.", "en-US", heard=set(), uncertain=set())
    assert same.startswith("C") and reverted == 1          # confident words stay protected


def test_uncertain_words_are_listed_in_the_prompt():
    messages = stage2.build_messages("C doesn't know.", "en-US", [], ["C"])
    assert "unsure about (possibly misheard): C" in messages[-1]["content"]


def test_many_grammar_fixes_in_one_sentence_are_all_kept():
    # real JFLEG case: four grammar fixes used to exceed the change limit, so all were thrown away
    r = stage2.finalize("When price of gas goes up, the consumer do not want buy gas for fuels.",
                        "When the price of gas goes up, the consumer does not want to buy gas for fuel.",
                        "en-US", 50, "fake")
    assert r.text == "When the price of gas goes up, the consumer does not want to buy gas for fuel."


def test_spelling_fixes_next_to_other_changes_are_kept():
    fixed, _, _ = stage2.filter_edits("Unforturntly, older people can not use internet.",
                                      "Unfortunately, older people cannot use the internet.", "en-US")
    assert fixed == "Unfortunately, older people cannot use the internet."


def test_english_split_words_and_word_class_fixes():
    assert stage2.filter_edits("Forexample, my cousin is 12years old.", "For example, my cousin is 12 years old.",
                               "en-US")[0] == "For example, my cousin is 12 years old."
    assert stage2.filter_edits("Bigger farming use more product.", "Bigger farms use more products.",
                               "en-US")[0] == "Bigger farms use more products."
    assert stage2.filter_edits("He wants to success.", "He wants to succeed.", "en-US")[0] == "He wants to succeed."
    # not grammar: synonyms, compounds with another word, every-day merges
    assert stage2.filter_edits("Through the night it rained.", "Throughout the night it rained.",
                               "en-US")[0] == "Through the night it rained."
    assert stage2.filter_edits("This rebuffs the claim.", "This refutes the claim.", "en-US")[0] == "This rebuffs the claim."
    assert stage2.filter_edits("I walk every day.", "I walk everyday.", "en-US")[0] == "I walk every day."


def test_typo_of_a_grammar_word_is_fixed():
    assert stage2.filter_edits("I come form Delhi.", "I come from Delhi.", "en-US")[0] == "I come from Delhi."
    assert stage2.filter_edits("Fo me it is hard.", "For me it is hard.", "en-US")[0] == "For me it is hard."


def test_unsure_real_word_is_kept_unless_intensity_is_high():
    # real FLEURS case: Whisper was unsure about "movies" (it was right); the LLM wanted "films"
    src, ans = "These movies look at people.", "These films look at people."
    assert stage2.filter_edits(src, ans, "en-US", set(), {"movies"})[0] == src
    assert stage2.filter_edits(src, ans, "en-US", set(), {"movies"}, level=2)[0] == ans


def test_final_full_stop_is_never_dropped():
    assert stage2.finalize("I went home.", "I went home", "en-US", 50, "fake").text == "I went home."


def test_hindi_tense_change_of_the_same_verb_is_grammar():
    # live case: "मैं कल बाजार जाता हूँ" - कल needs past (or future), and गया is the past of जाना
    assert stage2.filter_edits("मैं कल बाजार जाता हूँ।", "मैं कल बाजार गया था।", "hi-IN")[0] == "मैं कल बाजार गया था।"
    assert stage2.filter_edits("मैं कल बाजार जाता हूँ।", "मैं कल बाजार जाऊँगा।", "hi-IN")[0] == "मैं कल बाजार जाऊँगा।"
    assert stage2.filter_edits("वह खाना खाता है।", "वह खाना खाया।", "hi-IN")[0] == "वह खाना खाया।"
    # a different verb is not a tense fix
    assert stage2.filter_edits("मैं घर जाता हूँ।", "मैं घर आता हूँ।", "hi-IN")[0] == "मैं घर जाता हूँ।"


def test_indic_spelling_and_grammar_word_fixes_from_indicgec_train():
    # cases taken from the IndicGEC 2025 training data (not the test set)
    assert stage2.filter_edits("आज प्रदुषण बढ़ रहा है।", "आज प्रदूषण बढ़ रहा है।", "hi-IN")[0] == "आज प्रदूषण बढ़ रहा है।"
    assert stage2.filter_edits("शिक्षा पे सबका अधिकार है।", "शिक्षा पर सबका अधिकार है।", "hi-IN")[0] == \
        "शिक्षा पर सबका अधिकार है।"
    # two common, different words with the same consonants are not a spelling fix: कम (less) -> काम (work)
    assert stage2.filter_edits("मुझे कम है।", "मुझे काम है।", "hi-IN")[0] == "मुझे कम है।"


def test_bengali_verb_forms_and_auxiliaries():
    assert stage2._inflection("শুনো", "শুনতে", "bn-IN") and stage2._inflection("শুনলে", "শুনে", "bn-IN")
    assert stage2.filter_edits("ঘর বাড়ি যেমন থাকিবে।", "ঘর বাড়ি যেমন ছিল।", "bn-IN")[0] == "ঘর বাড়ি যেমন ছিল।"


def test_llm_may_not_change_the_speakers_gender():
    # live case: the LLM turned "मैं स्कूल जाता हूँ" into "मैं स्कूल जाती हूँ" - both are correct
    assert stage2.filter_edits("मैं स्कूल जाता हूँ।", "मैं स्कूल जाती हूँ।", "hi-IN")[0] == "मैं स्कूल जाता हूँ।"
    assert stage2.filter_edits("ਮੈਂ ਸਕੂਲ ਜਾਂਦਾ ਹਾਂ।", "ਮੈਂ ਸਕੂਲ ਜਾਂਦੀ ਹਾਂ।", "pa-IN")[0] == "ਮੈਂ ਸਕੂਲ ਜਾਂਦਾ ਹਾਂ।"
    # a third-person subject's gender is grammar, and tense fixes still pass
    assert stage2.filter_edits("लड़की स्कूल जाता है।", "लड़की स्कूल जाती है।", "hi-IN")[0] == "लड़की स्कूल जाती है।"
    assert stage2.filter_edits("मैं कल बाजार जाता हूँ।", "मैं कल बाजार गया था।", "hi-IN")[0] == "मैं कल बाजार गया था।"
