"""Every language: typical grammar mistakes are fixed by the rules, and correct sentences stay unchanged."""

import pytest

from app import stage1
from eval.language_cases import CASES

ALL = [(code, wrong, right) for code, cases in CASES.items() for wrong, right in cases]


@pytest.mark.parametrize("code,wrong,right", ALL)
def test_mistake_is_fixed(code, wrong, right):
    assert stage1.correct(wrong, code) == right


@pytest.mark.parametrize("code,wrong,right", ALL)
def test_correct_sentence_is_left_alone(code, wrong, right):
    assert stage1.correct(right, code) == right


@pytest.mark.parametrize("code,sentence", [
    ("hi-IN", "माँ ने खाना बनाया था।"),          # ergative: the verb agrees with the object
    ("pa-IN", "ਮਾਂ ਨੇ ਰੋਟੀ ਬਣਾਇਆ ਸੀ।"),
    ("mr-IN", "मुलगी आंबा खाल्ला।"),
    ("hi-IN", "लड़का स्कूल जाता है।"),
    ("mr-IN", "तो घरी जातो।"),
])
def test_tricky_correct_sentences_are_left_alone(code, sentence):
    assert stage1.correct(sentence, code) == sentence


@pytest.mark.parametrize("code,wrong,right", [
    # live: ਭੈਣ was heard as ਬਹਿਣ, ਕੱਲ੍ਹ as ਕਲ, and sometimes the auxiliary was dropped
    ("pa-IN", "ਮੇਰੀ ਬਹਿਣ ਕਲ ਬਾਜ਼ਾਰ ਗਿਆ ਹੈ।", "ਮੇਰੀ ਬਹਿਣ ਕਲ ਬਾਜ਼ਾਰ ਗਈ ਸੀ।"),
    ("pa-IN", "ਮੇਰੀ ਬਹਿਣ ਕਲ ਬਾਜ਼ਾਰ ਗਿਆ।", "ਮੇਰੀ ਬਹਿਣ ਕਲ ਬਾਜ਼ਾਰ ਗਈ।"),
    ("hi-IN", "मेरी बहिन कल बाजार गया।", "मेरी बहिन कल बाजार गई।"),
])
def test_recogniser_spellings_and_missing_auxiliary(code, wrong, right):
    assert stage1.correct(wrong, code) == right


@pytest.mark.parametrize("code,sentence", [
    ("pa-IN", "ਦਾਦਾ ਜੀ ਬਾਜ਼ਾਰ ਗਿਆ।"), ("pa-IN", "ਮੇਰਾ ਚਾਚਾ ਬਾਜ਼ਾਰ ਗਿਆ।"),   # masculine look-alikes
    ("hi-IN", "दादा बाजार गया।"), ("hi-IN", "बेटा घर आया।"), ("pa-IN", "ਮੈਂ ਕੱਲ੍ਹ ਬਾਜ਼ਾਰ ਗਿਆ।"),
])
def test_masculine_subjects_keep_masculine_verbs(code, sentence):
    assert stage1.correct(sentence, code) == sentence


@pytest.mark.parametrize("code,wrong,right", ALL)
def test_typed_text_in_the_selected_language(code, wrong, right):
    """The whole text path as on the website (selected language -> rules -> LLM step), with an LLM that changes nothing."""
    from app.pipeline import build_router
    from tests.conftest import FakeLLM
    out = build_router(llm=FakeLLM(wrong)).process_text(wrong, code, 50)
    assert out["language"] == code and out["final_text"] == right


@pytest.mark.parametrize("wrong,right", [
    ("সে গোটকাল বাড়ি যায়।", "সে গতকাল বাড়ি গেল।"),        # live: গতকাল heard as গোটকাল
    ("আমি গটকাল স্কুলে যাই।", "আমি গতকাল স্কুলে গেলাম।"),
])
def test_bengali_recogniser_spellings_of_yesterday(wrong, right):
    assert stage1.correct(wrong, "bn-IN") == right


def test_bengali_words_that_only_look_similar_are_kept():
    for sentence in ["সে কাল বাড়ি যাবে।", "আমি গত বছর গিয়েছিলাম।", "সে কাজ করে।"]:
        assert stage1.correct(sentence, "bn-IN") == sentence


def test_rare_recogniser_spelling_becomes_the_common_word():
    # live: বাড়ি (home) heard as বারি
    assert stage1.correct("সে গতকাল বারি যায়।", "bn-IN") == "সে গতকাল বাড়ি গেল।"
    assert stage1.correct("সে গতকাল বারি যায়।", "bn-IN", spelling=False) == "সে গতকাল বারি গেল।"
    assert stage1.repair_spelling(["বাড়ি", "যায়"], "bn-IN") == ["বাড়ি", "যায়"]      # common words stay
    assert stage1.repair_spelling(["ਬਾਜ਼ਾਰ"], "pa-IN") == ["ਬਾਜ਼ਾਰ"]                    # no word list: untouched


def test_marathi_yesterday_written_the_hindi_way():
    # live: काल heard as कल, so the past-tense rule didn't fire
    assert stage1.correct("मी कल बाजारात जातो।", "mr-IN") == "मी काल बाजारात गेलो।"
    assert stage1.correct("मी काल बाजारात गेलो।", "mr-IN") == "मी काल बाजारात गेलो।"


@pytest.mark.parametrize("wrong,right", [
    ("আমি ভাত খাচ্ছে।", "আমি ভাত খাচ্ছি।"),      # -চ্ছে (live: "আমি ভাত খাইচ্ছে")
    ("আমি কাজ করছে।", "আমি কাজ করছি।"),
    ("আমি ভালো আছে।", "আমি ভালো আছি।"),
    ("আমি ভাত খেয়েছে।", "আমি ভাত খেয়েছি।"),
])
def test_bengali_first_person_verb_endings(wrong, right):
    assert stage1.correct(wrong, "bn-IN") == right


def test_bengali_words_ending_like_verbs_are_kept():
    for sentence in ["আমি তার কাছে।", "পাখি গাছে।", "সে ভাত খাচ্ছে।"]:
        assert stage1.correct(sentence, "bn-IN") == sentence


@pytest.mark.parametrize("code,heard,right", [
    # exactly what the app heard in the live test (2026-10-25) -> what it must show
    ("hi-IN", "हम लोग दिल्ली में रहता है", "हम लोग दिल्ली में रहते हैं।"),
    ("hi-IN", "लड़की स्कूल जाता है", "लड़की स्कूल जाती है।"),
    ("hi-IN", "मेरी बहन कल आया था", "मेरी बहन कल आई थी।"),
    ("mr-IN", "मी काल बाजारात जातो", "मी काल बाजारात गेलो।"),
    ("mr-IN", "मुलगी सालेत गेला", "मुलगी शाळेत गेली।"),
    ("mr-IN", "मुलगी शालेत गेला", "मुलगी शाळेत गेली।"),
    ("mr-IN", "मी शालेत जातो आहे", "मी शाळेत जात आहे।"),
    ("bn-IN", "আমি ভাত খেয়েছে", "আমি ভাত খেয়েছি।"),
    ("pa-IN", "ਕੁੜੀ ਸਕੂਲ ਜਾਂਦਾ ਹੈ", "ਕੁੜੀ ਸਕੂਲ ਜਾਂਦੀ ਹੈ।"),
    ("pa-IN", "ਅਸੀਨ ਘਰ ਜਾਂਦਾ ਹੈ", "ਅਸੀਂ ਘਰ ਜਾਂਦੇ ਹਾਂ।"),
])
def test_live_recordings(code, heard, right):
    assert stage1.correct(heard, code) == right


def test_real_marathi_words_that_look_like_school_are_kept():
    for sentence in ["तो माझा साला आहे।", "शाळा सुरू झाली।", "मी शाळेत जातो."]:
        out = stage1.correct(sentence, "mr-IN")
        assert ("साला" in out) if "साला" in sentence else ("शाळ" in out)


@pytest.mark.parametrize("code,wrong,right", [
    # the second live-test set (2026-10-26)
    ("en-US", "she don't like coffee", "She doesn't like coffee."),
    ("en-US", "he did not went home", "He did not go home."),
    ("en-US", "I buy a apple yesterday", "I bought an apple yesterday."),
    ("hi-IN", "लड़का स्कूल जाती है", "लड़का स्कूल जाता है।"),          # masculine subject, feminine verb
    ("hi-IN", "मेरी माँ खाना बनाता है", "मेरी माँ खाना बनाती है।"),
    ("hi-IN", "हम खाना खाता है", "हम खाना खाते हैं।"),
    ("hi-IN", "मेरा भाई कल आई थी", "मेरा भाई कल आया था।"),
    ("mr-IN", "ती शाळेत जातो", "ती शाळेत जाते।"),
    ("mr-IN", "आम्ही घरी जात आहे", "आम्ही घरी जात आहोत।"),
    ("mr-IN", "माझी आई घरी जातो", "माझी आई घरी जाते।"),            # feminine possessive = feminine subject
    ("bn-IN", "সে ভাত খাই", "সে ভাত খায়।"),
    ("bn-IN", "আমি গতকাল বাজারে যাই", "আমি গতকাল বাজারে গেলাম।"),
    ("bn-IN", "আমি কাজ করছে", "আমি কাজ করছি।"),
    ("pa-IN", "ਮੈਂ ਰੋਟੀ ਖਾਂਦਾ ਹੈ", "ਮੈਂ ਰੋਟੀ ਖਾਂਦਾ ਹਾਂ।"),
    ("pa-IN", "ਮੇਰੀ ਭੈਣ ਸਕੂਲ ਜਾਂਦਾ ਹੈ", "ਮੇਰੀ ਭੈਣ ਸਕੂਲ ਜਾਂਦੀ ਹੈ।"),
    ("pa-IN", "ਤੁਸੀਂ ਘਰ ਜਾਂਦਾ ਹੈ", "ਤੁਸੀਂ ਘਰ ਜਾਂਦੇ ਹੋ।"),
])
def test_second_live_set(code, wrong, right):
    assert stage1.correct(wrong, code) == right


def test_gender_rules_leave_correct_sentences_alone():
    for code, sentence in [("hi-IN", "मेरा भाई स्कूल जाता है।"), ("hi-IN", "लड़की स्कूल जाती है।"),
                           ("hi-IN", "मेरे भाई ने रोटी खाई।"), ("mr-IN", "माझा भाऊ घरी जातो।"),
                           ("mr-IN", "माझी आई घरी जाते।")]:
        assert stage1.correct(sentence, code) == sentence


def test_realistic_sentences_fixed_by_the_rules():
    """The realistic set for the languages without the LLM step (Bengali, Punjabi) and the Hindi/Marathi ones the
    rules cover: all must be fixed by Stage 1 alone."""
    from eval.language_cases import REAL_CASES
    for code in ["hi-IN", "mr-IN", "bn-IN", "pa-IN"]:
        for wrong, right in REAL_CASES[code]:
            assert stage1.correct(wrong, code) == right, (code, wrong)


@pytest.mark.parametrize("code,sentence", [
    ("hi-IN", "मेरे घर में बिल्ली रहती है।"), ("hi-IN", "उनके पापा डॉक्टर हैं।"), ("hi-IN", "मेरे पापा रोज टहलते हैं।"),
    ("hi-IN", "हमने खाना खाया।"), ("mr-IN", "ते रोज व्यायाम करते।"), ("bn-IN", "আমার মনে হয়।"),
    ("pa-IN", "ਉਹ ਸਕੂਲ ਜਾਂਦਾ ਹੈ।"),
])
def test_extended_rules_leave_correct_sentences_alone(code, sentence):
    assert stage1.correct(sentence, code) == sentence


@pytest.mark.parametrize("code,wrong,right", [
    # rules added after the first 1000-sentence test (2026-10-26); hand-written examples
    ("en-US", "The children goes to school", "The children go to school."),
    ("en-US", "They likes coffee every day", "They like coffee every day."),
    ("en-US", "Last week they goes to Delhi", "Last week they went to Delhi."),
    ("hi-IN", "बच्चे रोज़ बाज़ार जाता है", "बच्चे रोज़ बाज़ार जाते हैं।"),
    ("hi-IN", "लड़कियाँ कल रात जल्दी सोती हैं", "लड़कियाँ कल रात जल्दी सोईं।"),
    ("hi-IN", "मेरा भाई आम खाया", "मेरे भाई ने आम खाया।"),
    ("mr-IN", "ते रोज बाजारात जातो", "ते रोज बाजारात जातात।"),
    ("mr-IN", "तुम्ही काल ऑफिसला जाता", "तुम्ही काल ऑफिसला गेलात।"),
    ("mr-IN", "तू उद्या मैदानावर खेळलास", "तू उद्या मैदानावर खेळशील।"),
    ("bn-IN", "আপনি প্রতিদিন স্কুলে যায়", "আপনি প্রতিদিন স্কুলে যান।"),
    ("bn-IN", "আমরা আগামীকাল ভাত খেলাম", "আমরা আগামীকাল ভাত খাব।"),
    ("bn-IN", "তারা গতকাল টিভি দেখে", "তারা গতকাল টিভি দেখল।"),
    ("pa-IN", "ਕੁੜੀਆਂ ਰੋਜ਼ ਸਕੂਲ ਜਾਂਦਾ ਹੈ", "ਕੁੜੀਆਂ ਰੋਜ਼ ਸਕੂਲ ਜਾਂਦੀਆਂ ਹਨ।"),
    ("pa-IN", "ਮੈਂ ਪਿਛਲੇ ਹਫ਼ਤੇ ਪਾਰਕ ਵਿੱਚ ਖੇਡਦਾ ਹਾਂ", "ਮੈਂ ਪਿਛਲੇ ਹਫ਼ਤੇ ਪਾਰਕ ਵਿੱਚ ਖੇਡਿਆ।"),
    ("pa-IN", "ਮੇਰਾ ਦੋਸਤ ਰੋਟੀ ਖਾਧੀ", "ਮੇਰੇ ਦੋਸਤ ਨੇ ਰੋਟੀ ਖਾਧੀ।"),
    ("pa-IN", "ਮੇਰਾ ਭਰਾ ਅਗਲੇ ਹਫ਼ਤੇ ਦਿੱਲੀ ਜਾਵੇਗੀ", "ਮੇਰਾ ਭਰਾ ਅਗਲੇ ਹਫ਼ਤੇ ਦਿੱਲੀ ਜਾਵੇਗਾ।"),
])
def test_rules_from_the_large_test(code, wrong, right):
    assert stage1.correct(wrong, code) == right


@pytest.mark.parametrize("code,sentence", [
    # must stay exactly as they are (found while checking the rules against the real test sets)
    ("en-US", "The even ask themself how this happened."), ("en-US", "The bus goes to Delhi."),
    ("hi-IN", "पापा के जाते समय मैंने उन्हें देखा।"), ("hi-IN", "लड़के ने रोटी खाई।"),
    ("mr-IN", "मी रोज शाळेत जाते।"), ("mr-IN", "त्या काल घरी आल्या।"),
    ("bn-IN", "সে খেলে।"), ("bn-IN", "তুমি আগামীকাল যাবে।"),
    ("pa-IN", "ਮੈਂ ਰੋਟੀ ਖਾਧੀ।"), ("pa-IN", "ਮੁੰਡੇ ਨੇ ਰੋਟੀ ਖਾਧੀ।"),
])
def test_new_rules_leave_correct_sentences_alone(code, sentence):
    assert stage1.correct(sentence, code) == sentence


def test_synthetic_generator_is_deterministic_and_complete():
    from eval.synthetic_gec import generate
    a, b = generate(5), generate(5)
    assert a == b and len(a) == 1000
    assert all(it["wrong"] not in it["right"] for it in a)
