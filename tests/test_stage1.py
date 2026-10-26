import pytest

from app.stage1 import correct, normalize

CASES = [
    ("en-US", "i go to park yesterday", "I went to the park yesterday."),
    ("en-US", "i have went to school", "I have gone to school."),
    ("en-US", "she don't like apples", "She doesn't like apples."),
    ("en-US", "the informations is correct", "The information is correct."),
    ("en-US", "i could of done it", "I could have done it."),
    ("en-US", "he have a apple and a hour", "He has an apple and an hour."),
    ("en-US", "we was late, they has a car", "We were late, they have a car."),
    ("en-US", "the the cat is very very cute", "The cat is very cute."),
    ("en-US", "this is more better", "This is better."),
    ("en-US", "what is your name", "What is your name?"),
    ("en-US", "she doesn't knows what I was talking about", "She doesn't know what I was talking about."),
    ("en-US", "he didn't went to school", "He didn't go to school."),
    ("en-US", "she can speaks English", "She can speak English."),
    ("en-US", "I will always remembers this", "I will always remember this."),
    ("en-US", "I didn't liked it", "I didn't like it."),
    ("en-US", "she was looking at me like she knowed me, but I never seen her before in my life",
     "She was looking at me like she knew me, but I have never seen her before in my life."),
    ("en-US", "I seen it yesterday", "I saw it yesterday."),
    ("en-US", "he already eaten", "He has already eaten."),
    ("en-US", "we buyed a new car", "We bought a new car."),
    ("en-US", "I have knowed him for years", "I have known him for years."),
    ("hi-IN", "मैं जाता है", "मैं जाता हूँ।"),
    ("hi-IN", "मैं कल विद्यालय गया है", "मैं कल विद्यालय गया था।"),
    ("hi-IN", "हम जाता है", "हम जाते हैं।"),
    ("hi-IN", "वह खाना खाया", "उसने खाना खाया।"),
    ("mr-IN", "मी जातो आहे", "मी जात आहे।"),
    ("mr-IN", "मी काल शाळेत जातो", "मी काल शाळेत गेलो।"),
    ("mr-IN", "आम्ही जातो आहे", "आम्ही जात आहोत।"),
    ("bn-IN", "আমি যাচ্ছি হয়", "আমি যাচ্ছি।"),
    ("bn-IN", "আমি গতকাল স্কুলে যাই", "আমি গতকাল স্কুলে গেলাম।"),
    ("bn-IN", "আমি স্কুলে গিয়েছে", "আমি স্কুলে গিয়েছি।"),
    ("bn-IN", "সে কাজ করি", "সে কাজ করে।"),
    ("pa-IN", "ਮੈਂ ਜਾਂਦਾ ਹੈ", "ਮੈਂ ਜਾਂਦਾ ਹਾਂ।"),
    ("pa-IN", "ਮੈਂ ਕੱਲ੍ਹ ਸਕੂਲ ਗਿਆ ਹੈ", "ਮੈਂ ਕੱਲ੍ਹ ਸਕੂਲ ਗਿਆ ਸੀ।"),
    ("pa-IN", "ਅਸੀਂ ਜਾਂਦਾ ਹੈ", "ਅਸੀਂ ਜਾਂਦੇ ਹਾਂ।"),
    ("pa-IN", "ਉਹ ਰੋਟੀ ਖਾਧੀ", "ਉਸਨੇ ਰੋਟੀ ਖਾਧੀ।"),
]


@pytest.mark.parametrize("lang,raw,expected", CASES)
def test_rules_fix_common_asr_errors(lang, raw, expected):
    assert correct(raw, lang) == expected


@pytest.mark.parametrize("lang,raw,expected", CASES)
def test_rules_are_idempotent(lang, raw, expected):
    assert correct(expected, lang) == expected


@pytest.mark.parametrize("lang,text", [
    ("en-US", "I am going to Delhi."),          # original inserted "the" -> "the Delhi"
    ("en-US", "If I were you I would go."),     # subjunctive "were" must stay
    ("en-US", "Let it go."),                    # "it" is an object here, not a subject
    ("en-US", "I went to school yesterday."),   # no article before school
    ("en-US", "This does its job."),            # "its" after "does" is not a verb
    ("en-US", "What does she want?"),
    ("en-US", "The film was seen by many."),     # passive participle, not a missing "have"
    ("en-US", "She has already eaten."),
    ("hi-IN", "मैं कल जाऊँगा।"),                 # कल = tomorrow here, future must stay
    ("mr-IN", "तो जात आहे।"),                   # original rule broke this correct sentence
    ("bn-IN", "আমি কাল স্কুলে যাই।"),            # কাল is ambiguous, original forced past tense
])
def test_rules_leave_correct_sentences_alone(lang, text):
    assert correct(text, lang) == text


def test_normalize_only_fixes_spacing_case_and_end_mark():
    assert normalize("i  go to park", "en-US") == "I go to park."
    assert normalize("मैं जाता है", "hi-IN") == "मैं जाता है।"


def test_unknown_language_is_an_error_not_a_silent_fallback():
    with pytest.raises(ValueError):
        correct("hello", "xx-XX")


def test_third_person_agreement_for_common_verbs():
    from app import stage1
    assert stage1.correct("She was looking at me. She know me.", "en-US") == "She was looking at me. She knows me."
    assert stage1.correct("He try his best.", "en-US") == "He tries his best."
    assert stage1.correct("Does she know me?", "en-US") == "Does she know me?"      # after does: base form is right
    assert stage1.correct("I made her study.", "en-US") == "I made her study."        # not a subject
    assert stage1.correct("She knows me.", "en-US") == "She knows me."


def test_indic_subject_rules_skip_long_multi_clause_sentences():
    from app import stage1
    # real IndicGEC sentence: the last verb belongs to the second clause (आग ... है), not to मैं
    s = "मैं जब भी उसके सफर का सोचता हूं तो मुझमें एक नई आग दौड़ पड़ती है।"
    assert stage1.correct(s, "hi-IN") == s
    assert stage1.correct("मैं कल बाजार गया हूँ।", "hi-IN") == "मैं कल बाजार गया था।"   # simple clause still fixed


def test_bengali_reduplication_is_not_a_stutter():
    from app import stage1
    s = "কলমের নিব খোলা থেকে থেকে কালি শুকিয়ে গেছে।"
    assert stage1.correct(s, "bn-IN") == s
