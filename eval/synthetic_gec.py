"""1000 generated grammar mistakes, 200 per language, through the whole app (rules + LLM, as on the website).

    python -m eval.synthetic_gec                 # seed 1: the frozen test set
    python -m eval.synthetic_gec --seed 2        # a fresh set (to re-test after any change)
    python -m eval.synthetic_gec --no-llm        # rules only (fast, no Ollama)

Public test sets with real mistakes exist only for English (JFLEG) and Hindi/Bengali (IndicGEC); Marathi and Punjabi
have none. So sentences are built from hand-written grammar tables (subjects with person/number/gender, verbs in
four tenses, places, time words), and one typical mistake is put into each: wrong person, number or gender
agreement, a tense that doesn't fit the time word, a missing ergative ने/ਨੇ, the progressive form, and for English
-s, didn't + past, can + -s, have/has, was/were, a/an and "could of". Some of these the rules target, some they
don't (e.g. Hindi future, noun subjects with ने, Punjabi "last week", Bengali "tomorrow").

Each mistake sentence is scored: fixed (the output is one of the correct versions), unchanged (left as it was) or
wrong (changed into something else). Each correct version also goes through the app: changing it is a false alarm.
Writes eval/results/synthetic.md and eval/data_synth/seed<N>.jsonl.
"""

import argparse
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR, DATA_DIR = HERE / "results", HERE / "data_synth"
PER_LANGUAGE = 200


def N(s: str) -> str:
    return unicodedata.normalize("NFC", s)


# --------------------------------------------------------------------------- English

EN_THIRD = ["he", "she", "my brother", "my sister", "the manager"]
EN_OTHER = ["I", "you", "we", "they", "the children"]
EN_PLURAL = ["we", "they", "the children", "you"]
# base, 3rd person, past, past participle, complement
EN_VERBS = [("go", "goes", "went", "gone", "to the office"), ("go", "goes", "went", "gone", "to the market"),
            ("work", "works", "worked", "worked", "at the bank"), ("play", "plays", "played", "played", "football"),
            ("eat", "eats", "ate", "eaten", "rice"), ("watch", "watches", "watched", "watched", "the news"),
            ("write", "writes", "wrote", "written", "letters"), ("buy", "buys", "bought", "bought", "vegetables"),
            ("take", "takes", "took", "taken", "the bus"), ("like", "likes", "liked", "liked", "coffee")]
EN_VOWEL_NOUNS = ["apple", "orange", "umbrella", "egg", "onion"]
EN_CONSONANT_NOUNS = ["car", "book", "pen", "bag", "phone"]


def _cap(s: str) -> str:
    return s[0].upper() + s[1:] + "."


def english(rng: random.Random) -> tuple[str, list[str], str]:
    kind = rng.choice(["3rd-person -s", "plural with -s", "past with yesterday", "didn't + past", "modal + -s",
                       "have/has", "was/were", "a/an", "could of", "don't/doesn't"])
    base, third, past, pp, comp = rng.choice(EN_VERBS)
    s3, so, sp = rng.choice(EN_THIRD), rng.choice(EN_OTHER), rng.choice(EN_PLURAL)
    if kind == "3rd-person -s":
        return _cap(f"{s3} {base} {comp} every day"), [_cap(f"{s3} {third} {comp} every day")], kind
    if kind == "plural with -s":
        return _cap(f"{sp} {third} {comp} every day"), [_cap(f"{sp} {base} {comp} every day")], kind
    if kind == "past with yesterday":
        subj = rng.choice(EN_THIRD + EN_OTHER)
        form = third if subj in EN_THIRD else base
        return _cap(f"{subj} {form} {comp} yesterday"), [_cap(f"{subj} {past} {comp} yesterday")], kind
    if kind == "didn't + past":
        subj = rng.choice(EN_THIRD + EN_OTHER)
        return _cap(f"{subj} didn't {past} {comp}"), [_cap(f"{subj} didn't {base} {comp}"),
                                                       _cap(f"{subj} did not {base} {comp}")], kind
    if kind == "modal + -s":
        modal = rng.choice(["can", "should", "must", "will"])
        return _cap(f"{s3} {modal} {third} {comp}"), [_cap(f"{s3} {modal} {base} {comp}")], kind
    if kind == "have/has":
        if rng.random() < 0.5:
            return _cap(f"{s3} have {pp} {comp}"), [_cap(f"{s3} has {pp} {comp}")], kind
        return _cap(f"{sp} has {pp} {comp}"), [_cap(f"{sp} have {pp} {comp}")], kind
    if kind == "was/were":
        place = rng.choice(["late", "at home", "tired", "in the office"])
        if rng.random() < 0.5:
            return _cap(f"{sp} was {place} yesterday"), [_cap(f"{sp} were {place} yesterday")], kind
        subj = rng.choice(EN_THIRD + ["I"])
        return _cap(f"{subj} were {place} yesterday"), [_cap(f"{subj} was {place} yesterday")], kind
    if kind == "a/an":
        subj = rng.choice(EN_THIRD + EN_OTHER)
        if rng.random() < 0.5:
            noun = rng.choice(EN_VOWEL_NOUNS)
            return _cap(f"{subj} bought a {noun}"), [_cap(f"{subj} bought an {noun}")], kind
        noun = rng.choice(EN_CONSONANT_NOUNS)
        return _cap(f"{subj} bought an {noun}"), [_cap(f"{subj} bought a {noun}")], kind
    if kind == "could of":
        modal = rng.choice(["could", "should", "would"])
        subj = rng.choice(EN_THIRD + EN_OTHER)
        return _cap(f"{subj} {modal} of {pp} {comp}"), [_cap(f"{subj} {modal} have {pp} {comp}"),
                                                       _cap(f"{subj} {modal}'ve {pp} {comp}")], kind
    obj = rng.choice(["coffee", "spicy food", "cricket", "this song"])
    return _cap(f"{s3} don't like {obj}"), [_cap(f"{s3} doesn't like {obj}"), _cap(f"{s3} does not like {obj}")], kind


# --------------------------------------------------------------------------- Hindi

# text, ergative form, person (1/2/h/3), number, gender
HI_SUBJECTS = [("मैं", "मैंने", "1", "sg", "m"), ("हम", "हमने", "1", "pl", "m"), ("तुम", "तुमने", "2", "pl", "m"),
               ("आप", "आपने", "h", "pl", "m"), ("लड़का", "लड़के ने", "3", "sg", "m"),
               ("लड़की", "लड़की ने", "3", "sg", "f"), ("मेरा भाई", "मेरे भाई ने", "3", "sg", "m"),
               ("मेरी बहन", "मेरी बहन ने", "3", "sg", "f"), ("बच्चे", "बच्चों ने", "3", "pl", "m"),
               ("लड़कियाँ", "लड़कियों ने", "3", "pl", "f")]
HI_PERFECTIVE = {"जा": ("गया", "गई", "गए", "गईं"), "आ": ("आया", "आई", "आए", "आईं"), "सो": ("सोया", "सोई", "सोए", "सोईं"),
                 "बैठ": ("बैठा", "बैठी", "बैठे", "बैठीं")}
HI_PLACE = {"जा": ["स्कूल", "बाज़ार", "दफ़्तर", "मंदिर"], "आ": ["घर"], "सो": ["जल्दी"], "बैठ": ["कुर्सी पर"]}
HI_PAST_TIME = {"जा": "पिछले हफ़्ते", "आ": "पिछले हफ़्ते", "सो": "कल रात", "बैठ": "कल शाम"}


def _hi_aux(p, n):
    return {"1sg": "हूँ", "1pl": "हैं", "2pl": "हो", "hpl": "हैं", "3sg": "है", "3pl": "हैं"}[p + n]


def _hi_idx(n, g):
    return {"sgm": 0, "sgf": 1, "plm": 2, "plf": 3}[n + g]


def _hi_imperf(stem, n, g):
    return stem + ["ता", "ती", "ते", "ती"][_hi_idx(n, g)]


def _hi_prog(stem, n, g):
    return stem + " " + ["रहा", "रही", "रहे", "रही"][_hi_idx(n, g)]


def _hi_future(stem, p, n, g):
    key = "1sg" if p + n == "1sg" else "2pl" if p + n == "2pl" else "3sg" if p + n == "3sg" else "pl"
    end = {"1sg": ("ऊँगा", "ऊँगी"), "2pl": ("ओगे", "ओगी"), "3sg": ("एगा", "एगी"), "pl": ("एँगे", "एँगी")}[key]
    return stem + end[g == "f"]


def hindi(rng: random.Random) -> tuple[str, list[str], str]:
    kind = rng.choice(["person (aux)", "number", "gender", "tense with time word", "ergative ने"])
    stem = rng.choice(list(HI_PERFECTIVE))
    place = rng.choice(HI_PLACE[stem])
    if kind == "ergative ने":
        s, erg, *_ = rng.choice(HI_SUBJECTS)
        verb, obj, form = rng.choice([("खा", "रोटी", "खाई"), ("खा", "आम", "खाया"), ("पढ़", "किताब", "पढ़ी"),
                                      ("देख", "फ़िल्म", "देखी"), ("खा", "खाना", "खाया")])
        return f"{s} {obj} {form}।", [f"{erg} {obj} {form}।"], kind
    if kind == "person (aux)":
        s, _, p, n, g = rng.choice([x for x in HI_SUBJECTS if _hi_aux(x[2], x[3]) != "है"])
        verb = rng.choice([_hi_imperf(stem, n, g), _hi_prog(stem, n, g)])
        time = "रोज़" if verb.endswith(("ता", "ती", "ते")) else "अभी"
        return f"{s} {time} {place} {verb} है।", [f"{s} {time} {place} {verb} {_hi_aux(p, n)}।"], kind
    if kind == "number":
        s, _, p, n, g = rng.choice([x for x in HI_SUBJECTS if x[3] == "pl" and x[2] == "3"])
        return f"{s} रोज़ {place} {stem}ता है।", [f"{s} रोज़ {place} {_hi_imperf(stem, n, g)} {_hi_aux(p, n)}।"], kind
    if kind == "gender":
        s, _, p, n, g = rng.choice([x for x in HI_SUBJECTS if x[2] == "3" and x[3] == "sg"])
        other = "m" if g == "f" else "f"
        tense = rng.choice(["habitual", "past", "future"]) if stem in ("जा", "आ", "सो") else rng.choice(["habitual", "past"])
        if tense == "habitual":
            return f"{s} रोज़ {place} {_hi_imperf(stem, n, other)} है।", [f"{s} रोज़ {place} {_hi_imperf(stem, n, g)} है।"], kind
        if tense == "past":
            time = HI_PAST_TIME[stem]
            return (f"{s} {time} {place} {HI_PERFECTIVE[stem][_hi_idx(n, other)]}।",
                    [f"{s} {time} {place} {HI_PERFECTIVE[stem][_hi_idx(n, g)]}।"], kind)
        return (f"{s} अगले हफ़्ते {place} {_hi_future(stem, p, n, other)}।",
                [f"{s} अगले हफ़्ते {place} {_hi_future(stem, p, n, g)}।"], kind)
    s, _, p, n, g = rng.choice(HI_SUBJECTS)
    time = HI_PAST_TIME[stem]
    past = HI_PERFECTIVE[stem][_hi_idx(n, g)]
    past_aux = ["था", "थी", "थे", "थीं"][_hi_idx(n, g)]
    return (f"{s} {time} {place} {_hi_imperf(stem, n, g)} {_hi_aux(p, n)}।",
            [f"{s} {time} {place} {past}।", f"{s} {time} {place} {past} {past_aux}।"], kind)


# --------------------------------------------------------------------------- Marathi

# text, person, number, gender
MR_SUBJECTS = [("मी", "1", "sg", "m"), ("आम्ही", "1", "pl", "m"), ("तू", "2", "sg", "m"), ("तुम्ही", "2", "pl", "m"),
               ("तो", "3", "sg", "m"), ("ती", "3", "sg", "f"), ("मुलगा", "3", "sg", "m"), ("मुलगी", "3", "sg", "f"),
               ("माझा भाऊ", "3", "sg", "m"), ("माझी बहीण", "3", "sg", "f"), ("ते", "3", "pl", "m")]
MR_PLACE = {"जा": ["शाळेत", "बाजारात", "ऑफिसला"], "ये": ["घरी"], "खेळ": ["मैदानावर"], "झोप": ["लवकर"]}
MR_PAST_STEM = {"जा": "गेल", "ये": "आल", "खेळ": "खेळल", "झोप": "झोपल"}
MR_FUTURE = {"जा": {"1sg": "जाईन", "1pl": "जाऊ", "2sg": "जाशील", "2pl": "जाल", "3sg": "जाईल", "3pl": "जातील"},
             "ये": {"1sg": "येईन", "1pl": "येऊ", "2sg": "येशील", "2pl": "याल", "3sg": "येईल", "3pl": "येतील"}}


def _mr_present(stem, p, n, g):
    end = {"1sg": "तो", "1pl": "तो", "2sg": "तोस", "2pl": "ता", "3pl": "तात"}.get(p + n, "ते" if g == "f" else "तो")
    return stem + end


def _mr_past(stem, p, n, g):
    end = {"1sg": "ो", "1pl": "ो", "2sg": "ास", "2pl": "ात", "3pl": "े"}.get(p + n, "ी" if g == "f" else "ा")
    return MR_PAST_STEM[stem] + end


def _mr_aux(p, n):
    return {"1sg": "आहे", "1pl": "आहोत", "2sg": "आहेस", "2pl": "आहात", "3sg": "आहे", "3pl": "आहेत"}[p + n]


def marathi(rng: random.Random) -> tuple[str, list[str], str]:
    kind = rng.choice(["person", "gender", "tense with time word", "progressive form"])
    stem = rng.choice(list(MR_PLACE))
    place = rng.choice(MR_PLACE[stem])
    if kind == "person":
        s, p, n, g = rng.choice([x for x in MR_SUBJECTS if x[1] + x[2] in ("2sg", "2pl", "3pl", "1pl")])
        if rng.random() < 0.5 or p + n == "1pl":
            # progressive with the wrong auxiliary: आम्ही जात आहे -> आहोत
            return f"{s} आत्ता {place} {stem}त आहे।", [f"{s} आत्ता {place} {stem}त {_mr_aux(p, n)}।"], kind
        return f"{s} रोज {place} {stem}तो।", [f"{s} रोज {place} {_mr_present(stem, p, n, g)}।"], kind
    if kind == "gender":
        s, p, n, g = rng.choice([x for x in MR_SUBJECTS if x[1] == "3" and x[2] == "sg"])
        other = "m" if g == "f" else "f"
        if rng.random() < 0.5:
            return (f"{s} रोज {place} {_mr_present(stem, p, n, other)}।",
                    [f"{s} रोज {place} {_mr_present(stem, p, n, g)}।"], kind)
        return f"{s} काल {place} {_mr_past(stem, p, n, other)}।", [f"{s} काल {place} {_mr_past(stem, p, n, g)}।"], kind
    if kind == "progressive form":
        s, p, n, g = rng.choice([x for x in MR_SUBJECTS if x[1] + x[2] in ("1sg", "3sg")])
        wrong_verb = _mr_present(stem, p, n, g)
        return f"{s} आत्ता {place} {wrong_verb} आहे।", [f"{s} आत्ता {place} {stem}त आहे।"], kind
    s, p, n, g = rng.choice(MR_SUBJECTS)
    if stem in MR_FUTURE and rng.random() < 0.5:     # उद्या (tomorrow) with the past -> future
        return (f"{s} उद्या {place} {_mr_past(stem, p, n, g)}।", [f"{s} उद्या {place} {MR_FUTURE[stem][p + n]}।"], kind)
    return f"{s} काल {place} {_mr_present(stem, p, n, g)}।", [f"{s} काल {place} {_mr_past(stem, p, n, g)}।"], kind


# --------------------------------------------------------------------------- Bengali

BN_SUBJECTS = [("আমি", "1"), ("আমরা", "1"), ("তুমি", "2"), ("আপনি", "h"), ("সে", "3"), ("ও", "3"),
               ("আমার ভাই", "3"), ("আমার বোন", "3"), ("তারা", "3"), ("ছেলেরা", "3")]
# present, continuous, past, future: forms for persons 1, 2, h, 3
BN_VERBS = {
    "যা": (("যাই", "যাও", "যান", "যায়"), ("যাচ্ছি", "যাচ্ছ", "যাচ্ছেন", "যাচ্ছে"), ("গেলাম", "গেলে", "গেলেন", "গেল"),
           ("যাব", "যাবে", "যাবেন", "যাবে"), ["স্কুলে", "বাজারে", "অফিসে"]),
    "খা": (("খাই", "খাও", "খান", "খায়"), ("খাচ্ছি", "খাচ্ছ", "খাচ্ছেন", "খাচ্ছে"), ("খেলাম", "খেলে", "খেলেন", "খেল"),
           ("খাব", "খাবে", "খাবেন", "খাবে"), ["ভাত", "রুটি"]),
    "কর": (("করি", "করো", "করেন", "করে"), ("করছি", "করছ", "করছেন", "করছে"), ("করলাম", "করলে", "করলেন", "করল"),
           ("করব", "করবে", "করবেন", "করবে"), ["কাজ"]),
    "পড়": (("পড়ি", "পড়ো", "পড়েন", "পড়ে"), ("পড়ছি", "পড়ছ", "পড়ছেন", "পড়ছে"), ("পড়লাম", "পড়লে", "পড়লেন", "পড়ল"),
           ("পড়ব", "পড়বে", "পড়বেন", "পড়বে"), ["বই"]),
    "দেখ": (("দেখি", "দেখো", "দেখেন", "দেখে"), ("দেখছি", "দেখছ", "দেখছেন", "দেখছে"), ("দেখলাম", "দেখলে", "দেখলেন", "দেখল"),
            ("দেখব", "দেখবে", "দেখবেন", "দেখবে"), ["সিনেমা", "টিভি"]),
}
BN_PERSON = {"1": 0, "2": 1, "h": 2, "3": 3}
BN_WRONG_PERSON = {"1": "3", "2": "1", "h": "3", "3": "1"}     # the typical mix-up for each person


def bengali(rng: random.Random) -> tuple[str, list[str], str]:
    kind = rng.choice(["person (present)", "person (continuous)", "tense with time word"])
    verb = rng.choice(list(BN_VERBS))
    pres, cont, past, fut, comps = BN_VERBS[verb]
    comp = rng.choice(comps)
    s, p = rng.choice(BN_SUBJECTS)
    i, w = BN_PERSON[p], BN_PERSON[BN_WRONG_PERSON[p]]
    if kind == "person (present)":
        return f"{s} প্রতিদিন {comp} {pres[w]}।", [f"{s} প্রতিদিন {comp} {pres[i]}।"], kind
    if kind == "person (continuous)":
        return f"{s} এখন {comp} {cont[w]}।", [f"{s} এখন {comp} {cont[i]}।"], kind
    # (তুমি + আগামীকাল + past is left out: "তুমি আগামীকাল অফিসে গেলে" also means "if you go to the office tomorrow")
    if rng.random() < 0.5 or p == "2":
        return f"{s} গতকাল {comp} {pres[i]}।", [f"{s} গতকাল {comp} {past[i]}।"], kind
    return f"{s} আগামীকাল {comp} {past[i]}।", [f"{s} আগামীকাল {comp} {fut[i]}।"], kind


# --------------------------------------------------------------------------- Punjabi

PA_SUBJECTS = [("ਮੈਂ", "1", "sg", "m"), ("ਅਸੀਂ", "1", "pl", "m"), ("ਤੁਸੀਂ", "2", "pl", "m"), ("ਮੁੰਡਾ", "3", "sg", "m"),
               ("ਕੁੜੀ", "3", "sg", "f"), ("ਮੇਰਾ ਭਰਾ", "3", "sg", "m"), ("ਮੇਰੀ ਭੈਣ", "3", "sg", "f"),
               ("ਮੇਰਾ ਦੋਸਤ", "3", "sg", "m"), ("ਮੇਰੀ ਸਹੇਲੀ", "3", "sg", "f"),
               ("ਬੱਚੇ", "3", "pl", "m"), ("ਕੁੜੀਆਂ", "3", "pl", "f")]
PA_ERGATIVE = {"ਮੁੰਡਾ": "ਮੁੰਡੇ ਨੇ", "ਕੁੜੀ": "ਕੁੜੀ ਨੇ", "ਮੇਰਾ ਭਰਾ": "ਮੇਰੇ ਭਰਾ ਨੇ", "ਮੇਰੀ ਭੈਣ": "ਮੇਰੀ ਭੈਣ ਨੇ",
               "ਮੇਰਾ ਦੋਸਤ": "ਮੇਰੇ ਦੋਸਤ ਨੇ", "ਮੇਰੀ ਸਹੇਲੀ": "ਮੇਰੀ ਸਹੇਲੀ ਨੇ"}
# imperfective stem, progressive root, perfective forms (m sg, f sg, m pl, f pl), places
PA_VERBS = {"ਜਾ": ("ਜਾਂ", "ਜਾ", ("ਗਿਆ", "ਗਈ", "ਗਏ", "ਗਈਆਂ"), ["ਸਕੂਲ", "ਬਾਜ਼ਾਰ", "ਦਫ਼ਤਰ", "ਗੁਰਦੁਆਰੇ", "ਦਿੱਲੀ"]),
            "ਆ": ("ਆਉਂ", "ਆ", ("ਆਇਆ", "ਆਈ", "ਆਏ", "ਆਈਆਂ"), ["ਘਰ", "ਦਫ਼ਤਰ ਤੋਂ"]),
            "ਸੌਂ": ("ਸੌਂ", "ਸੌਂ", ("ਸੁੱਤਾ", "ਸੁੱਤੀ", "ਸੁੱਤੇ", "ਸੁੱਤੀਆਂ"), ["ਜਲਦੀ"]),
            "ਖੇਡ": ("ਖੇਡ", "ਖੇਡ", None, ["ਪਾਰਕ ਵਿੱਚ", "ਮੈਦਾਨ ਵਿੱਚ"])}
PA_FUTURE = {"ਜਾ": {"1sg": ("ਜਾਵਾਂਗਾ", "ਜਾਵਾਂਗੀ"), "1pl": ("ਜਾਵਾਂਗੇ", "ਜਾਵਾਂਗੀਆਂ"), "2pl": ("ਜਾਓਗੇ", "ਜਾਓਗੀਆਂ"),
                     "3sg": ("ਜਾਵੇਗਾ", "ਜਾਵੇਗੀ"), "3pl": ("ਜਾਣਗੇ", "ਜਾਣਗੀਆਂ")},
             "ਆ": {"1sg": ("ਆਵਾਂਗਾ", "ਆਵਾਂਗੀ"), "1pl": ("ਆਵਾਂਗੇ", "ਆਵਾਂਗੀਆਂ"), "2pl": ("ਆਓਗੇ", "ਆਓਗੀਆਂ"),
                    "3sg": ("ਆਵੇਗਾ", "ਆਵੇਗੀ"), "3pl": ("ਆਉਣਗੇ", "ਆਉਣਗੀਆਂ")}}


def _pa_idx(n, g):
    return {"sgm": 0, "sgf": 1, "plm": 2, "plf": 3}[n + g]


def _pa_aux(p, n):
    return {"1sg": "ਹਾਂ", "1pl": "ਹਾਂ", "2pl": "ਹੋ", "3sg": "ਹੈ", "3pl": "ਹਨ"}[p + n]


def _pa_imperf(stem, n, g):
    return stem + ["ਦਾ", "ਦੀ", "ਦੇ", "ਦੀਆਂ"][_pa_idx(n, g)]


def punjabi(rng: random.Random) -> tuple[str, list[str], str]:
    kind = rng.choice(["person (aux)", "number", "gender", "tense with time word", "ergative ਨੇ"])
    verb = rng.choice(list(PA_VERBS))
    imp, root, perf, places = PA_VERBS[verb]
    place = rng.choice(places)
    if kind == "ergative ਨੇ":
        s = rng.choice(list(PA_ERGATIVE))
        obj, form = rng.choice([("ਰੋਟੀ", "ਖਾਧੀ"), ("ਖਾਣਾ", "ਖਾਧਾ"), ("ਕਿਤਾਬ", "ਪੜ੍ਹੀ"), ("ਫ਼ਿਲਮ", "ਵੇਖੀ"),
                                ("ਚਾਹ", "ਪੀਤੀ"), ("ਚਿੱਠੀ", "ਲਿਖੀ"), ("ਅਖ਼ਬਾਰ", "ਪੜ੍ਹਿਆ")])
        return f"{s} {obj} {form}।", [f"{PA_ERGATIVE[s]} {obj} {form}।"], kind
    if kind == "person (aux)":
        s, p, n, g = rng.choice([x for x in PA_SUBJECTS if x[1] in ("1", "2")])
        return (f"{s} ਰੋਜ਼ {place} {_pa_imperf(imp, n, g)} ਹੈ।",
                [f"{s} ਰੋਜ਼ {place} {_pa_imperf(imp, n, g)} {_pa_aux(p, n)}।"], kind)
    if kind == "number":
        s, p, n, g = rng.choice([x for x in PA_SUBJECTS if x[2] == "pl"])
        return (f"{s} ਰੋਜ਼ {place} {imp}ਦਾ ਹੈ।", [f"{s} ਰੋਜ਼ {place} {_pa_imperf(imp, n, g)} {_pa_aux(p, n)}।"], kind)
    if kind == "gender":
        s, p, n, g = rng.choice([x for x in PA_SUBJECTS if x[1] == "3" and x[2] == "sg"])
        other = "m" if g == "f" else "f"
        choice = rng.choice(["habitual", "past", "future"] if verb in PA_FUTURE else ["habitual", "past"]) \
            if perf else "habitual"
        if choice == "habitual":
            return (f"{s} ਰੋਜ਼ {place} {_pa_imperf(imp, n, other)} ਹੈ।",
                    [f"{s} ਰੋਜ਼ {place} {_pa_imperf(imp, n, g)} ਹੈ।"], kind)
        if choice == "past":
            return (f"{s} ਪਿਛਲੇ ਹਫ਼ਤੇ {place} {perf[_pa_idx(n, other)]} ਸੀ।",
                    [f"{s} ਪਿਛਲੇ ਹਫ਼ਤੇ {place} {perf[_pa_idx(n, g)]} ਸੀ।"], kind)
        fut = PA_FUTURE[verb]["3sg"]
        return (f"{s} ਅਗਲੇ ਹਫ਼ਤੇ {place} {fut[other == 'f']}।", [f"{s} ਅਗਲੇ ਹਫ਼ਤੇ {place} {fut[g == 'f']}।"], kind)
    if not perf:
        verb = "ਜਾ"
        imp, root, perf, places = PA_VERBS[verb]
        place = rng.choice(places)
    s, p, n, g = rng.choice(PA_SUBJECTS)
    past = perf[_pa_idx(n, g)]
    past_aux = ["ਸਨ", "ਸੀ"] if p == "3" and n == "pl" else ["ਸੀ"]      # ਅਸੀਂ/ਤੁਸੀਂ ... ਗਏ ਸੀ
    return (f"{s} ਪਿਛਲੇ ਹਫ਼ਤੇ {place} {_pa_imperf(imp, n, g)} {_pa_aux(p, n)}।",
            [f"{s} ਪਿਛਲੇ ਹਫ਼ਤੇ {place} {past}।"] + [f"{s} ਪਿਛਲੇ ਹਫ਼ਤੇ {place} {past} {a}।" for a in past_aux], kind)


GENERATORS = {"en-US": english, "hi-IN": hindi, "mr-IN": marathi, "bn-IN": bengali, "pa-IN": punjabi}


def generate(seed: int) -> list[dict]:
    rng = random.Random(seed)
    items = []
    for code, make in GENERATORS.items():
        seen = set()
        tries = 0
        while len(seen) < PER_LANGUAGE and tries < 20000:
            tries += 1
            wrong, rights, kind = make(rng)
            wrong, rights = N(wrong), [N(r) for r in rights]
            if wrong in seen or wrong in rights:
                continue
            seen.add(wrong)
            items.append({"language": code, "wrong": wrong, "right": rights, "type": kind})
    return items


def norm(text: str) -> str:
    """Compare ignoring case, final punctuation, extra spaces and contractions."""
    t = unicodedata.normalize("NFC", unicodedata.normalize("NFD", N(text)).replace("\u093c", "").replace("\u0a3c", ""))
    t = t.strip().lower()   # the nukta dot is ignored: रोज and रोज़, हफ्ते and हफ़्ते are both standard
    for a, b in (("doesn't", "does not"), ("don't", "do not"), ("didn't", "did not"), ("'ve", " have")):
        t = t.replace(a, b)
    t = re.sub(r"[.।!?]+$", "", t)
    return re.sub(r"\s+", " ", t).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-llm", action="store_true", help="Stage 1 rules only")
    parser.add_argument("--per-lang", type=int, default=PER_LANGUAGE,
                        help="sentences per language (the free cloud tiers allow a few hundred calls a day; "
                             "each sentence is 2 calls: the mistake and its correct version)")
    args = parser.parse_args()

    from app import stage1
    from app.languages import LANGUAGES
    items = generate(args.seed)
    if args.per_lang < PER_LANGUAGE:   # the first N of each language, so smaller runs are subsets of the full one
        count = Counter()
        items = [it for it in items if (count.update([it["language"]]) or count[it["language"]] <= args.per_lang)]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_DIR / f"seed{args.seed}.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    if args.no_llm:
        correct = stage1.correct
    else:
        from app.pipeline import build_router
        from app import stage2
        router = build_router(stage2.default_client(wait_on_rate_limit=True))
        st = router.llm.status()
        if not st["online"] or not st["model_available"]:
            raise SystemExit(f"No grammar model is reachable ({st}). Check the keys in .env, open Ollama, "
                             f"or use --no-llm.")
        print(f"Grammar model: {st['model']}")
        correct = lambda text, code: router.process_text(text, code, 50)["final_text"]

    score = defaultdict(Counter)
    details = []
    for i, it in enumerate(items, 1):
        code = it["language"]
        out = correct(it["wrong"], code)
        if norm(out) in {norm(r) for r in it["right"]}:
            verdict = "fixed"
        elif norm(out) == norm(it["wrong"]):
            verdict = "unchanged"
        else:
            verdict = "wrong"
        kept = correct(it["right"][0], code)
        false_alarm = norm(kept) != norm(it["right"][0])
        score[(code, it["type"])][verdict] += 1
        score[(code, it["type"])]["false_alarm"] += false_alarm
        details.append({**it, "output": out, "verdict": verdict, "correct_output": kept, "false_alarm": false_alarm})
        print(f"  {i}/{len(items)}  {code}  {verdict:9s}  {it['wrong']}  ->  {out}", flush=True)

    mode = "Stage 1 rules only" if args.no_llm else f"the whole app (grammar model: {st['model']}), intensity 50"
    md = [f"# {len(items)} generated grammar mistakes (seed {args.seed}), {mode}", "",
          "Fixed = the output is a correct version; unchanged = left as it was; wrong = changed into something else. "
          "False alarms = correct versions the app changed.", "",
          "| Language | Fixed | Unchanged | Wrong | False alarms on correct sentences |", "|---|---|---|---|---|"]
    by_type = ["", "| Language | Mistake type | Sentences | Fixed | Unchanged | Wrong | False alarms |",
               "|---|---|---|---|---|---|---|"]
    total = Counter()
    for code in GENERATORS:
        lang = Counter()
        for (c, kind), cnt in sorted(score.items()):
            if c != code:
                continue
            lang.update(cnt)
            n = cnt["fixed"] + cnt["unchanged"] + cnt["wrong"]
            by_type.append(f"| {LANGUAGES[code].name} | {kind} | {n} | {cnt['fixed']} | {cnt['unchanged']} | "
                           f"{cnt['wrong']} | {cnt['false_alarm']} |")
        n = lang["fixed"] + lang["unchanged"] + lang["wrong"]
        total.update(lang)
        md.append(f"| {LANGUAGES[code].name} | {lang['fixed']}/{n} ({lang['fixed'] / n:.0%}) | {lang['unchanged']} | "
                  f"{lang['wrong']} | {lang['false_alarm']}/{n} |")
    n = total["fixed"] + total["unchanged"] + total["wrong"]
    md.append(f"| **All** | **{total['fixed']}/{n} ({total['fixed'] / n:.0%})** | {total['unchanged']} | {total['wrong']} | "
              f"{total['false_alarm']}/{n} |")
    md += by_type
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    name = f"synthetic_seed{args.seed}" + ("_rules" if args.no_llm else "") + \
        (f"_n{args.per_lang}" if args.per_lang < PER_LANGUAGE else "")
    (OUT_DIR / f"{name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    with open(OUT_DIR / f"{name}_details.jsonl", "w", encoding="utf-8") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print("\n".join(md))
    print(f"Saved {OUT_DIR / (name + '.md')}")


if __name__ == "__main__":
    main()
