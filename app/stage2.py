"""Stage 2: contextual correction with a local LLM (Ollama, default model llama3.2).

Fixes over the original:
- it really runs an LLM (the original's endpoint was dead and its "fallback" just re-ran Stage 1);
- the model's answer is checked before it is used (the guard below), because a small LLM can
  translate, rewrite the whole sentence or hallucinate. A rejected answer keeps the Stage 1 text;
- the status is reported honestly (applied / unchanged / rejected / skipped / unavailable);
- "correction intensity" controls Stage 2 for real: low = rules only, higher = the LLM may
  change more of the sentence;
- every single edit the LLM makes is checked (edit filter below). Measured on FLEURS, a small LLM
  swaps correct words for synonyms, invents words and drops numbers; those edits are reverted,
  while grammar fixes, close spelling/inflection fixes and words the recogniser itself offered as
  alternatives are kept.
"""

import json
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache

import httpx

from . import config
from .languages import get_language, script_ratio
from .text_utils import join_tokens, tokenize

PROMPT_VERSION = 3   # bump when the prompt changes (the evaluation re-applies filter changes to cached answers)

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"corrected": {"type": "string"}},
    "required": ["corrected"],
}


@dataclass
class Stage2Result:
    text: str            # text after Stage 2 (equals the input when not applied)
    status: str          # applied | unchanged | rejected | skipped | unavailable
    reason: str = ""
    model: str = ""
    raw_output: str = "" # what the LLM returned, kept for transparency
    kept_edits: int = 0
    reverted_edits: int = 0


# --------------------------------------------------------------------------- intensity

def correction_level(intensity: int) -> int:
    """What the correction intensity setting (0-100) allows.

    0  (0-33)   rules only, Stage 2 is skipped
    1  (34-79)  + LLM fixes grammar and misspelt words, and words the recogniser misheard as non-words
    2  (80-100) + LLM may also replace real words the recogniser was unsure about
    """
    intensity = max(0, min(100, int(intensity)))
    return 0 if intensity < 34 else 1 if intensity < 80 else 2


# --------------------------------------------------------------------------- guard

def word_edit_distance(a: list[str], b: list[str]) -> int:
    """Levenshtein distance between two token lists."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def _words(text: str) -> list[str]:
    return [t.lower() for t in tokenize(text) if any(ch.isalnum() for ch in t)]


def check_output(source: str, output: str, lang_code: str, max_ratio: float) -> str:
    """Return "" if the LLM output is acceptable, otherwise the reason it was rejected."""
    if not output.strip():
        return "empty answer"
    lang = get_language(lang_code)
    min_script = 0.9 if lang.script_range is None else 0.8
    if script_ratio(output, lang) < min_script:
        return f"answer is not in {lang.script} script (translated or mixed)"
    src, out = _words(source), _words(output)
    if not src:
        return "empty source"
    if not 0.5 <= len(out) / len(src) <= 1.6:
        return "answer length changed too much"
    ratio = word_edit_distance(src, out) / len(src)
    if ratio > max_ratio:
        return f"changed {ratio:.0%} of the words (limit {max_ratio:.0%} at this intensity)"
    return ""


# --------------------------------------------------------------------------- edit filter

def _n(words: list[str]) -> set[str]:
    return {unicodedata.normalize("NFC", w) for w in words}


# Grammar words: adding, removing or swapping these is what grammar correction is made of.
FUNCTION_WORDS = {
    "en-US": _n("a an the is are am was were be been being has have had do does did to of in on at for "
                "with by from and or but not it its this that these those will would can could shall should "
                "may might must there i me my mine you your yours he him his she her hers we us our ours "
                "they them their theirs who whom whose which what".split()),
    "hi-IN": _n("है हैं था थी थे थीं हूँ हूं हो का की के को में से ने पर पे तक और भी ही तो कि न ना नहीं व एवं "
                "मैं मुझे मेरा मेरी मेरे हम हमें हमारा हमारी हमारे तुम तुम्हें आप यह ये वह वे वो इस इन उस उन इसे उसे "
                "इसका इसकी इसके उसका उसकी उसके जो जिस जिन जब तब लिए द्वारा साथ बाद पहले रहा रही रहे गया गई गए".split()),
    "mr-IN": _n("आहे आहेत आहोत आहात होता होती होते ला ने चा ची चे च्या त आणि ही तर की".split()),
    "bn-IN": _n("হয় আছে আছি আছেন আছিস আছ ছিল ছিলাম ছিলেন ছিলে ছিলি থাকবে থাকব থাকবেন থাকিবে থাকিব থাকিবেন "
                "এবং ও কে এর থেকে তে যে না নি আর কি মধ্যে দিকে সঙ্গে সাথে দ্বারা জন্য উপরে "
                "আমি আমার আমাকে আমরা তুমি তোমার আপনি সে তার তাকে তারা যার যে সেই যেই এই ওই এটা সেটা".split()),
    "pa-IN": _n("ਹੈ ਹਨ ਸੀ ਸਨ ਹਾਂ ਹੋ ਦਾ ਦੀ ਦੇ ਨੂੰ ਵਿੱਚ ਤੋਂ ਨੇ ਅਤੇ ਵੀ ਹੀ ਤੇ ਕਿ".split()),
}
SIMILARITY = 0.7   # character similarity that counts as a spelling/inflection fix of the same word


def _is_word(token: str) -> bool:
    return any(ch.isalnum() for ch in token)


def _similar(a: str, b: str) -> bool:
    """Same word with a spelling or inflection fix (go -> goes, problem -> problems)?"""
    a, b = a.lower(), b.lower()
    ratio = SequenceMatcher(None, a, b, autojunk=False).ratio()
    short, long = sorted((a, b), key=len)
    return ratio >= SIMILARITY or (long.startswith(short) and ratio >= 0.6)


PUNCTUATION = set(".,!?;:।'\"()“”‘’")


EN_SUFFIXES = {"", "s", "es", "ed", "d", "ing"}


# Commonly confused words: choosing between them is a grammar/usage fix, not a word swap
CONFUSION_SETS = [{"their", "there", "they're"}, {"your", "you're"}, {"its", "it's"}, {"to", "too", "two"},
                  {"then", "than"}, {"were", "where", "we're"}, {"whose", "who's"}, {"affect", "effect"},
                  {"lose", "loose"}, {"accept", "except"}, {"a", "an"}, {"much", "many"}, {"less", "fewer"},
                  {"who", "whom"}, {"me", "i"}, {"him", "he"}, {"her", "she"}, {"them", "they"}, {"us", "we"},
                  {"this", "these"}, {"that", "those"}, {"since", "for"}, {"say", "tell"}, {"bring", "take"}]


def _english_inflection(a: str, b: str) -> bool:
    """Same English word in a different grammatical form (knows/know, liked/like, knowed/knew, their/there)."""
    from .stage1 import IRREGULAR_BASE, OVERREGULARISED
    a, b = a.lower(), b.lower()
    if IRREGULAR_BASE.get(a) == b or IRREGULAR_BASE.get(b) == a:
        return True
    if b in OVERREGULARISED.get(a, ()) or a in OVERREGULARISED.get(b, ()):
        return True
    if any(a in group and b in group for group in CONFUSION_SETS):
        return True
    short, long = sorted((a, b), key=len)
    if len(short) < 3:
        return False
    if short.endswith("e") and long.startswith(short[:-1]) and long[len(short) - 1:] in {"ed", "ing", "es"}:
        return True                                   # like -> liked / liking
    if short.endswith("y") and long.startswith(short[:-1]) and long[len(short) - 1:] in {"ies", "ied"}:
        return True                                   # study -> studies / studied
    if long.startswith(short) and long[len(short):] in EN_SUFFIXES:
        return True
    if _en_stem(a) == _en_stem(b):                    # farming -> farms, studying -> studies
        return True
    return _same_root(a, b)                           # success -> succeed, strange -> strangely


def _same_root(a: str, b: str) -> bool:
    """A different part of speech of the same word (wrong word class is a grammar error)."""
    p = 0
    while p < min(len(a), len(b)) and a[p] == b[p]:
        p += 1
    if p < max(4, 0.6 * min(len(a), len(b))) or min(len(a), len(b)) < 4:
        return False
    short, long = sorted((a, b), key=len)
    tail = long[len(short):]
    # through -> throughout, some -> something: a compound with another word, not a word-class change
    return not (long.startswith(short) and len(tail) >= 3 and _zipf(tail, "en-US") >= 4.0)


def _en_stem(word: str) -> str:
    for suffix in ("ing", "ies", "ied", "es", "ed", "s", "d"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[:-len(suffix)]
            break
    return word.rstrip("e").rstrip("y")


VOWEL_LETTER = re.compile(r"LETTER (A|AA|I|II|U|UU|E|EE|AI|O|OO|AU)$")


def _indic_inflection(a: str, b: str) -> bool:
    """Same Indic word with a different grammatical ending: करेगी/करेगा, जाता/जाते, যাই/যায়.
    The words must share their stem and differ only in final vowel signs (not in consonants)."""
    a, b = unicodedata.normalize("NFC", a), unicodedata.normalize("NFC", b)
    p = 0
    while p < min(len(a), len(b)) and a[p] == b[p]:
        p += 1
    tails = a[p:], b[p:]
    if p < 2 or max(len(t) for t in tails) > 2:
        return False
    return all(unicodedata.category(ch) in {"Mn", "Mc"} or VOWEL_LETTER.search(unicodedata.name(ch, ""))
               for t in tails for ch in t)


# Hindi verbs: tense/aspect/gender endings, and past forms that don't share the stem (जाता -> गया)
HI_VERB_ENDINGS = sorted((unicodedata.normalize("NFC", e) for e in
                          "ऊँगा ऊंगा ऊँगी ऊंगी ूँगा ूंगा ूँगी ूंगी एंगे एँगे ेंगे ेंगी एगा एगी ेगा ेगी ओगे ोगे "
                          "ता ती ते ना नी ने या यी ये ा ी े ीं ें ो ए ई".split()), key=len, reverse=True)
HI_SUPPLETIVE = {unicodedata.normalize("NFC", form): unicodedata.normalize("NFC", stem)
                 for stem, forms in {"जा": "गया गई गए गयी गये", "कर": "किया किए किये", "ले": "लिया ली लिए लिये",
                                     "दे": "दिया दी दिए दिये", "हो": "हुआ हुई हुए", "पी": "पिया"}.items()
                 for form in forms.split()}


def _hindi_verb_stem(word: str) -> str:
    if word in HI_SUPPLETIVE:
        return HI_SUPPLETIVE[word]
    for ending in HI_VERB_ENDINGS:
        if word.endswith(ending) and len(word) > len(ending):
            return word[:-len(ending)]
    return word


def _hindi_verb_form(a: str, b: str) -> bool:
    """Same Hindi verb in another tense/gender: जाता हूँ -> गया था, जाता -> जाऊँगा, करती -> किया."""
    a, b = unicodedata.normalize("NFC", a), unicodedata.normalize("NFC", b)
    stem = _hindi_verb_stem(a)
    return stem == _hindi_verb_stem(b) and (a in HI_SUPPLETIVE or b in HI_SUPPLETIVE or len(stem) >= 2
                                            or stem in HI_SUPPLETIVE.values())


_NASALS = set("ङञणनमঙঞণনমਙਞਣਨਮ")


def _skeleton(word: str) -> str:
    """The consonants of an Indic word, without vowels, nukta or candrabindu; any nasal written as an
    anusvara (परंतु) or as nasal + virama (परन्तु) becomes one symbol. मिडिया and मीडिया share one skeleton."""
    out, chars = [], unicodedata.normalize("NFD", word)
    for i, ch in enumerate(chars):
        name = unicodedata.name(ch, "")
        if ch in _NASALS and i + 1 < len(chars) and unicodedata.name(chars[i + 1], "").endswith("SIGN VIRAMA"):
            out.append("~")
        elif name.endswith("SIGN ANUSVARA"):
            out.append("~")
        elif " LETTER " in name and not VOWEL_LETTER.search(name) and "VOCALIC" not in name:
            out.append(ch)
    return "".join(out)


def _indic_spelling_variant(wrong: str, right: str, lang_code: str) -> bool:
    """A misspelling that differs only in vowels/nukta/nasal (प्रदुषण -> प्रदूषण, दुर -> दूर): same
    consonants, and the correction is clearly the more common spelling. Needs a word list, so only
    Hindi and Bengali; a common word is never "corrected" into another common word (कम -> काम)."""
    if lang_code not in ("hi-IN", "bn-IN") or _skeleton(wrong) != _skeleton(right) or not _skeleton(wrong):
        return False
    from wordfreq import zipf_frequency
    wf = WORDFREQ_LANG[lang_code]
    z_wrong, z_right = zipf_frequency(wrong, wf), zipf_frequency(right, wf)
    return z_wrong < 4.0 and z_right >= z_wrong + 0.5


BN_VERB_TAIL = set("তলবছমনয়") | {"য়"}


def _bengali_verb_form(a: str, b: str) -> bool:
    """Same Bengali verb with another ending: শুনো / শুনতে / শুনে / শোন / শুনলে."""
    sa, sb = _skeleton(a), _skeleton(b)
    p = 0
    while p < min(len(sa), len(sb)) and sa[p] == sb[p]:
        p += 1
    tails = sa[p:], sb[p:]
    return p >= 2 and all(len(t) <= 3 and set(t) <= BN_VERB_TAIL for t in tails)


def _inflection(a: str, b: str, lang_code: str) -> bool:
    if lang_code == "en-US":
        return _english_inflection(a, b)
    return (_indic_inflection(a, b) or (lang_code == "hi-IN" and _hindi_verb_form(a, b))
            or (lang_code == "bn-IN" and _bengali_verb_form(a, b)))


WORDFREQ_LANG = {"en-US": "en", "hi-IN": "hi", "bn-IN": "bn"}   # Marathi and Punjabi aren't in wordfreq


def _is_misspelling_of(wrong: str, right: str, lang_code: str) -> bool:
    """Is `wrong` a misspelt/non-existent form of `right`? (recieved/received, definately/definitely)

    Misspellings do appear in web text, so instead of "not in the dictionary" the test is relative:
    the two look alike, `wrong` is uncommon, and `right` is at least ~30x more common.
    Marathi and Punjabi aren't covered by wordfreq, so nothing is treated as a misspelling there.
    """
    wf = WORDFREQ_LANG.get(lang_code)
    if not wf or any(ch.isdigit() for ch in wrong):
        return False
    if lang_code != "en-US" and _indic_spelling_variant(wrong, right, lang_code):
        return True
    if not _similar(wrong, right):
        return False
    from wordfreq import zipf_frequency
    z_wrong, z_right = zipf_frequency(wrong.lower(), wf), zipf_frequency(right.lower(), wf)
    return z_wrong < 3.5 and z_right - z_wrong >= 1.5


# Spaced spellings that are always one word in standard English ("can not" -> "cannot")
CLOSED_COMPOUNDS = {"cannot", "without", "because", "something", "everything", "nothing", "anything", "someone",
                    "everybody", "somebody", "nobody", "anybody", "nowadays", "yourself", "myself", "itself",
                    "themselves", "himself", "herself", "ourselves", "whenever", "however", "whatever",
                    "wherever", "throughout", "freshman", "weekend", "homework", "classroom", "website"}
LETTER_WORDS = {"a", "i"}   # the only single letters that are English words


def _zipf(word: str, lang_code: str) -> float:
    wf = WORDFREQ_LANG.get(lang_code)
    if not wf:
        return 8.0                         # no word list: treat every word as real
    from wordfreq import zipf_frequency
    return zipf_frequency(word.lower(), wf)


def _is_real_word(word: str, lang_code: str) -> bool:
    if any(ch.isdigit() for ch in word):
        return word.isdigit()              # "12" is fine, "12years" is two words run together
    if lang_code == "en-US" and len(word) == 1:
        return word.lower() in LETTER_WORDS
    return _zipf(word, lang_code) >= 3.0


def _split_or_merge_ok(old_w: list[str], new_w: list[str], lang_code: str) -> bool:
    """English only: "Forexample" -> "For example", "12years" -> "12 years", "can not" -> "cannot".
    Elsewhere (e.g. Bengali compounds) splitting/merging changed correct words, so it is not allowed."""
    if lang_code != "en-US":
        return False
    if len(old_w) == 1:                    # split: the joined form must not be a real word
        return not _is_real_word(old_w[0], lang_code) and all(_is_real_word(w, lang_code) for w in new_w)
    if len(new_w) == 1:                    # merge: only into words that are never written apart
        return new_w[0].lower() in CLOSED_COMPOUNDS
    return False


def _edit_allowed(old: list[str], new: list[str], lang_code: str, heard: set[str] | None,
                  uncertain: set[str] | None = None, level: int = 1) -> bool:
    """Decide whether one LLM edit (old tokens -> new tokens) is allowed, in priority order.
    Every changed word must be explained by one of:

    1. grammar   - a grammar word added/removed/swapped, another form of the same word (knows/know,
                   farming/farms, करेगी/करेगा), a commonly confused word (their/there), or (English)
                   a wrongly split/joined word (Forexample -> For example)
    2. non-word  - a word that doesn't exist replaced by the real word it resembles (recieved -> received)
    3. unsure    - (last resort) a word the recogniser was unsure about: if it isn't a real word ("C"),
                   it may be replaced by a real word; at level 2, real words it was unsure about too
    (plus: a word the recogniser heard in one of its alternative hypotheses)
    Everything else - synonyms, rewrites, dropped numbers, added content - is reverted.
    """
    grammar = FUNCTION_WORDS[lang_code]
    old_w = [w for w in old if _is_word(w)]
    new_w = [w for w in new if _is_word(w)]
    if any(not _is_word(t) and t not in PUNCTUATION for t in new):
        return False                       # stray symbols, e.g. the "|" separator copied from the prompt
    if old_w and new_w and "".join(old_w).lower() == "".join(new_w).lower():
        return _split_or_merge_ok(old_w, new_w, lang_code)
    if lang_code == "en-US" and old_w and new_w and \
            " ".join(old_w).replace("-", " ").lower().split() == " ".join(new_w).replace("-", " ").lower().split():
        return True                        # only hyphenation: two wheeler -> two-wheeler
    if not old_w:
        return all(w.lower() in grammar for w in new_w)      # insertions: grammar words only

    old_l, new_l = [w.lower() for w in old_w], [w.lower() for w in new_w]

    def covered(word: str, others: list[str]) -> bool:      # explained as a grammar change
        return word in grammar or word in others or any(_inflection(word, o, lang_code) for o in others)

    extra_new = [w for w in new_l if not covered(w, old_l)]
    extra_old = [o for o in old_l if not covered(o, new_l)]
    if any("'" in o or "’" in o for o in extra_old):
        return False                       # don't expand contractions: who's -> who is
    # 2. misspelt / non-existent words, each paired with the real word it resembles
    for o in list(extra_old):
        match = next((w for w in extra_new if _is_misspelling_of(o, w, lang_code)), None)
        if match:
            extra_old.remove(o)
            extra_new.remove(match)
        elif any(w in grammar and (_is_misspelling_of(o, w, lang_code) or
                                   (lang_code == "en-US" and _similar(o, w) and len(o) <= 5))
                 for w in new_l if w not in old_l):
            extra_old.remove(o)            # a typo of a grammar word: "Fo" -> "For", "form" -> "from"
    if not extra_new and not extra_old:
        return True
    # 3. last resort: words the recogniser itself was unsure about ("C doesn't know" -> "She doesn't know")
    replacements = extra_new or [w for w in new_l if w not in old_l]
    if uncertain and extra_old and all(o in uncertain for o in extra_old) and \
            len(replacements) == len(extra_old) and all(_is_real_word(w, lang_code) for w in replacements):
        if level >= 2 or not any(_is_real_word(o, lang_code) for o in extra_old):
            return True
    if not extra_new or len(extra_new) != len(extra_old):
        return False                       # content words dropped (e.g. numbers) or added
    # evidence from the recogniser's alternative hypotheses
    if heard and all(w in heard for w in extra_new):
        return True
    return False


# I / we / you: in Hindi, Punjabi and Marathi the verb shows the speaker's gender (मैं जाता / मैं जाती),
# which can't be known from the text, so the LLM may not flip it.
SPEAKER_PRONOUNS = _n("मैं हम तुम आप तू ਮੈਂ ਅਸੀਂ ਤੁਸੀਂ ਤੂੰ ਤੁਸੀ मी आम्ही तुम्ही".split())
GENDER_PAIRS = [tuple(_n(p.split())) for p in
                ["ता ती", "ते ती", "ा ी", "या ई", "ये ई", "ए ई", "ਦਾ ਦੀ", "ਦੇ ਦੀ", "ਾ ੀ", "ਆ ਈ", "ਏ ਈ", "ो े", "ो ी", "ा े"]]


def _gender_flip(a: str, b: str) -> bool:
    """Same verb with only its gender ending changed: जाता/जाती, गया/गई, ਜਾਂਦਾ/ਜਾਂਦੀ, जातो/जाते."""
    a, b = unicodedata.normalize("NFC", a), unicodedata.normalize("NFC", b)
    for x, y in GENDER_PAIRS:
        for p, q in ((x, y), (y, x)):
            if a.endswith(p) and b.endswith(q) and a[:-len(p)] == b[:-len(q)] and len(a) > len(p):
                return True
    return False


def filter_edits(source: str, answer: str, lang_code: str, heard: set[str] | None = None,
                 uncertain: set[str] | None = None, level: int = 1) -> tuple[str, int, int]:
    """Apply only the allowed edits of `answer` to `source`. Returns (text, kept, reverted).

    heard=None means typed text (no audio); a set (even empty) means audio input.
    uncertain = words the recogniser itself was unsure about; only these may be swapped for other words.
    """
    heard = None if heard is None else {w.lower() for w in heard}
    uncertain = {w.lower() for w in (uncertain or set())}
    src, ans = tokenize(source), tokenize(answer)
    ops = SequenceMatcher(None, [t.lower() for t in src], [t.lower() for t in ans], autojunk=False).get_opcodes()
    out: list[str] = []
    kept = reverted = 0
    speaker = lang_code != "en-US" and bool(src) and unicodedata.normalize("NFC", src[0]) in SPEAKER_PRONOUNS
    for tag, i1, i2, j1, j2 in ops:
        old, new = src[i1:i2], ans[j1:j2]
        if tag == "equal":
            out += new
        elif speaker and tag == "replace" and len(old) == len(new) and \
                any(_gender_flip(o, w) for o, w in zip(old, new) if o != w):
            out += old                                        # the speaker's gender is not a grammar error
            reverted += 1
        elif not any(_is_word(t) for t in old + new):
            out += [t for t in new if t in PUNCTUATION]      # punctuation-only edit
        elif tag == "insert":                                 # judge inserted words one by one
            for t in new:
                if _edit_allowed([], [t], lang_code, heard):
                    out.append(t)
                    kept += _is_word(t)
                else:
                    reverted += 1
        elif _edit_allowed(old, new, lang_code, heard, uncertain, level):
            out += new
            kept += 1
        else:
            out += old
            reverted += 1
    return join_tokens(out), kept, reverted


# --------------------------------------------------------------------------- prompt

@lru_cache(maxsize=None)
def load_examples(lang_code: str, strong: bool = False) -> tuple[tuple[str, str], ...]:
    """strong: the large models' examples, which add several mistakes in two sentences and misheard or
    run-together words (मेराभाई -> मेरा भाई, गुमने -> घूमने)."""
    folder = config.FEWSHOT_DIR.with_name("fewshot_strong") if strong else config.FEWSHOT_DIR
    path = folder / f"{lang_code}.json"
    with open(path, encoding="utf-8") as f:
        return tuple((ex["input"], ex["output"]) for ex in json.load(f))


def build_messages(text: str, lang_code: str, alternatives: list[str] | None = None,
                   uncertain: list[str] | None = None) -> list[dict]:
    lang = get_language(lang_code)
    system = (
        f"You correct the grammar of {lang.name} sentences that a person spoke and a speech recogniser "
        f"wrote down.\n"
        f"Priorities, in this order:\n"
        f"1. Fix every grammatical error: verb form and tense, subject-verb agreement, gender and number "
        f"agreement, missing or wrong helper verbs, articles, prepositions and postpositions, word order.\n"
        f"2. Only if a word does not exist (a misspelling or misheard non-word), replace it with the real word "
        f"that was most likely meant.\n"
        f"3. Only if a real word is listed as unsure and makes no sense, replace it with a similar-sounding "
        f"word that fits. This is the last resort.\n"
        f"Never replace a correct real word with a synonym or a different word. Keep names, numbers and "
        f"contractions. Never add or remove information, never answer or explain the text. Write in "
        f"{lang.name} using the {lang.script} script, never translate. If the text is already correct, "
        f"return it unchanged.\n"
        f'Reply only with JSON: {{"corrected": "<the corrected text>"}}'
    )
    messages = [{"role": "system", "content": system}]
    for inp, out in load_examples(lang_code):
        messages.append({"role": "user", "content": f"Transcript: {inp}"})
        messages.append({"role": "assistant", "content": json.dumps({"corrected": out}, ensure_ascii=False)})
    request = f"Transcript: {text}"
    others = [a for a in (alternatives or []) if a and a.strip() != text.strip()][:3]
    if others:
        request += "\nOther hypotheses from the recogniser: " + " | ".join(others)
    if uncertain:
        request += ("\nWords the recogniser was unsure about (possibly misheard): " + ", ".join(uncertain) +
                    ". If one of them makes no sense here, replace it with a word that sounds similar and fits.")
    messages.append({"role": "user", "content": request})
    return messages


TIME_WORDS = {"en-US": "yesterday, last week, tomorrow, every day", "hi-IN": "कल रात, पिछले हफ़्ते, अगले महीने, रोज़",
              "mr-IN": "काल, गेल्या आठवड्यात, उद्या, रोज", "bn-IN": "গতকাল, গত সপ্তাহে, আগামীকাল, প্রতিদিন",
              "pa-IN": "ਕੱਲ੍ਹ, ਪਿਛਲੇ ਹਫ਼ਤੇ, ਅਗਲੇ ਮਹੀਨੇ, ਰੋਜ਼"}


def build_messages_strong(text: str, lang_code: str, alternatives: list[str] | None = None,
                          uncertain: list[str] | None = None) -> list[dict]:
    """Prompt for a large model: fix every grammar mistake in every sentence, repair clearly misheard
    words, change nothing else."""
    lang = get_language(lang_code)
    system = (
        f"You are an expert in {lang.name} grammar. The text below was spoken by a person and written down by a "
        f"speech recogniser, so it may contain the speaker's grammar mistakes and words the recogniser misheard. "
        f"It may be one sentence or several.\n"
        f"Correct it:\n"
        f"1. Fix every grammatical error in every sentence: verb tense and form, agreement of the verb with its "
        f"subject in person, number and gender, the ergative marker where the language needs it, helper verbs, "
        f"postpositions or prepositions, case endings, articles, word order. The tense must fit time words "
        f"such as {TIME_WORDS[lang_code]}. The sentences of one recording describe one situation: a PRESENT-tense "
        f"verb in a sentence without its own time word takes the time of the sentence before it (\"Yesterday I "
        f"went to the market. There I buy fish.\" -> \"There I bought fish.\"), unless it is a habit or a fact. "
        f"Never change a future or past tense that the speaker chose (\"He went to Mumbai yesterday. He will stay "
        f"there two days.\" is correct).\n"
        f"2. If a word clearly does not exist or makes no sense in the sentence, it was misheard: replace it "
        f"with the similar-sounding word that fits. If two words were run together or one word was split, "
        f"write it correctly.\n"
        f"3. Change nothing else: keep the speaker's words, meaning, word order, names and numbers. No synonyms, "
        f"no paraphrasing, no extra formality, nothing added or removed, no explanations. Don't change "
        f"spellings that are already correct.\n"
        f"4. With I / we / you as the subject, the verb ending shows the speaker's own gender; keep it as spoken.\n"
        f"5. Write in {lang.name} in the {lang.script} script, never translate. If the text is already correct, "
        f"return it unchanged.\n"
        f'Reply only with JSON: {{"corrected": "<the corrected text>"}}'
    )
    messages = [{"role": "system", "content": system}]
    for inp, out in load_examples(lang_code, strong=True):
        messages.append({"role": "user", "content": f"Text: {inp}"})
        messages.append({"role": "assistant", "content": json.dumps({"corrected": out}, ensure_ascii=False)})
    request = f"Text: {text}"
    others = [a for a in (alternatives or []) if a and a.strip() != text.strip()][:3]
    if others:
        request += "\nOther ways the recogniser heard it: " + " | ".join(others)
    if uncertain:
        request += "\nWords the recogniser was unsure about: " + ", ".join(uncertain)
    messages.append({"role": "user", "content": request})
    return messages


CLAUSE_BREAKS = _n(". । ! ? , ; और और आणि पण ਅਤੇ ਤੇ ਪਰ लेकिन पर মগর এবং ও কিন্তু and but".split())


def _speaker_clause(src: list[str], i: int, lang_code: str) -> bool:
    """Is token i in a clause whose subject is I / we / you (the clause starts with that pronoun)?"""
    if lang_code == "en-US":
        return False
    j = i
    while j > 0 and unicodedata.normalize("NFC", src[j - 1]) not in CLAUSE_BREAKS:
        j -= 1
    return unicodedata.normalize("NFC", src[j]) in SPEAKER_PRONOUNS if j < len(src) else False


_PA_SUBJECT_FORMS = _n("ਦਾ ਦੀ ਦੇ ਦੀਆਂ ਗਾ ਗੀ ਗੇ ਗੀਆਂ".split())


def _agrees_with_subject(word: str, lang_code: str) -> bool:
    """Punjabi ਮੈਂ has no ਨੇ, so a past transitive verb agrees with the object (ਮੈਂ ਰੋਟੀ ਖਾਧੀ), not with the
    speaker: only present and future forms (ਜਾਂਦਾ/ਜਾਂਦੀ, ਜਾਵਾਂਗਾ) and ਰਿਹਾ/ਰਹੀ show the speaker's gender."""
    if lang_code != "pa-IN":
        return True
    w = unicodedata.normalize("NFC", word)
    return any(w.endswith(e) for e in _PA_SUBJECT_FORMS) or w in _n("ਰਿਹਾ ਰਹੀ ਰਹੇ".split())


def finalize_strong(text: str, answer: str, lang_code: str, model: str) -> Stage2Result:
    """Light check for a large model's answer: right script, not rewritten wholesale, and the speaker's
    own gender (मैं जाता / मैं जाती) left as spoken. Its grammar edits are trusted (no word-by-word filter:
    that filter was built for a 3B model and would block multi-word fixes and misheard-word repairs)."""
    reason = check_output(text, answer, lang_code, max_ratio=0.8)
    if reason:
        return Stage2Result(text, "rejected", reason, model, answer)
    src, ans = tokenize(text), tokenize(answer)
    out, reverted = [], 0
    ops = SequenceMatcher(None, src, ans, autojunk=False).get_opcodes()
    for tag, i1, i2, j1, j2 in ops:
        old, new = src[i1:i2], ans[j1:j2]
        if tag == "replace" and len(old) == len(new) and old != new and \
                _speaker_clause(src, i1, lang_code) and \
                all(o == w or (_gender_flip(o, w) and _agrees_with_subject(o, lang_code)) for o, w in zip(old, new)):
            out += old
            reverted += 1
        else:
            out += new
    final = join_tokens(out)
    end = text.rstrip()[-1:]
    if end in ".!?।" and final.rstrip()[-1:] not in {".", "!", "?", "।"}:
        final = final.rstrip() + end
    if final.strip() == text.strip():
        return Stage2Result(text, "unchanged", "", model, answer, 0, reverted)
    kept = sum(tag != "equal" for tag, *_ in ops) - reverted
    return Stage2Result(final, "applied", "", model, answer, kept, reverted)


def default_client(wait_on_rate_limit: bool = False):
    """Cloud models first when keys are set (Groq -> Gemini -> local Ollama, see STAGE2_ORDER), else the local model.
    wait_on_rate_limit: for evaluations, wait out a free-tier limit instead of falling back."""
    from .cloud_llm import ChainClient, GeminiClient, GroqClient
    available = {"gemini": (config.GEMINI_API_KEY, GeminiClient), "groq": (config.GROQ_API_KEY, GroqClient)}
    clients = [cls() for name in config.STAGE2_ORDER if name in available
               for key, cls in [available[name]] if key]
    return ChainClient(clients + [OllamaClient()], wait_on_rate_limit) if clients else OllamaClient()


def _has_strong(client, lang_code: str) -> bool:
    members = getattr(client, "clients", [client])
    return any(getattr(c, "strong", False) and (not hasattr(c, "supports") or c.supports(lang_code))
               for c in members)


# --------------------------------------------------------------------------- Ollama

KEEP_ALIVE = "30m"

class OllamaClient:
    name = "Ollama"
    strong = False   # small local model: every edit goes through the strict edit filter

    def __init__(self, url: str = config.OLLAMA_URL, model: str = config.STAGE2_MODEL,
                 timeout: float = config.STAGE2_TIMEOUT):
        self.url, self.model, self.timeout = url, model, timeout

    def supports(self, lang_code: str) -> bool:
        """The local model is only used where it measurably helped (English, see config)."""
        return lang_code not in config.STAGE2_SKIP_LANGUAGES

    def chat(self, messages: list[dict], lang_code: str | None = None) -> str:
        # keep_alive: stay loaded between recordings (no reload delay). num_predict: an upper bound on the
        # answer's length (a correction is about as long as its input), so a runaway answer can't hang the app.
        budget = 64 + 4 * len(messages[-1]["content"])
        payload = {"model": self.model, "messages": messages, "stream": False, "keep_alive": KEEP_ALIVE,
                   "format": RESPONSE_SCHEMA, "options": {"temperature": 0, "num_predict": budget}}
        r = httpx.post(f"{self.url}/api/chat", json=payload, timeout=self.timeout)
        r.raise_for_status()
        content = r.json()["message"]["content"]
        return str(json.loads(content).get("corrected", "")).strip()

    def warm_up(self) -> None:
        """Load the model into memory now, so the first correction doesn't wait for it."""
        httpx.post(f"{self.url}/api/generate", json={"model": self.model, "keep_alive": KEEP_ALIVE},
                   timeout=self.timeout).raise_for_status()

    def status(self) -> dict:
        """Is Ollama running, and is the configured model downloaded?"""
        try:
            r = httpx.get(f"{self.url}/api/tags", timeout=3)
            r.raise_for_status()
            names = [m.get("name", "") for m in r.json().get("models", [])]
        except Exception as exc:
            return {"online": False, "model": self.model, "model_available": False, "error": str(exc)}
        available = any(n == self.model or n.split(":")[0] == self.model for n in names)
        return {"online": True, "model": self.model, "model_available": available}


def finalize(text: str, answer: str, lang_code: str, intensity: int, model: str,
             alternatives: list[str] | None = None, uncertain: list[str] | None = None) -> Stage2Result:
    """Check and filter one LLM answer. Deterministic, so the evaluation can re-apply it to cached answers.

    alternatives=None means typed text; a list (possibly empty) means audio input.
    """
    # 1. whole-answer checks: wrong script / translated, length blown up
    reason = check_output(text, answer, lang_code, max_ratio=1.0)
    if reason:
        return Stage2Result(text, "rejected", reason, model, answer)
    # 2. edit-by-edit filter: only grammar, spelling and (last resort) misheard-word fixes survive
    heard = None if alternatives is None else {w for alt in alternatives for w in _words(alt)}
    filtered, kept, reverted = filter_edits(text, answer, lang_code, heard, set(uncertain or []),
                                            max(1, correction_level(intensity)))
    end = text.rstrip()[-1:]
    if end in ".!?।" and filtered.rstrip()[-1:] not in {".", "!", "?", "।"}:
        filtered = filtered.rstrip() + end  # never drop the sentence's final punctuation
    if filtered.strip() == text.strip():
        why = f"all {reverted} LLM edits were reverted by the edit filter" if reverted else ""
        return Stage2Result(text, "unchanged", why, model, answer, 0, reverted)
    why = f"{reverted} LLM edit(s) reverted by the edit filter" if reverted else ""
    return Stage2Result(filtered, "applied", why, model, answer, kept, reverted)


def correct(text: str, lang_code: str, intensity: int = 50, client=None,
            alternatives: list[str] | None = None, uncertain: list[str] | None = None) -> Stage2Result:
    """Run Stage 2 on Stage 1 output. alternatives = other ASR hypotheses (None for typed text)."""
    client = client or default_client()
    if not text.strip():
        return Stage2Result(text, "skipped", "empty text", client.model)
    if correction_level(intensity) == 0:
        return Stage2Result(text, "skipped", "correction intensity is low (rules only)", client.model)
    strong = _has_strong(client, lang_code)
    build = build_messages_strong if strong else build_messages
    try:
        messages = build(text, lang_code, alternatives, uncertain)
        answer = client.chat(messages, lang_code) if getattr(client, "clients", None) is not None \
            else client.chat(messages)
    except Exception as exc:   # network, rate limit, bad answer: keep the Stage 1 text
        return Stage2Result(text, "unavailable", f"LLM call failed: {exc}", client.model)
    model = getattr(client, "last_model", "") or client.model
    if getattr(client, "last_strong", getattr(client, "strong", False)):
        return finalize_strong(text, answer, lang_code, model)
    return finalize(text, answer, lang_code, intensity, model, alternatives, uncertain)
