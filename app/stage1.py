"""Stage 1: rule-based correction.

Each language has a small set of general grammar rules for errors that speech
recognition commonly produces (wrong agreement, wrong tense with a time word,
missing ergative marker, misheard "could of", ...). Rules are written for the
general pattern, not for one test sentence, and they only fire in the position
where they are safe (e.g. a subject pronoun at the start of a clause).

Then the text is normalised: one space between words, capital first letter
(English), and a sentence-final mark ("." or "।") if ASR left it out.
"""

import os
import re
import os
import unicodedata

from .languages import Language, get_language
from .text_utils import join_tokens, normalize_unicode, split_sentences, tokenize


def N(s: str) -> str:
    """NFC-normalise a rule literal so it matches NFC-normalised input."""
    return unicodedata.normalize("NFC", s)


# --------------------------------------------------------------------------- English

CLAUSE_STARTERS = {"", "and", "but", "so", "because", "if", "when", "while", "that",
                   "then", "although", "since", "or", "now", "after", "before", ",", ".", "!", "?", ";",
                   # a time phrase opening the sentence: "Yesterday we went", "Last week they came"
                   "yesterday", "today", "tomorrow", "tonight", "week", "month", "year", "weekend"}

# common verbs whose bare form after he/she/it at the start of a clause is an agreement error
# ("she know me" -> "she knows me"); do/go/have are handled by their irregular forms
THIRD_SG_VERBS = {"know", "like", "want", "need", "say", "think", "make", "take", "come", "see", "look", "love",
                  "live", "work", "play", "eat", "feel", "seem", "get", "give", "tell", "call", "try", "ask",
                  "leave", "keep", "mean", "help", "believe", "understand", "remember", "speak", "read",
                  "write", "run", "walk", "sleep", "watch", "hate", "teach", "study", "talk", "listen",
                  "stay", "use", "buy", "sell", "drink", "cook", "drive", "sing", "dance", "smile", "cry"}


def _third_person_s(verb: str) -> str:
    if verb.endswith(("s", "sh", "ch", "x", "z", "o")):
        return verb + "es"
    if verb.endswith("y") and verb[-2:-1] not in set("aeiou"):
        return verb[:-1] + "ies"
    return verb + "s"

THIRD_SG = {"he", "she", "it"}
SUBJECT_PRONOUNS = {"i", "you", "we", "they", "he", "she", "it"}
NON_THIRD = {"i", "you", "we", "they"}
# a determiner + noun at the start of a clause is a subject too: "my manager don't" -> "doesn't", "the children goes"
SINGULAR_DETERMINERS = {"my", "your", "his", "her", "our", "their", "the", "this", "that", "a", "an", "every", "each"}
PLURAL_DETERMINERS = {"my", "your", "his", "her", "our", "their", "the", "these", "those", "some", "many", "all"}
PLURAL_NOUNS = {"children", "people", "men", "women", "police", "teeth", "feet", "mice", "parents", "friends"}
# words that follow "the"/"my" but aren't the subject noun ("The even ask" is a typo for "They even ask")
NOT_SUBJECT_NOUNS = {"even", "only", "same", "other", "most", "best", "first", "last", "next", "very", "more",
                     "whole", "rest", "main", "real", "new", "old", "big", "small", "own", "such", "few", "many",
                     "much", "all", "both", "each", "either", "neither", "former", "latter", "following"}


def _noun_number(word: str) -> str | None:
    """"sg" or "pl" for a word that looks like a noun after a determiner, else None."""
    if not word.isalpha() or word in SUBJECT_PRONOUNS or word in NOT_SUBJECT_NOUNS:
        return None
    if word in PLURAL_NOUNS or (word.endswith("s") and not word.endswith(("ss", "us", "is"))):
        return "pl"
    return "sg"

PAST_MARKERS = re.compile(r"\b(yesterday|last (night|week|month|year|time)|\d+ (days?|weeks?|months?|years?) ago|ago)\b", re.I)

PAST_OF = {"go": "went", "goes": "went", "come": "came", "comes": "came", "eat": "ate", "eats": "ate",
           "see": "saw", "sees": "saw", "buy": "bought", "buys": "bought", "take": "took", "takes": "took",
           "make": "made", "makes": "made", "get": "got", "gets": "got", "do": "did", "does": "did",
           "write": "wrote", "writes": "wrote", "meet": "met", "meets": "met", "tell": "told", "tells": "told",
           "say": "said", "says": "said", "give": "gave", "gives": "gave", "find": "found", "finds": "found",
           "leave": "left", "leaves": "left", "drink": "drank", "drinks": "drank", "play": "played",
           "plays": "played", "visit": "visited", "visits": "visited", "call": "called", "calls": "called",
           "watch": "watched", "watches": "watched", "work": "worked", "works": "worked",
           "study": "studied", "studies": "studied", "cook": "cooked", "cooks": "cooked"}

PARTICIPLE_OF = {"went": "gone", "ate": "eaten", "came": "come", "did": "done", "saw": "seen",
                 "wrote": "written", "took": "taken", "gave": "given", "ran": "run", "began": "begun",
                 "drank": "drunk", "spoke": "spoken", "broke": "broken", "chose": "chosen",
                 "forgot": "forgotten", "knew": "known", "grew": "grown", "threw": "thrown",
                 "flew": "flown", "drove": "driven", "rode": "ridden", "swam": "swum", "sang": "sung"}

GERUND_OF = {"go": "going", "come": "coming", "eat": "eating", "play": "playing", "watch": "watching",
             "write": "writing", "work": "working", "study": "studying", "do": "doing",
             "make": "making", "take": "taking", "cook": "cooking", "run": "running", "walk": "walking",
             "sleep": "sleeping", "drive": "driving", "talk": "talking", "wait": "waiting"}

UNCOUNTABLE = {"informations": "information", "advices": "advice", "furnitures": "furniture",
               "equipments": "equipment", "luggages": "luggage", "homeworks": "homework",
               "knowledges": "knowledge"}

COMPARATIVES = {"better", "worse", "easier", "bigger", "faster", "smaller", "larger", "longer",
                "shorter", "cheaper", "happier", "older", "younger", "harder", "higher", "lower"}

MOTION_VERBS = {"go", "goes", "went", "going", "gone", "come", "comes", "came", "coming", "walk",
                "walked", "walking", "drive", "drove", "driving", "run", "ran", "running", "get", "got"}
PLACES_WITH_THE = {"park", "market", "store", "shop", "mall", "hospital", "airport", "station", "cinema",
                   "theatre", "theater", "library", "gym", "beach", "bank", "zoo", "museum", "office",
                   "river", "temple", "restaurant", "supermarket", "pharmacy", "post office"}

DEDUPE_EN = {"the", "a", "an", "i", "is", "are", "was", "were", "to", "of", "and", "in", "on", "at", "it",
             "that", "this", "for", "with", "my", "we", "you", "he", "she", "they", "be", "have", "has",
             "had", "very"}

QUESTION_STARTERS = {"what", "why", "how", "where", "when", "who", "whom", "whose", "which", "is", "are",
                     "am", "do", "does", "did", "can", "could", "will", "would", "should", "shall", "have",
                     "has", "was", "were", "may"}

AN_EXCEPTIONS = ("one", "once", "uni", "use", "usu", "uti", "ufo", "eu", "ewe", "ure", "uro", "ubi")
SILENT_H = ("hour", "honest", "honor", "honour", "heir")


def _match_case(src: str, repl: str) -> str:
    if src[:1].isupper():
        return repl[:1].upper() + repl[1:]
    return repl


def _vowel_sound(word: str) -> bool:
    w = word.lower()
    if not w.isalpha():
        return False
    if w.startswith(SILENT_H):
        return True
    if w.startswith(AN_EXCEPTIONS):
        return False
    return w[0] in "aeiou"


# After do-support and modal verbs the verb must be in its base form:
# "doesn't knows" -> "doesn't know", "didn't went" -> "didn't go", "can goes" -> "can go".
BASE_FORM_TRIGGERS = {"do", "does", "did", "don't", "doesn't", "didn't", "can", "can't", "cannot",
                      "could", "couldn't", "will", "won't", "would", "wouldn't", "should", "shouldn't",
                      "must", "mustn't", "may", "might", "shall"}
ADVERBS_BETWEEN = {"not", "never", "always", "really", "even", "just", "ever", "actually", "also", "still", "only"}
IRREGULAR_BASE = {"went": "go", "ate": "eat", "came": "come", "saw": "see", "took": "take", "made": "make",
                  "got": "get", "told": "tell", "said": "say", "gave": "give", "found": "find", "knew": "know",
                  "thought": "think", "bought": "buy", "brought": "bring", "wrote": "write", "ran": "run",
                  "left": "leave", "felt": "feel", "kept": "keep", "met": "meet", "has": "have", "does": "do",
                  "goes": "go", "is": "be", "was": "be", "were": "be"}
NOT_VERB_S = {"is", "was", "has", "this", "his", "its", "us", "yes", "always", "perhaps", "sometimes",
              "less", "unless", "across", "news", "series", "species", "thanks", "plus", "bus", "gas"}
REGULAR_PAST_OK = {"need", "use", "like", "want", "love", "hate", "play", "work", "talk", "walk", "watch",
                   "call", "ask", "help", "open", "close", "start", "stop", "live", "move", "try", "study",
                   "finish", "visit", "cook", "clean", "wait", "listen", "answer", "believe", "decide"}


# Over-regularised past tenses that speakers (and learners) produce: (simple past, past participle)
OVERREGULARISED = {
    "knowed": ("knew", "known"), "goed": ("went", "gone"), "eated": ("ate", "eaten"), "buyed": ("bought", "bought"),
    "catched": ("caught", "caught"), "teached": ("taught", "taught"), "thinked": ("thought", "thought"),
    "bringed": ("brought", "brought"), "runned": ("ran", "run"), "writed": ("wrote", "written"),
    "telled": ("told", "told"), "comed": ("came", "come"), "gived": ("gave", "given"), "taked": ("took", "taken"),
    "maked": ("made", "made"), "feeled": ("felt", "felt"), "keeped": ("kept", "kept"), "sleeped": ("slept", "slept"),
    "speaked": ("spoke", "spoken"), "breaked": ("broke", "broken"), "drinked": ("drank", "drunk"),
    "drived": ("drove", "driven"), "finded": ("found", "found"), "getted": ("got", "got"), "growed": ("grew", "grown"),
    "leaved": ("left", "left"), "losed": ("lost", "lost"), "meeted": ("met", "met"), "payed": ("paid", "paid"),
    "sayed": ("said", "said"), "selled": ("sold", "sold"), "sended": ("sent", "sent"), "sitted": ("sat", "sat"),
    "standed": ("stood", "stood"), "swimmed": ("swam", "swum"), "throwed": ("threw", "thrown"),
    "understanded": ("understood", "understood"), "winned": ("won", "won"), "falled": ("fell", "fallen"),
    "forgetted": ("forgot", "forgotten"), "heared": ("heard", "heard"),
    "spended": ("spent", "spent"), "builded": ("built", "built"), "fighted": ("fought", "fought"),
    "flied": ("flew", "flown"), "hided": ("hid", "hidden"), "holded": ("held", "held"), "readed": ("read", "read"),
    "rided": ("rode", "ridden"), "shaked": ("shook", "shaken"), "singed": ("sang", "sung"), "stealed": ("stole", "stolen"),
    "weared": ("wore", "worn"), "choosed": ("chose", "chosen"), "drawed": ("drew", "drawn"), "becomed": ("became", "become"),
}
# Past participles used without "have": "I seen it" -> "I saw it", "I never seen it" -> "I have never seen it"
PARTICIPLE_TO_PAST = {"seen": "saw", "done": "did", "gone": "went", "taken": "took", "eaten": "ate",
                      "written": "wrote", "given": "gave", "known": "knew", "spoken": "spoke", "broken": "broke",
                      "driven": "drove", "forgotten": "forgot", "chosen": "chose", "drunk": "drank", "begun": "began",
                      "ridden": "rode", "stolen": "stole", "swum": "swam", "sung": "sang", "flown": "flew"}
PERFECT_ADVERBS = {"never", "already", "just", "ever", "always", "recently", "finally", "also"}
HAVE_FORMS = {"have", "has", "had", "'ve", "'d", "having"}


def _base_form(word: str) -> str | None:
    """Base form of an inflected verb (knows -> know, went -> go), or None if it doesn't look inflected."""
    w = word.lower()
    if w in IRREGULAR_BASE:
        return IRREGULAR_BASE[w]
    if w in NOT_VERB_S or len(w) <= 3 or "'" in w:
        return None
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"                      # studies -> study
    if w.endswith(("ches", "shes", "sses", "xes", "zes", "oes")):
        return w[:-2]                            # watches -> watch, goes handled above
    if w.endswith("s") and not w.endswith("ss"):
        return w[:-1]                            # knows -> know
    if w.endswith("ed"):
        for cand in (w[:-2], w[:-1], w[:-3] if w[-3:-2] == w[-4:-3] else None, w[:-3] + "y" if w.endswith("ied") else None):
            if cand and cand in REGULAR_PAST_OK:
                return cand                      # liked -> like, studied -> study
    return None


def _english_tokens(tokens: list[str]) -> list[str]:
    t = list(tokens)
    low = lambda i: t[i].lower() if 0 <= i < len(t) else ""
    prev = lambda i: low(i - 1) if i > 0 else ""

    i = 0
    while i < len(t):
        w = low(i)
        nxt = low(i + 1)

        # standalone i, i'm, i've, i'll, i'd
        if w in {"i", "i'm", "i've", "i'll", "i'd"}:
            t[i] = "I" + t[i][1:]

        # could of / should of / would of / must of / might of  ->  ... have
        if w in {"could", "should", "would", "must", "might"} and nxt == "of":
            t[i + 1] = "have"

        if w == "alot":
            t[i] = _match_case(t[i], "a lot")

        if w in UNCOUNTABLE:
            t[i] = _match_case(t[i], UNCOUNTABLE[w])

        # more better -> better, most best -> best
        if (w == "more" and nxt in COMPARATIVES) or (w == "most" and nxt == "best"):
            t[i + 1] = _match_case(t[i], t[i + 1])
            del t[i]
            continue

        # have went -> have gone
        if w in {"have", "has", "had", "'ve"} and nxt in PARTICIPLE_OF:
            t[i + 1] = PARTICIPLE_OF[nxt]

        # am/is/are + bare verb -> gerund  ("I am go to school")
        if w in {"am", "is", "are"} and nxt in GERUND_OF:
            t[i + 1] = GERUND_OF[nxt]

        # subject-verb agreement, only for a subject that starts a clause: a pronoun (he, they) or a determiner +
        # noun (my manager, the children; then the verb is two words on)
        if prev(i) in CLAUSE_STARTERS and w not in SUBJECT_PRONOUNS and i + 2 < len(t):
            number = _noun_number(low(i + 1))
            verb_at, verb = i + 2, low(i + 2)
            if number == "sg" and w in SINGULAR_DETERMINERS and w != "a" and w != "an":
                fix = {"have": "has", "do": "does", "go": "goes", "don't": "doesn't", "were": "was"}
                if verb in fix:
                    t[verb_at] = fix[verb]
                elif verb in THIRD_SG_VERBS:
                    t[verb_at] = _match_case(t[verb_at], _third_person_s(verb))
            elif number == "pl" and w in PLURAL_DETERMINERS:
                fix = {"has": "have", "does": "do", "goes": "go", "doesn't": "don't", "was": "were"}
                if verb in fix:
                    t[verb_at] = fix[verb]
                elif _base_form(verb) in THIRD_SG_VERBS and verb.endswith("s"):
                    t[verb_at] = _match_case(t[verb_at], _base_form(verb))
        if prev(i) in CLAUSE_STARTERS:
            if w in THIRD_SG:
                fix = {"have": "has", "do": "does", "go": "goes", "don't": "doesn't", "were": "was"}
                if nxt in fix and not (nxt == "were" and prev(i) == "if"):
                    t[i + 1] = fix[nxt]
                elif nxt in THIRD_SG_VERBS:
                    t[i + 1] = _match_case(t[i + 1], _third_person_s(nxt))
            elif w in NON_THIRD:
                fix = {"has": "have", "does": "do", "goes": "go", "doesn't": "don't"}
                if w != "i":
                    fix["was"] = "were"
                elif prev(i) != "if":
                    fix["were"] = "was"
                if nxt in fix:
                    t[i + 1] = fix[nxt]
                elif nxt.endswith("s") and _base_form(nxt) in THIRD_SG_VERBS:   # they likes -> they like
                    t[i + 1] = _match_case(t[i + 1], _base_form(nxt))

        # there is many -> there are many
        if w == "there" and nxt in {"is", "was", "'s"} and low(i + 2) in {"many", "several", "few", "two",
                                                                        "three", "four", "five", "lots"}:
            t[i + 1] = "are" if nxt in {"is", "'s"} else "were"

        # motion verb + to + place -> "to the place"
        if w in MOTION_VERBS and nxt == "to" and low(i + 2) in PLACES_WITH_THE:
            t.insert(i + 2, "the")

        # a/an
        if w in {"a", "an"} and i + 1 < len(t) and t[i + 1].isalpha() and not t[i + 1].isupper():
            want = "an" if _vowel_sound(t[i + 1]) else "a"
            if w != want:
                t[i] = _match_case(t[i], want)

        # over-regularised past tense: "knowed" -> "knew" (after have/has/had: "known")
        if w in OVERREGULARISED:
            past, participle = OVERREGULARISED[w]
            helper = prev(i) if prev(i) not in ADVERBS_BETWEEN | PERFECT_ADVERBS else low(i - 2)
            t[i] = _match_case(t[i], participle if helper in HAVE_FORMS or helper in {"be", "been", "being", "was", "were", "is", "are"} else past)
            w = t[i].lower()

        # participle without "have" after a clause-starting pronoun:
        #   "I seen it" -> "I saw it";  "I never seen it" -> "I have never seen it"
        if w in SUBJECT_PRONOUNS and prev(i) in CLAUSE_STARTERS:
            j = i + 1
            adverb = low(j) in PERFECT_ADVERBS
            k = j + 1 if adverb else j
            if low(k) in PARTICIPLE_TO_PAST:
                if adverb:
                    t.insert(j, "has" if w in THIRD_SG else "have")
                else:
                    t[k] = _match_case(t[k], PARTICIPLE_TO_PAST[low(k)])

        # base form after do-support / modals (skipping one adverb like "not", "always")
        if w in BASE_FORM_TRIGGERS:
            j = i + 1 + (1 if low(i + 1) in ADVERBS_BETWEEN else 0)
            if j < len(t) and t[j].replace("-", "").isalpha():
                base = _base_form(t[j])
                # "does is"/"can was" etc. are not this error; only fix when a base form exists
                if base and not (w in {"do", "does", "did"} and prev(i) in {"", "what", "how", "why", "where", "when", "who"}
                                 and low(j) in {"you", "he", "she", "it", "we", "they", "i"}):
                    t[j] = _match_case(t[j], base)

        # stutter: "the the" -> "the"
        if w in DEDUPE_EN and nxt == w:
            del t[i + 1]
            continue
        i += 1
    return t


def _english_sentence(sentence: str, apply_rules: bool = True) -> str:
    tokens = tokenize(sentence)
    if not tokens:
        return ""
    end = tokens.pop() if tokens[-1] in {".", "!", "?"} else ""
    if apply_rules:
        tokens = _english_tokens(tokens)

    # past time word + present verb right after a clause-starting subject (a pronoun, or determiner + noun)
    # -> past tense
    if apply_rules and PAST_MARKERS.search(" ".join(tokens)):
        for i in range(len(tokens) - 1):
            starter = tokens[i - 1].lower() if i > 0 else ""
            if starter not in CLAUSE_STARTERS:
                continue
            w = tokens[i].lower()
            if w in THIRD_SG | NON_THIRD:
                at = i + 1
            elif w in SINGULAR_DETERMINERS | PLURAL_DETERMINERS and i + 2 < len(tokens) and \
                    _noun_number(tokens[i + 1].lower()):
                at = i + 2
            else:
                continue
            if tokens[at].lower() in PAST_OF:
                tokens[at] = PAST_OF[tokens[at].lower()]

    if not tokens:
        return ""
    tokens[0] = tokens[0][:1].upper() + tokens[0][1:]
    if not end:
        end = "?" if tokens[0].lower() in QUESTION_STARTERS else "."
    return join_tokens(tokens + [end])


# --------------------------------------------------------------------------- Indic helpers

def _indic_sentence(sentence: str, lang: Language, fix_tokens) -> str:
    tokens = tokenize(sentence)
    if not tokens:
        return ""
    end = tokens.pop() if tokens[-1] in {"।", ".", "!", "?", "॥"} else ""
    if end in {".", "॥"}:
        end = "।"
    tokens = fix_tokens(tokens)
    if not tokens:
        return ""
    if not end:
        end = "?" if tokens[0] in QUESTION_WORDS.get(lang.code, set()) else lang.end_mark
    return join_tokens(tokens + [end])


def _dedupe(tokens: list[str], words: set[str]) -> list[str]:
    out: list[str] = []
    for tok in tokens:
        if out and tok == out[-1] and tok in words:
            continue
        out.append(tok)
    return out


def _consonants(word: str) -> str:
    """The consonants of a word, so recogniser spellings match: ਭੈਣ / ਬਹਿਣ / ਬਹਨ, बहन / बहिन.
    Vowel signs, nasal marks and nukta are dropped, and ਣ/ण count as ਨ/न."""
    out = []
    for ch in unicodedata.normalize("NFD", word):
        cat = unicodedata.category(ch)
        if cat in ("Mn", "Mc"):
            continue
        out.append({"ਣ": "ਨ", "ण": "न", "ਭ": "ਬ", "भ": "ब"}.get(ch, ch))
    return "".join(out).replace("ਬਹ", "ਬ").replace("बह", "ब")


def _feminine_subject(t: list[str], nouns: set[str], possessives: set[str]) -> bool:
    """Sentence starts with a feminine noun, optionally after a possessive: लड़की ..., मेरी बहन ...
    Nouns of 2+ consonants also match by their consonants, so ਬਹਿਣ is recognised as ਭੈਣ."""
    # only nouns whose gender isn't written in a final vowel (ਭੈਣ, बहन, औरत): matching ਦਾਦੀ by consonants
    # would also match ਦਾਦਾ (grandfather), लड़की would match लड़का
    def unmarked(word: str) -> bool:
        return len(_consonants(word)) >= 2 and unicodedata.category(word[-1]) not in ("Mn", "Mc")
    keys = {_consonants(n) for n in nouns if unmarked(n)}

    def feminine(word: str) -> bool:
        return word in nouns or (unmarked(word) and _consonants(word) in keys)
    return feminine(t[0]) or (len(t) > 2 and t[0] in possessives and feminine(t[1]))


def _feminine_verb(t: list[str], aux: set[str], swaps: dict[str, str], past_aux: dict[str, str]) -> None:
    """Make the final verb (and past auxiliary) feminine: जाता है -> जाती है, गया था -> गई थी."""
    v = len(t) - 2 if t[-1] in aux | set(past_aux) and len(t) >= 3 else len(t) - 1
    t[v] = _swap_suffix(t[v], swaps)
    if t[-1] in past_aux:
        t[-1] = past_aux[t[-1]]


def _swap_suffix(word: str, mapping: dict[str, str]) -> str:
    for old, new in mapping.items():
        if word.endswith(old):
            return word[: len(word) - len(old)] + new
    return word


QUESTION_WORDS = {
    "hi-IN": {N("क्या")},
    "mr-IN": {N("काय")},
    "bn-IN": {N("কি"), N("কী")},
    "pa-IN": {N("ਕੀ")},
}


# --------------------------------------------------------------------------- Hindi

HI_AUX_PRESENT = {N(w) for w in ["है", "हैं", "हो", "हूँ", "हूं"]}
HI_IMPERFECTIVE = tuple(N(s) for s in ["ता", "ती", "ते", "रहा", "रही", "रहे"])
HI_SUBJECT_AUX = {N("मैं"): N("हूँ"), N("हम"): N("हैं"), N("आप"): N("हैं"), N("तुम"): N("हो"), N("वे"): N("हैं"),
                  N("वह"): N("है"), N("यह"): N("है"), N("वो"): N("है")}
HI_PLURAL_SUBJECTS = {N(w) for w in ["हम", "आप", "तुम", "वे"]}
HI_FEMININE_NOUNS = {N(w) for w in "लड़की लडकी औरत महिला माँ मां माता बहन बहिन बेटी दीदी दादी नानी चाची मौसी बुआ भाभी पत्नी "
                     "टीचर अध्यापिका रानी गाय बिल्ली चिड़िया".split()}
# words before a noun that don't show its gender: वह लड़की, यह आदमी
HI_DEMONSTRATIVES = {N(w) for w in "वह वो यह ये".split()}
HI_POSSESSIVES = {N(w) for w in "मेरी तेरी उसकी इसकी हमारी तुम्हारी आपकी उनकी".split()} | HI_DEMONSTRATIVES
HI_TO_FEMININE = {N("रहा"): N("रही"), N("गया"): N("गई"), N("ता"): N("ती"), N("आया"): N("आई"),
                  N("एगा"): N("एगी"), N("ेगा"): N("ेगी")}         # future: जाएगा -> जाएगी, करेगा -> करेगी
HI_FUTURE_MASCULINE = (N("एगा"), N("ेगा"))
HI_FUTURE_FEMININE = (N("एगी"), N("ेगी"))
HI_FEMININE_PERFECTIVE = {N(w) for w in "गया आया चला बैठा सोया उठा पहुँचा पहुंचा".split()}   # intransitive past
# the mirror case: a masculine subject with a feminine verb (लड़का जाती है, मेरा भाई आई थी)
HI_MASCULINE_NOUNS = {N(w) for w in "लड़का लडका आदमी बेटा भाई पिता पापा पिताजी दादा नाना चाचा मामा पति राजा".split()}
HI_MASC_POSSESSIVES = {N(w) for w in "मेरा तेरा उसका इसका हमारा तुम्हारा आपका उनका".split()}
# मेरे, उनके...: before a respected/plural noun (मेरे पापा) but also before places (मेरे घर में), so they only
# count when the next word is a masculine noun
HI_OBLIQUE_POSSESSIVES = {N(w) for w in "मेरे तेरे उसके इसके हमारे तुम्हारे आपके उनके".split()} | HI_DEMONSTRATIVES
HI_TO_MASCULINE = {N("रही"): N("रहा"), N("गई"): N("गया"), N("गयी"): N("गया"), N("ती"): N("ता"), N("आई"): N("आया"),
                   N("एगी"): N("एगा"), N("ेगी"): N("ेगा")}
# with हैं/थे (respect or plural: पापा ... हैं) the masculine form is plural: टहलती हैं -> टहलते हैं
HI_TO_MASCULINE_PLURAL = {N("रही"): N("रहे"), N("गई"): N("गए"), N("ती"): N("ते"), N("आई"): N("आए")}
# plural subjects with a singular perfective: हम (लोग) ... गया था -> गए थे
HI_PLURAL_PRONOUNS = {N(w) for w in "हम वे आप तुम".split()}
HI_MASCULINE_PERFECTIVE = {N(w) for w in "गई गयी आई चली बैठी सोई उठी पहुँची पहुंची".split()}
HI_TO_PLURAL = {N("रहा"): N("रहे"), N("ता"): N("ते")}
HI_ERGATIVE = {N("मैं"): N("मैंने"), N("हम"): N("हमने"), N("तुम"): N("तुमने"), N("आप"): N("आपने"),
               N("वह"): N("उसने"), N("वो"): N("उसने"), N("यह"): N("इसने"), N("वे"): N("उन्होंने")}
HI_TRANSITIVE_PERFECTIVE = {N(w) for w in [
    "खाया", "खाई", "खाए", "खायी", "देखा", "देखी", "देखे", "किया", "किए", "किये", "लिखा", "लिखी", "लिखे",
    "पढ़ा", "पढ़ी", "पढ़े", "सुना", "सुनी", "सुने", "बनाया", "बनाई", "बनाए", "पिया", "खरीदा", "खरीदी",
    "खरीदे", "कहा", "दिया", "दिए", "लिया", "भेजा", "भेजी", "पकड़ा", "धोया", "खोला", "बेचा"]}
HI_PAST_AUX = {N("है"): N("था"), N("हैं"): N("थे"), N("हूँ"): N("था"), N("हूं"): N("था")}
# plural nouns as subjects: बच्चे स्कूल जाता है -> जाते हैं, लड़कियाँ ... जाती हैं
HI_PLURAL_NOUNS_M = {N(w) for w in "बच्चे लड़के लडके बेटे लोग छात्र".split()}
HI_PLURAL_NOUNS_F = {N(w) for w in "लड़कियाँ लड़कियां लडकियां औरतें महिलाएँ महिलाएं बेटियाँ बेटियां बहनें छात्राएँ छात्राएं".split()}
# a time phrase that puts the sentence in the past: पिछले हफ़्ते, बीते साल, कल रात (with a present verb: wrong)
HI_PAST_TIME_WORDS = {N(w) for w in "पिछले पिछली पिछला बीते".split()}
HI_YESTERDAY_PARTS = {N(w) for w in "रात शाम सुबह दोपहर".split()}           # कल रात, कल शाम ...
# verbs whose past takes no ने, so the habitual can simply become the past: जाता है -> गया
HI_INTRANSITIVE = {N(w) for w in "जा आ सो रो बैठ चल दौड़ उठ पहुँच पहुंच लौट घूम नाच हँस हंस रुक तैर उड़ जाग गिर "
                   "निकल उतर चढ़ भाग मुस्कुरा".split()}
HI_PERFECTIVE_IRREGULAR = {N("जा"): (N("गया"), N("गई"), N("गए"), N("गईं"))}
# noun subjects with ने: the noun takes its oblique form (लड़का -> लड़के ने, बच्चे -> बच्चों ने)
HI_POSTPOSITIONS = {N(w) for w in "के की का को से में पर तक ने वाले वाली वाला".split()}
HI_OBLIQUE = {N(a): N(b) for a, b in [("लड़का", "लड़के"), ("लडका", "लडके"), ("बेटा", "बेटे"), ("बच्चा", "बच्चे"),
                                         ("बच्चे", "बच्चों"), ("लड़के", "लड़कों"), ("बेटे", "बेटों"), ("लोग", "लोगों"),
                                         ("लड़कियाँ", "लड़कियों"), ("लड़कियां", "लड़कियों"), ("औरतें", "औरतों"),
                                         ("महिलाएँ", "महिलाओं"), ("महिलाएं", "महिलाओं"), ("बेटियाँ", "बेटियों"),
                                         ("बहनें", "बहनों"), ("मेरा", "मेरे"), ("तेरा", "तेरे"), ("उसका", "उसके"),
                                         ("इसका", "इसके"), ("हमारा", "हमारे"), ("तुम्हारा", "तुम्हारे"),
                                         ("आपका", "आपके"), ("उनका", "उनके")]}


def _hi_perfective(stem: str, ending: str, aux: str) -> str:
    """Simple past of an intransitive verb from its habitual ending: जा + ती (+ हैं) -> गईं."""
    idx = 0 if ending == N("ता") else 2 if ending == N("ते") else (3 if aux == N("हैं") else 1)
    if stem in HI_PERFECTIVE_IRREGULAR:
        return HI_PERFECTIVE_IRREGULAR[stem][idx]
    vowel_final = unicodedata.category(stem[-1]) in ("Mn", "Mc") or stem[-1] in "अआइईउऊएऐओऔ"
    endings = (N("या"), N("ई"), N("ए"), N("ईं")) if vowel_final else (N("ा"), N("ी"), N("े"), N("ीं"))
    return stem + endings[idx]


def _hi_noun_subject(t: list[str]) -> int:
    """Length of a noun subject at the start (1 = a noun, 2 = possessive/demonstrative + noun), 0 if none."""
    nouns = HI_FEMININE_NOUNS | HI_MASCULINE_NOUNS | HI_PLURAL_NOUNS_M | HI_PLURAL_NOUNS_F
    if t and t[0] in nouns:
        return 1
    possessives = HI_POSSESSIVES | HI_MASC_POSSESSIVES | HI_OBLIQUE_POSSESSIVES
    if len(t) > 1 and t[0] in possessives and t[1] in nouns | {N("दोस्त"), N("पड़ोसी"), N("टीचर")}:
        return 2
    return 0
# Rules that link the first word (subject) to the last (verb/auxiliary) are only safe in a short,
# single clause. Measured on IndicGEC: in long sentences the last verb often belongs to another clause
# with another subject ("मैं ... सोचता हूं तो मुझमें आग दौड़ पड़ती है" must keep है).
MAX_SIMPLE_TOKENS = 12
CLAUSE_LINKS = {
    "hi-IN": {N(w) for w in "कि तो जो जब जिसमें जिससे जिसे जिसका जिसकी लेकिन क्योंकि और जहाँ जहां अगर यदि , ;".split()},
    "mr-IN": {N(w) for w in "की तर जो जी जे जेव्हा आणि पण कारण जर , ;".split()},
    "bn-IN": {N(w) for w in "যে তাহলে যখন এবং কিন্তু কারণ যদি আর , ;".split()},
    "pa-IN": {N(w) for w in "ਕਿ ਤਾਂ ਜੋ ਜਦੋਂ ਅਤੇ ਪਰ ਕਿਉਂਕਿ ਜੇ , ;".split()},
}


def _simple_clause(t: list[str], lang_code: str) -> bool:
    return len(t) <= MAX_SIMPLE_TOKENS and not (CLAUSE_LINKS[lang_code] & set(t))


HI_DEDUPE = {N(w) for w in ["है", "हैं", "था", "थी", "थे", "की", "का", "के", "को", "में", "से", "ने", "पर", "और"]}


def _hindi_tokens(t: list[str]) -> list[str]:
    t = _dedupe(t, HI_DEDUPE)
    if not _simple_clause(t, "hi-IN"):
        return t
    if len(t) < 2:
        return t
    subj = t[0]

    # 1. subject-auxiliary agreement: मैं जाता है -> मैं जाता हूँ, हम जाता है -> हम जाते हैं
    if subj in HI_SUBJECT_AUX and t[-1] in HI_AUX_PRESENT and len(t) >= 3 and t[-2].endswith(HI_IMPERFECTIVE):
        t[-1] = HI_SUBJECT_AUX[subj]
        if subj in HI_PLURAL_SUBJECTS:
            t[-2] = _swap_suffix(t[-2], HI_TO_PLURAL)

    # 1a. plural noun subject, singular verb: बच्चे स्कूल जाता है -> जाते हैं, लड़कियाँ ... जाता है -> जाती हैं
    plural_noun = subj in HI_PLURAL_NOUNS_M | HI_PLURAL_NOUNS_F or \
        (len(t) > 2 and subj in HI_OBLIQUE_POSSESSIVES | HI_POSSESSIVES and t[1] in HI_PLURAL_NOUNS_M | HI_PLURAL_NOUNS_F)
    if plural_noun and t[-1] == N("है") and len(t) >= 3 and t[-2].endswith(HI_IMPERFECTIVE):
        noun = subj if subj in HI_PLURAL_NOUNS_M | HI_PLURAL_NOUNS_F else t[1]
        t[-1] = N("हैं")
        if noun in HI_PLURAL_NOUNS_M:
            t[-2] = _swap_suffix(t[-2], HI_TO_PLURAL)
        else:
            t[-2] = _swap_suffix(t[-2], {N("ता"): N("ती"), N("रहा"): N("रही")})

    # 1b. feminine subject, masculine verb: लड़की स्कूल जाता है -> लड़की स्कूल जाती है
    # (not with ने: then the verb agrees with the object - माँ ने खाना बनाया था is correct)
    if _feminine_subject(t, HI_FEMININE_NOUNS, HI_POSSESSIVES) and N("ने") not in t and \
            (t[-1] in {N("है"), N("था")} | set(HI_PAST_AUX) or t[-1] in HI_FEMININE_PERFECTIVE
             or t[-1].endswith(HI_FUTURE_MASCULINE)):
        _feminine_verb(t, {N("है")}, HI_TO_FEMININE, {N("था"): N("थी")})

    # 1c. masculine subject, feminine verb: लड़का स्कूल जाती है -> जाता है, मेरा भाई आई थी -> आया था
    masculine = subj in HI_MASCULINE_NOUNS or (len(t) > 2 and (subj in HI_MASC_POSSESSIVES or
                                                               (subj in HI_OBLIQUE_POSSESSIVES and t[1] in HI_MASCULINE_NOUNS)))
    if masculine and N("ने") not in t:
        if t[-1] in {N("हैं"), N("थे")} and len(t) >= 3:                  # respect/plural: पापा ... टहलती हैं
            t[-2] = _swap_suffix(t[-2], HI_TO_MASCULINE_PLURAL)
        elif t[-1] in {N("है"), N("थी")} | set(HI_PAST_AUX) or t[-1] in HI_MASCULINE_PERFECTIVE \
                or t[-1].endswith(HI_FUTURE_FEMININE):
            _feminine_verb(t, {N("है")}, HI_TO_MASCULINE, {N("थी"): N("था")})   # same steps, other direction

    # 1d. plural subject, singular perfective: हम लोग ... देखने गया था -> गए थे (not transitive verbs: there the
    # ergative rule below applies, हम खाना खाया -> हमने खाना खाया)
    if subj in HI_PLURAL_PRONOUNS and N("ने") not in t and len(t) >= 3:
        v = len(t) - 2 if t[-1] == N("था") else len(t) - 1
        if t[v].endswith(N("या")) and t[v] not in HI_TRANSITIVE_PERFECTIVE and t[v] != subj:
            t[v] = t[v][: -len(N("या"))] + N("ए")
            if t[-1] == N("था"):
                t[-1] = N("थे")

    # 2. "कल" (yesterday) + perfective + present aux -> past aux: गया है -> गया था
    if N("कल") in t and len(t) >= 3 and t[-1] in HI_PAST_AUX:
        verb = t[-2]
        if verb.endswith((N("ई"), N("यी"))):
            t[-1] = N("थीं") if t[-1] == N("हैं") else N("थी")
        elif verb.endswith(N("ए")) or verb.endswith(N("ये")):
            t[-1] = N("थे")
        elif verb.endswith(N("या")) or verb.endswith(N("आ")):
            t[-1] = HI_PAST_AUX[t[-1]]

    # 2b. a past time phrase (पिछले हफ़्ते, बीते साल, कल रात) with a present habitual verb -> simple past:
    # लड़का पिछले हफ़्ते बाज़ार जाता है -> लड़का पिछले हफ़्ते बाज़ार गया (intransitive verbs only: no ने needed)
    past_time = any(w in HI_PAST_TIME_WORDS for w in t) or \
        any(t[i] == N("कल") and t[i + 1] in HI_YESTERDAY_PARTS for i in range(len(t) - 1))
    if past_time and len(t) >= 3 and t[-1] in HI_AUX_PRESENT:
        verb = t[-2]
        for ending in (N("ता"), N("ती"), N("ते")):
            if verb.endswith(ending) and verb[: -len(ending)] in HI_INTRANSITIVE:
                t[-2:] = [_hi_perfective(verb[: -len(ending)], ending, t[-1])]
                break

    # 3. missing ergative ने with a transitive perfective verb: वह खाना खाया -> उसने खाना खाया
    last = t[-2] if t[-1] in {N("है"), N("था"), N("थी"), N("थे"), N("हैं")} and len(t) >= 3 else t[-1]
    if subj in HI_ERGATIVE and N("ने") not in t:
        if last in HI_TRANSITIVE_PERFECTIVE:
            t[0] = HI_ERGATIVE[subj]
    # 3b. the same with a noun subject: लड़का रोटी खाई -> लड़के ने रोटी खाई, मेरा भाई ... -> मेरे भाई ने ...
    elif N("ने") not in t and last in HI_TRANSITIVE_PERFECTIVE and not any(w.endswith(N("ने")) for w in t):
        n = _hi_noun_subject(t)
        # not when the noun is followed by a postposition (पापा के जाते समय ...): then it isn't the subject
        if n and len(t) >= n + 2 and t[n] not in HI_POSTPOSITIONS:
            t = [HI_OBLIQUE.get(w, w) for w in t[:n]] + [N("ने")] + t[n:]
    return t


# --------------------------------------------------------------------------- Marathi

MR_AUX = {N(w) for w in ["आहे", "आहेस", "आहोत", "आहात"]}
MR_SUBJECT_AUX = {N("मी"): N("आहे"), N("आम्ही"): N("आहोत"), N("आपण"): N("आहोत"), N("तुम्ही"): N("आहात"),
                  N("तू"): N("आहेस"), N("ते"): N("आहेत"), N("त्या"): N("आहेत"), N("मुले"): N("आहेत"),
                  N("मुली"): N("आहेत"), N("लोक"): N("आहेत")}

# Verb agreement by conjugation: the subject's person/number/gender decides the ending.
# subject -> key (1sg, 1pl, 2sg, 2pl, 3sg_m, 3sg_f, 3pl_m, 3pl_f); मी is 1sg of either gender
MR_SUBJECT_KEY = {N("मी"): "1sg", N("आम्ही"): "1pl", N("आपण"): "1pl", N("तू"): "2sg", N("तुम्ही"): "2pl",
                  N("तो"): "3sg_m", N("ती"): "3sg_f", N("ते"): "3pl_m", N("त्या"): "3pl_f", N("मुले"): "3pl_m",
                  N("मुली"): "3pl_f", N("लोक"): "3pl_m"}
MR_HABITUAL_END = {"1sg_m": "तो", "1sg_f": "ते", "1pl": "तो", "2sg": "तोस", "2pl": "ता", "3sg_m": "तो",
                   "3sg_f": "ते", "3pl_m": "तात", "3pl_f": "तात"}
MR_HABITUAL_ENDINGS = [N(e) for e in ("तोस", "तेस", "तात", "तो", "ते", "ता")]
# regular verb stems (habitual = stem + तो/ते...); intransitive ones also form the past without ने-type agreement
MR_STEMS = {N(w) for w in "जा ये खेळ झोप बस उठ कर बोल चाल धाव खा पाह वाच शिक राह दे घे ने पळ पोहोच फिर".split()}
MR_INTRANSITIVE = {N(w) for w in "जा ये खेळ झोप बस उठ चाल धाव पळ पोहोच फिर".split()}
MR_PAST_STEM = {N("जा"): N("गेल"), N("ये"): N("आल"), N("पळ"): N("पळाल")}
MR_PAST_END = {"1sg_m": "ो", "1sg_f": "े", "1pl": "ो", "2sg_m": "ास", "2sg_f": "ीस", "2pl": "ात", "3sg_m": "ा",
               "3sg_f": "ी", "3pl_m": "े", "3pl_f": "्या"}
MR_FUTURE_VOWEL = {N("जा"): ("जाईन", "जाऊ", "जाशील", "जाल", "जाईल", "जातील"),
                   N("ये"): ("येईन", "येऊ", "येशील", "याल", "येईल", "येतील")}
MR_FUTURE_END = ("ेन", "ू", "शील", "ाल", "ेल", "तील")       # consonant stems: खेळेन, खेळू, खेळशील ...
MR_PAST_TIME = {N(w) for w in "काल मागच्या गेल्या".split()}
MR_FUTURE_TIME = {N(w) for w in "उद्या पुढच्या".split()}


def _mr_subject_key(t: list[str]) -> str | None:
    if t[0] in MR_SUBJECT_KEY:
        return MR_SUBJECT_KEY[t[0]]
    if t[0] in MR_FEMININE_SUBJECTS or (len(t) > 2 and t[0] in MR_FEMININE_POSSESSIVES):
        return "3sg_f"
    if t[0] in MR_MASCULINE_SUBJECTS or (len(t) > 2 and t[0] in MR_MASCULINE_POSSESSIVES):
        return "3sg_m"
    return None


def _mr_split_habitual(word: str) -> tuple[str, str] | None:
    for ending in MR_HABITUAL_ENDINGS:
        if word.endswith(ending) and word[: -len(ending)] in MR_STEMS:
            return word[: -len(ending)], ending
    return None


def _mr_split_past(word: str) -> tuple[str, str] | None:
    """(stem, gender) of an intransitive past form: गेला -> (जा, m), खेळली -> (खेळ, f)."""
    for stem in MR_INTRANSITIVE:
        past = MR_PAST_STEM.get(stem, stem + N("ल"))
        if word.startswith(past) and word[len(past):] in {N(e) for e in MR_PAST_END.values()}:
            rest = word[len(past):]
            return stem, "f" if rest in (N("ी"), N("े"), N("ीस"), N("्या")) else "m"
    return None


def _mr_future(stem: str, key: str) -> str:
    i = {"1sg": 0, "1pl": 1, "2sg": 2, "2pl": 3, "3sg": 4, "3pl": 5}[key.split("_")[0]]
    if stem in MR_FUTURE_VOWEL:
        return N(MR_FUTURE_VOWEL[stem][i])
    return stem + N(MR_FUTURE_END[i])
MR_YESTERDAY_PAST = {N("जातो"): N("गेलो"), N("जाते"): N("गेले"), N("येतो"): N("आलो"), N("येते"): N("आले"),
                     N("करतो"): N("केले"), N("करते"): N("केले"), N("खातो"): N("खाल्ले"), N("खाते"): N("खाल्ले"),
                     N("पाहतो"): N("पाहिले"), N("पाहते"): N("पाहिले"), N("बघतो"): N("बघितले"),
                     N("बघते"): N("बघितले")}
MR_PERFECTIVE_FIRST = {N(w) for w in ["गेलो", "गेले", "आलो", "आले"]}
MR_FEMININE_SUBJECTS = {N(w) for w in "ती मुलगी आई बहीण आजी ताई बायको".split()}
MR_FEMININE_POSSESSIVES = {N(w) for w in "माझी तुझी आमची तुमची त्याची तिची त्यांची".split()}
# the mirror case: तो (he), a masculine noun or a masculine possessive with a feminine verb: तो ... करते -> करतो
MR_MASCULINE_SUBJECTS = {N(w) for w in "तो मुलगा भाऊ दादा".split()}
MR_MASCULINE_POSSESSIVES = {N(w) for w in "माझा तुझा आमचा तुमचा त्याचा तिचा त्यांचा".split()}
MR_FEMININE_OF = {N("गेला"): N("गेली"), N("आला"): N("आली"), N("होता"): N("होती"), N("बसला"): N("बसली"),
                  N("झोपला"): N("झोपली")}   # intransitive only: with a transitive verb Marathi agrees with the object
MR_DEDUPE = {N(w) for w in ["आहे", "आहेत", "होता", "होती", "आणि", "ला", "ने", "चा", "ची", "चे"]}


def _marathi_school_spellings() -> dict[str, str]:
    """शाळा (school) and its forms are always written with ळ in Marathi; the recogniser writes ल, स for श, or
    drops the ा (live: शालेत, सालेत; also शलेत). Every such spelling maps back to the real word."""
    out = {N("शाला"): N("शाळा")}      # the bare word only with ल: साला (brother-in-law) is a real word
    for word in ["शाळेत", "शाळेला", "शाळेचा", "शाळेची", "शाळेचे", "शाळेतून", "शाळेमध्ये", "शाळेतील"]:
        for first in ("श", "स"):
            for vowel in ("ा", ""):
                out[N(first + vowel + word[2:].replace("ळ", "ल"))] = N(word)
    return out


MR_SPELLINGS = {N("कल"): N("काल"), **_marathi_school_spellings()}


def _marathi_tokens(t: list[str]) -> list[str]:
    # recogniser spellings: काल (yesterday) written the Hindi way, शाळेत with ल (live: "मी कल बाजारात जातो",
    # "मुलगी सालेत गेला")
    t = [MR_SPELLINGS.get(tok, tok) for tok in t]
    t = _dedupe(t, MR_DEDUPE)
    if len(t) < 2:
        return t

    # 1. progressive: जातो आहे -> जात आहे  (verb + तो/ते before आहे loses the vowel)
    for i in range(len(t) - 1):
        if t[i + 1] in MR_AUX and len(t[i]) > 2 and t[i].endswith((N("तो"), N("ते"))):
            t[i] = t[i][:-1]

    if not _simple_clause(t, "mr-IN"):
        return t
    subj = t[0]
    # 2. subject-auxiliary agreement: आम्ही जात आहे -> आम्ही जात आहोत
    if subj in MR_SUBJECT_AUX and t[-1] in MR_AUX:
        t[-1] = MR_SUBJECT_AUX[subj]

    # 2b. ती (she) or a feminine noun with a masculine verb: ती घरी जातो -> ती घरी जाते. A feminine possessive
    # (माझी, तिची...) agrees with its noun, so "माझी आई ..." is a feminine subject too.
    if t[0] in MR_FEMININE_SUBJECTS or (len(t) > 2 and t[0] in MR_FEMININE_POSSESSIVES):
        last = t[-1]
        if last in MR_FEMININE_OF:
            t[-1] = MR_FEMININE_OF[last]
        elif last.endswith(N("तो")) and last not in MR_AUX:
            t[-1] = last[:-len(N("तो"))] + N("ते")

    # 2c. masculine subject, feminine verb: तो रोज व्यायाम करते -> करतो, माझा भाऊ घरी गेली -> गेला
    elif t[0] in MR_MASCULINE_SUBJECTS or (len(t) > 2 and t[0] in MR_MASCULINE_POSSESSIVES):
        last = t[-1]
        masculine_of = {v: k for k, v in MR_FEMININE_OF.items()}
        if last in masculine_of:
            t[-1] = masculine_of[last]
        elif last.endswith(N("ते")) and last not in MR_AUX and len(last) > 2:
            t[-1] = last[:-len(N("ते"))] + N("तो")

    # 3. काल (yesterday) with मी: present habitual -> past;  गेलो आहे -> गेलो
    if subj == N("मी") and N("काल") in t:
        if t[-1] in MR_YESTERDAY_PAST:
            t[-1] = MR_YESTERDAY_PAST[t[-1]]
        elif t[-1] == N("आहे") and len(t) >= 3 and t[-2] in MR_PERFECTIVE_FIRST:
            t.pop()

    key = _mr_subject_key(t)
    if key and len(t) >= 2:
        habitual = _mr_split_habitual(t[-1])
        # 4. present habitual ending by subject: तू रोज जातो -> जातोस, ते रोज जातो -> जातात, तुम्ही जातो -> जाता
        if habitual and not (MR_PAST_TIME & set(t)):
            stem, ending = habitual
            want = MR_HABITUAL_END.get(key) if key != "1sg" else (ending if ending in (N("तो"), N("ते")) else N("तो"))
            neuter = t[0] == N("ते") and ending == N("ते")        # ते also means "it": ते रडते is correct
            if want and N(want) != ending and not neuter:
                t[-1] = stem + N(want)
        # 5. a past time word (काल, गेल्या/मागच्या आठवड्यात) with the present -> past: ते काल खेळतात -> खेळले
        elif habitual and habitual[0] in MR_INTRANSITIVE:
            stem, ending = habitual
            gender = "f" if ending in (N("ते"), N("तेस")) and key in ("1sg", "2sg") else ""
            k = key if "_" in key or key in ("1pl", "2pl") else f"{key}_{gender or 'm'}"
            if k in MR_PAST_END:
                t[-1] = MR_PAST_STEM.get(stem, stem + N("ल")) + N(MR_PAST_END[k])
        # 6. उद्या (tomorrow) with the past -> future: मी उद्या शाळेत गेलो -> जाईन
        elif MR_FUTURE_TIME & set(t):
            past = _mr_split_past(t[-1])
            if past:
                t[-1] = _mr_future(past[0], key)
    return t


# --------------------------------------------------------------------------- Bengali

BN_STRAY_HOY_AFTER = tuple(N(s) for s in ["ছি", "ছে", "লাম", "বো"])
BN_FIRST_TO_THIRD = {N("করি"): N("করে"), N("যাই"): N("যায়"), N("খাই"): N("খায়"), N("আসি"): N("আসে"),
                     N("দেখি"): N("দেখে"), N("পড়ি"): N("পড়ে")}
BN_THIRD_TO_FIRST = {v: k for k, v in BN_FIRST_TO_THIRD.items()}
BN_YESTERDAY_PAST = {N("যাই"): N("গেলাম"), N("আসি"): N("এলাম"), N("খাই"): N("খেলাম"),
                     N("দেখি"): N("দেখলাম"), N("করি"): N("করলাম"), N("পড়ি"): N("পড়লাম")}
BN_THIRD_SUBJECTS = {N("সে"), N("ও")}
BN_FIRST_SUBJECTS = {N("আমি"), N("আমরা")}      # we takes the same verb forms as I: আমরা যাই, আমরা গেলাম
# a possessive + noun is a third-person subject: আমার ভাই ..., তার মা ...
BN_POSSESSIVES = {N(w) for w in "আমার তোমার তার ওর ওনার তাঁর আমাদের তোমাদের ওদের তাদের".split()}
BN_YESTERDAY_PAST_THIRD = {N("যায়"): N("গেল"), N("আসে"): N("এল"), N("খায়"): N("খেল"), N("দেখে"): N("দেখল"),
                           N("করে"): N("করল"), N("পড়ে"): N("পড়ল")}
BN_DEDUPE = {N(w) for w in ["হয়", "আছে", "ছিল", "এবং", "কে", "এর"]}   # not থেকে/আর: থেকে থেকে, আর আর are idioms


# Conjugation of common verbs: present, continuous, past, future, each for persons 1 (আমি/আমরা), 2 (তুমি),
# h (আপনি, formal) and 3 (সে, তারা, আমার ভাই). Bengali verbs don't change for gender or number.
_BN_TABLE = """
যা যাই যাও যান যায় যাচ্ছি যাচ্ছ যাচ্ছেন যাচ্ছে গেলাম গেলে গেলেন গেল যাব যাবে যাবেন যাবে
খা খাই খাও খান খায় খাচ্ছি খাচ্ছ খাচ্ছেন খাচ্ছে খেলাম খেলে খেলেন খেল খাব খাবে খাবেন খাবে
কর করি করো করেন করে করছি করছ করছেন করছে করলাম করলে করলেন করল করব করবে করবেন করবে
পড় পড়ি পড়ো পড়েন পড়ে পড়ছি পড়ছ পড়ছেন পড়ছে পড়লাম পড়লে পড়লেন পড়ল পড়ব পড়বে পড়বেন পড়বে
দেখ দেখি দেখো দেখেন দেখে দেখছি দেখছ দেখছেন দেখছে দেখলাম দেখলে দেখলেন দেখল দেখব দেখবে দেখবেন দেখবে
আস আসি আসো আসেন আসে আসছি আসছ আসছেন আসছে এলাম এলে এলেন এল আসব আসবে আসবেন আসবে
বল বলি বলো বলেন বলে বলছি বলছ বলছেন বলছে বললাম বললে বললেন বলল বলব বলবে বলবেন বলবে
শোন শুনি শোনো শোনেন শোনে শুনছি শুনছ শুনছেন শুনছে শুনলাম শুনলে শুনলেন শুনল শুনব শুনবে শুনবেন শুনবে
লেখ লিখি লেখো লেখেন লেখে লিখছি লিখছ লিখছেন লিখছে লিখলাম লিখলে লিখলেন লিখল লিখব লিখবে লিখবেন লিখবে
থাক থাকি থাকো থাকেন থাকে থাকছি থাকছ থাকছেন থাকছে থাকলাম থাকলে থাকলেন থাকল থাকব থাকবে থাকবেন থাকবে
ঘুমা ঘুমাই ঘুমাও ঘুমান ঘুমায় ঘুমাচ্ছি ঘুমাচ্ছ ঘুমাচ্ছেন ঘুমাচ্ছে ঘুমালাম ঘুমালে ঘুমালেন ঘুমাল ঘুমাব ঘুমাবে ঘুমাবেন ঘুমাবে
খেল খেলি খেলো খেলেন খেলে খেলছি খেলছ খেলছেন খেলছে খেললাম খেললে খেললেন খেলল খেলব খেলবে খেলবেন খেলবে
শেখ শিখি শেখো শেখেন শেখে শিখছি শিখছ শিখছেন শিখছে শিখলাম শিখলে শিখলেন শিখল শিখব শিখবে শিখবেন শিখবে
কেন কিনি কেনো কেনেন কেনে কিনছি কিনছ কিনছেন কিনছে কিনলাম কিনলে কিনলেন কিনল কিনব কিনবে কিনবেন কিনবে
দে দিই দাও দেন দেয় দিচ্ছি দিচ্ছ দিচ্ছেন দিচ্ছে দিলাম দিলে দিলেন দিল দেব দেবে দেবেন দেবে
নে নিই নাও নেন নেয় নিচ্ছি নিচ্ছ নিচ্ছেন নিচ্ছে নিলাম নিলে নিলেন নিল নেব নেবে নেবেন নেবে
পা পাই পাও পান পায় পাচ্ছি পাচ্ছ পাচ্ছেন পাচ্ছে পেলাম পেলে পেলেন পেল পাব পাবে পাবেন পাবে
বস বসি বসো বসেন বসে বসছি বসছ বসছেন বসছে বসলাম বসলে বসলেন বসল বসব বসবে বসবেন বসবে
ওঠ উঠি ওঠো ওঠেন ওঠে উঠছি উঠছ উঠছেন উঠছে উঠলাম উঠলে উঠলেন উঠল উঠব উঠবে উঠবেন উঠবে
হাঁট হাঁটি হাঁটো হাঁটেন হাঁটে হাঁটছি হাঁটছ হাঁটছেন হাঁটছে হাঁটলাম হাঁটলে হাঁটলেন হাঁটল হাঁটব হাঁটবে হাঁটবেন হাঁটবে
"""
BN_TENSES = ("present", "continuous", "past", "future")
BN_PERSONS = ("1", "2", "h", "3")
BN_CONJ: dict[str, dict[str, tuple[str, ...]]] = {}
BN_FORM_INDEX: dict[str, set[tuple[str, str, str]]] = {}
for _line in _BN_TABLE.strip().splitlines():
    _verb, *_forms = [N(w) for w in _line.split()]
    BN_CONJ[_verb] = {tense: tuple(_forms[i * 4:(i + 1) * 4]) for i, tense in enumerate(BN_TENSES)}
    for _ti, _tense in enumerate(BN_TENSES):
        for _pi, _person in enumerate(BN_PERSONS):
            BN_FORM_INDEX.setdefault(_forms[_ti * 4 + _pi], set()).add((_verb, _tense, _person))
BN_SUBJECT_PERSON = {**{N(w): "1" for w in "আমি আমরা".split()}, **{N(w): "2" for w in "তুমি তোমরা".split()},
                     **{N(w): "h" for w in "আপনি আপনারা উনি তিনি ওনারা".split()},
                     **{N(w): "3" for w in "সে ও তারা ওরা এরা ছেলেরা মেয়েরা লোকেরা বাচ্চারা ছাত্ররা".split()}}
BN_PAST_TIME = {N(w) for w in "গতকাল গত".split()}            # not কাল: it means both yesterday and tomorrow
BN_FUTURE_TIME = {N(w) for w in "আগামীকাল আগামী".split()}


def _bn_conjugate(t: list[str]) -> list[str]:
    """Make the final verb agree with the subject's person and with a time word: সে এখন বই পড়ছি -> পড়ছে,
    তুমি কাজ করি -> করো, আমার ভাই গতকাল অফিসে যায় -> গেল, আমরা আগামীকাল সিনেমা দেখলাম -> দেখব."""
    person = BN_SUBJECT_PERSON.get(t[0]) or ("3" if len(t) > 2 and t[0] in BN_POSSESSIVES else None)
    readings = BN_FORM_INDEX.get(t[-1])
    if not person or not readings or len({verb for verb, _, _ in readings}) != 1:
        return t                                   # unknown verb, or a form two verbs share (খেলে)
    verb = next(iter(readings))[0]
    tenses = {tense for _, tense, _ in readings}
    if BN_PAST_TIME & set(t) and tenses & {"present", "continuous"}:
        want = "past"
    elif BN_FUTURE_TIME & set(t) and "past" in tenses:
        want = "future"
    elif len(tenses) == 1:
        want = next(iter(tenses))
    else:
        return t
    if (verb, want, person) in readings:
        return t                                   # already right (also covers future: যাবে is 2nd and 3rd person)
    t[-1] = BN_CONJ[verb][want][BN_PERSONS.index(person)]
    return t


# Bengali sounds recognisers confuse: retroflex vs dental (ট/ত, ঠ/থ, ড/দ, ঢ/ধ, ণ/ন)
BN_SAME_SOUND = {"ট": "ত", "ঠ": "থ", "ড": "দ", "ঢ": "ধ", "ণ": "ন"}


def _bn_skeleton(word: str) -> str:
    return "".join(BN_SAME_SOUND.get(ch, ch) for ch in unicodedata.normalize("NFD", word)
                   if unicodedata.category(ch) not in ("Mn", "Mc"))


BN_YESTERDAY_KEY = _bn_skeleton(N("গতকাল"))


def _bengali_tokens(t: list[str]) -> list[str]:
    # recogniser spellings of গতকাল (yesterday), live: "গোটকাল", so the tense rules below still see it
    t = [N("গতকাল") if tok != N("গতকাল") and _bn_skeleton(tok) == BN_YESTERDAY_KEY else tok for tok in t]
    t = _dedupe(t, BN_DEDUPE)
    # 1. stray "হয়" after a finished verb: আমি যাচ্ছি হয় -> আমি যাচ্ছি
    t = [tok for i, tok in enumerate(t)
         if not (tok == N("হয়") and i > 0 and t[i - 1].endswith(BN_STRAY_HOY_AFTER))]
    if len(t) < 2 or not _simple_clause(t, "bn-IN"):
        return t
    subj, last = t[0], t[-1]

    if subj in BN_FIRST_SUBJECTS:
        # 2. গতকাল (yesterday) + present -> simple past: আমি গতকাল স্কুলে যাই -> ... গেলাম
        if N("গতকাল") in t and last in BN_YESTERDAY_PAST:
            t[-1] = BN_YESTERDAY_PAST[last]
        # 3. first-person agreement: আমি ... গিয়েছে -> গিয়েছি, আমি ... করে -> করি
        elif last in BN_THIRD_TO_FIRST:
            t[-1] = BN_THIRD_TO_FIRST[last]
        # -ছে -> -ছি: খেয়েছে, খাচ্ছে, করছে, আছে (not কাছে/গাছে "near"/"in the tree", which aren't verbs)
        elif last.endswith(N("ছে")) and not last.endswith(N("াছে")):
            t[-1] = last[: -len(N("ছে"))] + N("ছি")
    elif subj in BN_THIRD_SUBJECTS or (len(t) > 2 and subj in BN_POSSESSIVES):
        # 4. third-person agreement: সে ... করি -> করে
        if N("গতকাল") in t and (BN_FIRST_TO_THIRD.get(last, last)) in BN_YESTERDAY_PAST_THIRD:
            t[-1] = BN_YESTERDAY_PAST_THIRD[BN_FIRST_TO_THIRD.get(last, last)]   # সে গতকাল যায় -> গেল
        elif last in BN_FIRST_TO_THIRD:
            t[-1] = BN_FIRST_TO_THIRD[last]
        elif last.endswith(N("েছি")):
            t[-1] = last[: -len(N("ছি"))] + N("ছে")
    # 5. any verb in the conjugation table: person and tense from the subject and the time words
    return _bn_conjugate(t)


# --------------------------------------------------------------------------- Punjabi

PA_AUX = {N(w) for w in ["ਹੈ", "ਹਨ", "ਹੋ", "ਹਾਂ"]}
PA_IMPERFECTIVE = tuple(N(s) for s in ["ਦਾ", "ਦੀ", "ਦੇ", "ਰਿਹਾ", "ਰਹੀ", "ਰਹੇ"])
PA_SUBJECT_AUX = {N("ਮੈਂ"): N("ਹਾਂ"), N("ਅਸੀਂ"): N("ਹਾਂ"), N("ਤੁਸੀਂ"): N("ਹੋ"), N("ਉਹ"): None, N("ਇਹ"): None}
PA_PLURAL_SUBJECTS = {N("ਅਸੀਂ"), N("ਤੁਸੀਂ")}
# includes the spellings Google's recogniser writes (live: ਭੈਣ heard as ਬਹਨ)
PA_FEMININE_NOUNS = {N(w) for w in "ਕੁੜੀ ਲੜਕੀ ਔਰਤ ਮਾਂ ਮਾਤਾ ਭੈਣ ਬਹਨ ਬਹਿਨ ਭੈਨ ਧੀ ਦੀਦੀ ਦਾਦੀ ਨਾਨੀ ਪਤਨੀ ਬੀਬੀ "
                     "ਚਾਚੀ ਮਾਸੀ ਭੂਆ ਭਾਬੀ ਨੂੰਹ ਅਧਿਆਪਕਾ ਗਾਂ".split()}
PA_DEMONSTRATIVES = {N(w) for w in "ਉਹ ਇਹ".split()}       # ਉਹ ਕੁੜੀ, ਇਹ ਮੁੰਡਾ: the noun shows the gender
PA_POSSESSIVES = {N(w) for w in "ਮੇਰੀ ਤੇਰੀ ਉਸਦੀ ਸਾਡੀ ਤੁਹਾਡੀ ਉਹਨਾਂ ਦੀ".split()} | PA_DEMONSTRATIVES
# the mirror case: a masculine subject with a feminine verb: ਮੇਰਾ ਭਰਾ ... ਗਈ ਸੀ -> ਗਿਆ ਸੀ
PA_MASCULINE_NOUNS = {N(w) for w in "ਮੁੰਡਾ ਲੜਕਾ ਭਰਾ ਪੁੱਤਰ ਪੁੱਤ ਪਿਤਾ ਪਾਪਾ ਦਾਦਾ ਨਾਨਾ ਚਾਚਾ ਮਾਮਾ ਪਤੀ ਆਦਮੀ".split()}
PA_MASC_POSSESSIVES = {N(w) for w in "ਮੇਰਾ ਤੇਰਾ ਉਸਦਾ ਸਾਡਾ ਤੁਹਾਡਾ".split()}
PA_TO_MASCULINE = {N("ਰਹੀ"): N("ਰਿਹਾ"), N("ਗਈ"): N("ਗਿਆ"), N("ਦੀ"): N("ਦਾ"), N("ਆਈ"): N("ਆਇਆ"),
                   N("ਵੇਗੀ"): N("ਵੇਗਾ"), N("ਊਗੀ"): N("ਊਗਾ"), N("ਸੁੱਤੀ"): N("ਸੁੱਤਾ")}
PA_MASCULINE_PERFECTIVE = {N(w) for w in "ਗਈ ਆਈ".split()}
PA_TO_FEMININE = {N("ਰਿਹਾ"): N("ਰਹੀ"), N("ਗਿਆ"): N("ਗਈ"), N("ਦਾ"): N("ਦੀ"), N("ਆਇਆ"): N("ਆਈ"),
                  N("ਵੇਗਾ"): N("ਵੇਗੀ"), N("ਊਗਾ"): N("ਊਗੀ"), N("ਸੁੱਤਾ"): N("ਸੁੱਤੀ")}     # future, ਸੌਣਾ
PA_FUTURE_MASCULINE = (N("ਵੇਗਾ"), N("ਊਗਾ"))
PA_FUTURE_FEMININE = (N("ਵੇਗੀ"), N("ਊਗੀ"))
# a feminine possessive shows the noun is feminine whatever the noun: ਮੇਰੀ ਸਹੇਲੀ, ਸਾਡੀ ਅਧਿਆਪਕ
PA_FEM_POSSESSIVES = {N(w) for w in "ਮੇਰੀ ਤੇਰੀ ਉਸਦੀ ਸਾਡੀ ਤੁਹਾਡੀ".split()}
# plural nouns as subjects: ਬੱਚੇ ਸਕੂਲ ਜਾਂਦਾ ਹੈ -> ਜਾਂਦੇ ਹਨ, ਕੁੜੀਆਂ ... -> ਜਾਂਦੀਆਂ ਹਨ
PA_PLURAL_NOUNS_M = {N(w) for w in "ਬੱਚੇ ਮੁੰਡੇ ਲੜਕੇ ਲੋਕ ਆਦਮੀ".split()}
PA_PLURAL_NOUNS_F = {N(w) for w in "ਕੁੜੀਆਂ ਲੜਕੀਆਂ ਔਰਤਾਂ ਭੈਣਾਂ ਸਹੇਲੀਆਂ".split()}
# a past time phrase (ਪਿਛਲੇ ਹਫ਼ਤੇ, ਪਿਛਲੇ ਸਾਲ) with a present habitual -> simple past (ਕੱਲ੍ਹ is left out: it also means tomorrow)
PA_PAST_TIME = {N(w) for w in "ਪਿਛਲੇ ਪਿਛਲੀ ਪਿਛਲਾ".split()}
PA_PERFECTIVE = {N("ਜਾਂ"): ("ਗਿਆ", "ਗਈ", "ਗਏ", "ਗਈਆਂ"), N("ਆਉਂ"): ("ਆਇਆ", "ਆਈ", "ਆਏ", "ਆਈਆਂ"),
                 N("ਸੌਂ"): ("ਸੁੱਤਾ", "ਸੁੱਤੀ", "ਸੁੱਤੇ", "ਸੁੱਤੀਆਂ")}
PA_INTRANSITIVE_REGULAR = {N(w) for w in "ਖੇਡ ਬੈਠ ਉੱਠ ਤੁਰ ਦੌੜ ਨੱਚ ਹੱਸ ਪਹੁੰਚ ਭੱਜ ਰੁਕ".split()}     # ਖੇਡਦਾ -> ਖੇਡਿਆ
# noun subjects with ਨੇ take their oblique form: ਮੁੰਡਾ -> ਮੁੰਡੇ ਨੇ, ਬੱਚੇ -> ਬੱਚਿਆਂ ਨੇ, ਮੇਰਾ ਭਰਾ -> ਮੇਰੇ ਭਰਾ ਨੇ
PA_OBLIQUE = {N(a): N(b) for a, b in [("ਮੁੰਡਾ", "ਮੁੰਡੇ"), ("ਲੜਕਾ", "ਲੜਕੇ"), ("ਬੱਚਾ", "ਬੱਚੇ"), ("ਬੱਚੇ", "ਬੱਚਿਆਂ"),
                                         ("ਮੁੰਡੇ", "ਮੁੰਡਿਆਂ"), ("ਲੜਕੇ", "ਲੜਕਿਆਂ"), ("ਲੋਕ", "ਲੋਕਾਂ"), ("ਮੇਰਾ", "ਮੇਰੇ"),
                                         ("ਤੇਰਾ", "ਤੇਰੇ"), ("ਉਸਦਾ", "ਉਸਦੇ"), ("ਸਾਡਾ", "ਸਾਡੇ"), ("ਤੁਹਾਡਾ", "ਤੁਹਾਡੇ")]}
PA_FEMININE_PERFECTIVE = {N(w) for w in "ਗਿਆ ਆਇਆ".split()}   # intransitive past without an auxiliary
PA_TO_PLURAL = {N("ਰਿਹਾ"): N("ਰਹੇ"), N("ਦਾ"): N("ਦੇ")}
PA_YESTERDAY = {N(w) for w in "ਕੱਲ੍ਹ ਕੱਲ ਕਲ ਕਲ੍ਹ".split()}   # ਕਲ: the recogniser's spelling
PA_ERGATIVE = {N("ਉਹ"): N("ਉਸਨੇ"), N("ਇਹ"): N("ਇਸਨੇ")}
PA_TRANSITIVE_PERFECTIVE = {N(w) for w in [
    "ਲਿਖੀ", "ਪੜ੍ਹੀ", "ਵੇਖੀ", "ਦੇਖੀ", "ਖਰੀਦੀ", "ਬਣਾਈ", "ਸੁਣੀ", "ਭੇਜੀ",
    "ਖਾਧਾ", "ਖਾਧੀ", "ਖਾਧੇ", "ਕੀਤਾ", "ਕੀਤੀ", "ਕੀਤੇ", "ਵੇਖਿਆ", "ਦੇਖਿਆ", "ਲਿਖਿਆ", "ਪੜ੍ਹਿਆ", "ਪੀਤਾ",
    "ਪੀਤੀ", "ਸੁਣਿਆ", "ਬਣਾਇਆ", "ਖਰੀਦਿਆ", "ਭੇਜਿਆ"]}
PA_DEDUPE = {N(w) for w in ["ਹੈ", "ਹਨ", "ਸੀ", "ਦਾ", "ਦੀ", "ਦੇ", "ਨੂੰ", "ਵਿੱਚ", "ਤੋਂ", "ਨੇ", "ਅਤੇ"]}


# the recogniser writes the nasal ending of ਅਸੀਂ/ਤੁਸੀਂ (we/you) as a full ਨ, live: "ਅਸੀਨ ਘਰ ਜਾਂਦਾ ਹੈ"
PA_PRONOUN_SPELLINGS = {N("ਅਸੀਨ"): N("ਅਸੀਂ"), N("ਤੁਸੀਨ"): N("ਤੁਸੀਂ")}


def _punjabi_tokens(t: list[str]) -> list[str]:
    t = [PA_PRONOUN_SPELLINGS.get(tok, tok) for tok in t]
    t = _dedupe(t, PA_DEDUPE)
    if len(t) < 2 or not _simple_clause(t, "pa-IN"):
        return t
    subj = t[0]

    # 1. subject-auxiliary agreement: ਮੈਂ ਜਾਂਦਾ ਹੈ -> ਮੈਂ ਜਾਂਦਾ ਹਾਂ, ਅਸੀਂ ਜਾਂਦਾ ਹੈ -> ਅਸੀਂ ਜਾਂਦੇ ਹਾਂ
    if subj in PA_SUBJECT_AUX and t[-1] in PA_AUX and len(t) >= 3 and t[-2].endswith(PA_IMPERFECTIVE):
        want = PA_SUBJECT_AUX[subj]
        if want is None:                      # ਉਹ / ਇਹ: only ਹਾਂ is wrong
            if t[-1] == N("ਹਾਂ"):
                t[-1] = N("ਹੈ")
        else:
            t[-1] = want
            if subj in PA_PLURAL_SUBJECTS:
                t[-2] = _swap_suffix(t[-2], PA_TO_PLURAL)

    # 0. plural noun subject, singular verb: ਬੱਚੇ ਰੋਜ਼ ਸਕੂਲ ਜਾਂਦਾ ਹੈ -> ਜਾਂਦੇ ਹਨ, ਕੁੜੀਆਂ ... -> ਜਾਂਦੀਆਂ ਹਨ
    plural = subj if subj in PA_PLURAL_NOUNS_M | PA_PLURAL_NOUNS_F else \
        (t[1] if len(t) > 2 and t[1] in PA_PLURAL_NOUNS_M | PA_PLURAL_NOUNS_F and subj in PA_OBLIQUE.values() else None)
    if plural and t[-1] == N("ਹੈ") and len(t) >= 3 and t[-2].endswith(PA_IMPERFECTIVE):
        t[-1] = N("ਹਨ")
        if plural in PA_PLURAL_NOUNS_M:
            t[-2] = _swap_suffix(t[-2], {N("ਦਾ"): N("ਦੇ"), N("ਦੀ"): N("ਦੇ"), N("ਰਿਹਾ"): N("ਰਹੇ")})
        else:
            t[-2] = _swap_suffix(t[-2], {N("ਦਾ"): N("ਦੀਆਂ"), N("ਦੀ"): N("ਦੀਆਂ"), N("ਰਿਹਾ"): N("ਰਹੀਆਂ")})

    # 1b. feminine subject, masculine verb: ਕੁੜੀ ਸਕੂਲ ਜਾਂਦਾ ਹੈ -> ਕੁੜੀ ਸਕੂਲ ਜਾਂਦੀ ਹੈ
    feminine = _feminine_subject(t, PA_FEMININE_NOUNS, PA_POSSESSIVES) or (len(t) > 2 and subj in PA_FEM_POSSESSIVES)
    if feminine and N("ਨੇ") not in t and \
            (t[-1] in {N("ਹੈ"), N("ਸੀ")} or t[-1] in PA_FEMININE_PERFECTIVE or t[-1].endswith(PA_FUTURE_MASCULINE)):
        _feminine_verb(t, {N("ਹੈ"), N("ਸੀ")}, PA_TO_FEMININE, {})

    # 1c. masculine subject, feminine verb: ਮੇਰਾ ਭਰਾ ਕੱਲ੍ਹ ਦਿੱਲੀ ਗਈ ਸੀ -> ਗਿਆ ਸੀ, ਮੁੰਡਾ ਸਕੂਲ ਜਾਂਦੀ ਹੈ -> ਜਾਂਦਾ ਹੈ
    masculine = subj in PA_MASCULINE_NOUNS or (len(t) > 2 and (subj in PA_MASC_POSSESSIVES or
                                                               (subj in PA_DEMONSTRATIVES and t[1] in PA_MASCULINE_NOUNS)))
    if masculine and N("ਨੇ") not in t and (t[-1] in {N("ਹੈ"), N("ਸੀ")} or t[-1] in PA_MASCULINE_PERFECTIVE
                                            or t[-1].endswith(PA_FUTURE_FEMININE)):
        _feminine_verb(t, {N("ਹੈ"), N("ਸੀ")}, PA_TO_MASCULINE, {})      # same steps, other direction

    # 2. ਕੱਲ੍ਹ (yesterday) + perfective + present aux: ਗਿਆ ਹੈ -> ਗਿਆ ਸੀ
    if PA_YESTERDAY & set(t) and len(t) >= 3 and t[-1] in {N("ਹੈ"), N("ਹਨ")}:
        verb = t[-2]
        if verb.endswith((N("ਆ"), N("ਈ"))):
            t[-1] = N("ਸੀ")
        elif verb.endswith(N("ਏ")):
            t[-1] = N("ਸਨ")

    # 2b. a past time phrase with a present habitual verb -> simple past:
    # ਮੁੰਡਾ ਪਿਛਲੇ ਹਫ਼ਤੇ ਬਾਜ਼ਾਰ ਜਾਂਦਾ ਹੈ -> ਮੁੰਡਾ ਪਿਛਲੇ ਹਫ਼ਤੇ ਬਾਜ਼ਾਰ ਗਿਆ (intransitive verbs only)
    if PA_PAST_TIME & set(t) and len(t) >= 3 and t[-1] in PA_AUX:
        verb = t[-2]
        for i, ending in enumerate((N("ਦਾ"), N("ਦੀ"), N("ਦੇ"), N("ਦੀਆਂ"))):
            if verb.endswith(ending):
                stem = verb[: -len(ending)]
                if stem in PA_PERFECTIVE:
                    t[-2:] = [N(PA_PERFECTIVE[stem][i])]
                elif stem in PA_INTRANSITIVE_REGULAR:
                    t[-2:] = [stem + N(("ਿਆ", "ੀ", "ੇ", "ੀਆਂ")[i])]
                break

    # 3. third-person ergative: ਉਹ ਰੋਟੀ ਖਾਧੀ -> ਉਸਨੇ ਰੋਟੀ ਖਾਧੀ
    last = t[-2] if t[-1] in {N("ਹੈ"), N("ਸੀ"), N("ਹਨ")} and len(t) >= 3 else t[-1]
    if subj in PA_ERGATIVE and N("ਨੇ") not in t:
        if last in PA_TRANSITIVE_PERFECTIVE:
            t[0] = PA_ERGATIVE[subj]
    # 3b. the same with a noun subject (not ਮੈਂ/ਅਸੀਂ/ਤੁਸੀਂ, which take no ਨੇ in Punjabi):
    # ਮੁੰਡਾ ਰੋਟੀ ਖਾਧੀ -> ਮੁੰਡੇ ਨੇ ਰੋਟੀ ਖਾਧੀ, ਮੇਰਾ ਦੋਸਤ ... -> ਮੇਰੇ ਦੋਸਤ ਨੇ ...
    elif N("ਨੇ") not in t and last in PA_TRANSITIVE_PERFECTIVE and not any(w.endswith(N("ਨੇ")) for w in t):
        nouns = PA_FEMININE_NOUNS | PA_MASCULINE_NOUNS | PA_PLURAL_NOUNS_M | PA_PLURAL_NOUNS_F
        possessive = PA_MASC_POSSESSIVES | PA_FEM_POSSESSIVES | set(PA_OBLIQUE.values()) | PA_DEMONSTRATIVES
        n = 1 if subj in nouns else 2 if len(t) >= 4 and subj in possessive else 0
        postpositions = {N(w) for w in "ਦੇ ਦੀ ਦਾ ਨੂੰ ਤੋਂ ਵਿੱਚ ਤੇ ਕੋਲ ਨਾਲ".split()}
        if n and len(t) >= n + 2 and t[n] not in postpositions:
            t = [PA_OBLIQUE.get(w, w) for w in t[:n]] + [N("ਨੇ")] + t[n:]
    return t


INDIC_RULES = {"hi-IN": _hindi_tokens, "mr-IN": _marathi_tokens, "bn-IN": _bengali_tokens, "pa-IN": _punjabi_tokens}


# --------------------------------------------------------------------------- LanguageTool (optional, English)

_language_tool = None


def _language_tool_correct(text: str) -> str:
    """Optional extra English pass with LanguageTool's public API (set ENABLE_LANGUAGETOOL=1)."""
    global _language_tool
    if os.getenv("ENABLE_LANGUAGETOOL", "").lower() not in {"1", "true", "yes"}:
        return text
    try:
        import language_tool_python
        if _language_tool is None:
            _language_tool = language_tool_python.LanguageToolPublicAPI("en-US")
        return _language_tool.correct(text)
    except Exception:
        return text  # network/rate-limit failure: keep the rule-based result


# --------------------------------------------------------------------------- entry point

# --------------------------------------------------------------------------- recogniser spellings

# Letters speech recognisers confuse because they sound alike. A rare word becomes a common word that
# differs by one such letter: বারি -> বাড়ি (live: "home" heard with র). Only Bengali and Hindi have word
# frequencies (wordfreq); see eval/spelling_check.py for the measured effect.
SOUND_ALIKE = {
    "bn-IN": [["র", "ড়", "ঢ়"], ["ন", "ণ"], ["শ", "ষ", "স"], ["জ", "য"], ["ি", "ী"], ["ু", "ূ"]],
    "hi-IN": [["ड़", "र"], ["ढ़", "ढ"], ["न", "ण"], ["श", "ष", "स"], ["ि", "ी"], ["ु", "ू"]],
}
RARE_BELOW, COMMON_FROM, MIN_GAIN = 4.0, 4.5, 1.5     # wordfreq Zipf: rare word -> clearly more common word


def spelling_repair_enabled() -> bool:
    return os.getenv("SPELLING_REPAIR", "1").lower() in {"1", "true", "yes"}


def _variants(word: str, groups: list[list[str]]):
    for group in groups:
        groups_n = [N(g) for g in group]
        for a in groups_n:
            start = word.find(a)
            while start != -1:
                for b in groups_n:
                    if b != a:
                        yield word[:start] + b + word[start + len(a):]
                start = word.find(a, start + 1)


def repair_spelling(tokens: list[str], lang_code: str) -> list[str]:
    """Replace a rare recogniser spelling with the clearly more common word one sound-alike letter away."""
    groups = SOUND_ALIKE.get(lang_code)
    if not groups:
        return tokens
    from wordfreq import zipf_frequency
    lang = lang_code[:2]
    out = []
    for tok in tokens:
        if len(tok) >= 2 and any(ch.isalpha() for ch in tok):
            z = zipf_frequency(tok, lang)
            if z < RARE_BELOW:
                best = max(_variants(tok, groups), key=lambda v: zipf_frequency(v, lang), default=None)
                if best is not None:
                    bz = zipf_frequency(best, lang)
                    if bz >= COMMON_FROM and bz >= z + MIN_GAIN:
                        tok = N(best)
        out.append(tok)
    return out


def _run(text: str, lang_code: str, apply_rules: bool, spelling: bool | None = None) -> str:
    lang = get_language(lang_code)
    text = normalize_unicode(text)
    if not text:
        return ""
    sentences = split_sentences(text) or [text]
    if lang.code == "en-US":
        out = " ".join(s for s in (_english_sentence(x, apply_rules) for x in sentences) if s)
        return normalize_unicode(_language_tool_correct(out)) if apply_rules else out
    if not apply_rules:
        fix = lambda tokens: tokens
    elif spelling_repair_enabled() if spelling is None else spelling:
        fix = lambda tokens: INDIC_RULES[lang.code](repair_spelling(tokens, lang.code))   # words first, then grammar
    else:
        fix = INDIC_RULES[lang.code]
    return " ".join(s for s in (_indic_sentence(x, lang, fix) for x in sentences) if s)


def correct(text: str, lang_code: str, spelling: bool | None = None) -> str:
    """Run Stage 1 (grammar rules + normalisation) on `text` and return the corrected text.
    spelling: repair recogniser spellings first (default: the SPELLING_REPAIR setting, on)."""
    return _run(text, lang_code, apply_rules=True, spelling=spelling)


def normalize(text: str, lang_code: str) -> str:
    """Only the normalisation part of Stage 1: spacing, capital letter, sentence-final mark."""
    return _run(text, lang_code, apply_rules=False)
