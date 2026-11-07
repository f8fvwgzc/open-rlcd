"""Lexical signals: tokens, BM25, term coverage, number support, query cues and (query, text) pair features.

Built for English, Japanese and Mongolian (Cyrillic) without any language model or dictionary:

* Latin and Cyrillic text splits into words (`\\w+`, casefolded); Japanese / CJK runs, which have no spaces,
  split into overlapping character bigrams (a single character stays a unigram). That is the standard,
  dictionary-free way to index CJK and it makes BM25 and coverage work across all three languages.
* Stopword lists are deliberately tiny (function words and question words); for Japanese, bigrams made only of
  hiragana count as function material.

What this does not do: no stemming or lemmatisation (except prefix matching of a few cue stems), no synonym
handling, no morphological analysis of Japanese. These are cheap features a calibrated head weighs; they are
not a search engine.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from decimal import Decimal, InvalidOperation

import numpy as np

# ----------------------------------------------------------------------------- tokens

_RUN = re.compile(r"[^\W_]+")
_CJK = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]")
_CJK_SPLIT = re.compile(r"([\u3040-\u30ff\u31f0-\u31ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]+)")
_HIRAGANA_ONLY = re.compile(r"^[\u3040-\u309f\u30fc]+$")


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKC", str(text or "")).casefold()


def _is_cjk(ch: str) -> bool:
    return bool(_CJK.match(ch))


def tokens(text: str) -> list[str]:
    """Casefolded tokens: words for Latin / Cyrillic / digits, character bigrams for CJK runs.

    `"Acme の売上は 1,200 円"` -> `["acme", "の売", "売上", "上は", "1", "200", "円"]`. Punctuation and
    underscores never form tokens."""
    out: list[str] = []
    for run in _RUN.findall(_norm(text)):
        if not _CJK.search(run):
            out.append(run)
            continue
        for part in _CJK_SPLIT.split(run):
            if not part:
                continue
            if _is_cjk(part[0]):
                if len(part) == 1:
                    out.append(part)
                else:
                    out.extend(part[i:i + 2] for i in range(len(part) - 1))
            else:
                out.append(part)
    return out


def _wordset(text: str) -> frozenset[str]:
    return frozenset(text.split())


STOPWORDS_EN = _wordset("""
a an the is are was were be been being am of to in on at for by with from as about into over under and or not no
nor but if then than so it its this that these those there here what who whom whose when where why how which do
does did done can could would should will shall may might must i you he she we they me him her us them my your our
their his hers ours theirs please tell show give me let know any some all each every just also very much many more
most such only own same too s t
""")
STOPWORDS_MN = _wordset("""
ба бол нь юм вэ бэ уу үү юу хэн хэзээ хаана яагаад яаж аль ямар энэ тэр эдгээр тэдгээр болон эсвэл гэж гэсэн
байна байгаа байсан байх гэх мөн бас ч л даа дээ шүү хэд хэдэн хэчнээн ямар нэг би чи та бид тэд миний таны
""")
STOPWORDS_JA = _wordset("""
何 誰 教え 下さ くだ 場合 こと もの これ それ あれ どれ この その あの どの です ます した する ある いる
""")
STOPWORDS = STOPWORDS_EN | STOPWORDS_MN | STOPWORDS_JA


def content_tokens(text: str) -> list[str]:
    """Tokens minus stopwords and minus hiragana-only CJK bigrams (particles, inflections)."""
    return [t for t in tokens(text) if t not in STOPWORDS and not _HIRAGANA_ONLY.match(t)]


# ----------------------------------------------------------------------------- BM25

class BM25:
    """Okapi BM25 over pre-tokenised documents, with an inverted index so a query touches only its terms.

    idf = ln(1 + (N - df + 0.5) / (df + 0.5)) (never negative). Each distinct query term counts once.
    Serialises to plain JSON (`to_dict` / `from_dict`); meant for local corpora up to some hundred thousand
    chunks, not for millions (use a search engine there)."""

    def __init__(self, docs: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = float(k1), float(b)
        self.doc_len = np.asarray([len(d) for d in docs], dtype=np.float64)
        self.n_docs = len(docs)
        self.avgdl = float(self.doc_len.mean()) if self.n_docs else 0.0
        post: dict[str, tuple[list[int], list[int]]] = {}
        for i, d in enumerate(docs):
            for term, tf in Counter(d).items():
                p = post.setdefault(term, ([], []))
                p[0].append(i)
                p[1].append(tf)
        self.postings = {t: (np.asarray(ix, dtype=np.int64), np.asarray(tf, dtype=np.float64))
                         for t, (ix, tf) in post.items()}

    def idf(self, term: str) -> float:
        p = self.postings.get(term)
        df = 0 if p is None else len(p[0])
        return math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        out = np.zeros(self.n_docs, dtype=np.float64)
        if not self.n_docs or self.avgdl <= 0:
            return out
        norm = self.k1 * (1.0 - self.b + self.b * self.doc_len / self.avgdl)
        for term in dict.fromkeys(query_tokens):
            p = self.postings.get(term)
            if p is None:
                continue
            ix, tf = p
            out[ix] += self.idf(term) * tf * (self.k1 + 1.0) / (tf + norm[ix])
        return out

    def to_dict(self) -> dict:
        return {"format": "open-rlcd-bm25/1", "k1": self.k1, "b": self.b,
                "doc_len": [int(x) for x in self.doc_len],
                "postings": {t: [ix.tolist(), [int(x) for x in tf]] for t, (ix, tf) in self.postings.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> BM25:
        obj = cls([], k1=float(d.get("k1", 1.2)), b=float(d.get("b", 0.75)))
        obj.doc_len = np.asarray(d.get("doc_len") or [], dtype=np.float64)
        obj.n_docs = len(obj.doc_len)
        obj.avgdl = float(obj.doc_len.mean()) if obj.n_docs else 0.0
        obj.postings = {t: (np.asarray(v[0], dtype=np.int64), np.asarray(v[1], dtype=np.float64))
                        for t, v in (d.get("postings") or {}).items()}
        return obj


def bm25_scores(query: str, texts: list[str]) -> np.ndarray:
    """BM25 of `query` against just these texts (idf from these texts only: scores are relative)."""
    return BM25([tokens(t) for t in texts]).scores(tokens(query))


def coverage(query: str, text: str) -> float:
    """Share of the query's distinct content tokens that occur in `text` (all tokens when the query has no
    content tokens; 0.0 for an empty query)."""
    q = set(content_tokens(query)) or set(tokens(query))
    if not q:
        return 0.0
    t = set(tokens(text))
    return len(q & t) / len(q)


# ----------------------------------------------------------------------------- numbers

_NUM = re.compile(
    r"(?<![\d.])(\d{1,3}(?:[,\u00a0\u202f']\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(\s?%|[万億兆]|\s?(?:million|billion|thousand|trillion|сая|тэрбум|мянган|мянга)(?!\w))?",
    re.IGNORECASE)
_MULT = {"万": Decimal(10) ** 4, "億": Decimal(10) ** 8, "兆": Decimal(10) ** 12, "thousand": Decimal(10) ** 3,
         "million": Decimal(10) ** 6, "billion": Decimal(10) ** 9, "trillion": Decimal(10) ** 12,
         "мянга": Decimal(10) ** 3, "мянган": Decimal(10) ** 3, "сая": Decimal(10) ** 6, "тэрбум": Decimal(10) ** 9}


def _canon(d: Decimal) -> str:
    if d == d.to_integral_value():
        return str(int(d))
    s = format(d.normalize(), "f")
    return s.rstrip("0").rstrip(".") if "." in s else s


def numbers(text: str) -> set[str]:
    """Normalised numeric tokens: `1,200` = `1200` = `１２００` -> "1200"; `12.50%` -> "12.5%"; `1.2 million`
    and `120万` -> "1200000". Thousands separators are `,`, NBSP, narrow NBSP and `'`; a comma not followed by
    exactly three digits separates numbers. Signs are ignored. Dates split into their parts."""
    out: set[str] = set()
    for m in _NUM.finditer(unicodedata.normalize("NFKC", str(text or ""))):
        raw = re.sub(r"[,\u00a0\u202f']", "", m.group(1))
        try:
            val = Decimal(raw)
        except InvalidOperation:
            continue
        suffix = (m.group(2) or "").strip().casefold()
        if suffix == "%":
            out.add(_canon(val) + "%")
        elif suffix in _MULT:
            out.add(_canon(val * _MULT[suffix]))
        else:
            out.add(_canon(val))
    return out


def number_support(answer: str, evidence: str) -> tuple[float, int]:
    """(share of the answer's numbers found in the evidence, number of unsupported ones); (1.0, 0) when the
    answer has no numbers. "12.5%" in the answer is supported by "12.5%" or a bare "12.5" in the evidence."""
    a = numbers(answer)
    if not a:
        return 1.0, 0
    e = numbers(evidence)
    e_bare = e | {x.rstrip("%") for x in e}
    supported = sum(1 for x in a if x in e or x.rstrip("%") in e_bare)
    return supported / len(a), len(a) - supported


# ----------------------------------------------------------------------------- query features

# Cue lists: plain words match a token exactly, words ending in "*" match a token prefix, entries containing
# CJK or a space match as substrings of the normalised query.
_QW: dict[str, tuple[str, ...]] = {
    "what": ("what", "何", "なに", "なん", "юу", "ямар"),
    "who": ("who", "whom", "whose", "誰", "だれ", "хэн*"),
    "when": ("when", "いつ", "хэзээ"),
    "where": ("where", "どこ", "хаана*"),
    "why": ("why", "なぜ", "何故", "どうして", "яагаад"),
    "how": ("how", "どう", "どのように", "どうやって", "яаж", "хэрхэн"),
    "which": ("which", "どの", "どれ", "どちら", "аль"),
    "how_many": ("how many", "how much", "いくつ", "いくら", "何人", "何件", "何個", "хэд", "хэдэн", "хэчнээн"),
}
_CUES: dict[str, tuple[str, ...]] = {
    "compare": ("compar*", "versus", "vs", "differ*", "better than", "worse than", "contrast*", "比較", "違い",
                "比べ", "差", "харьцуул*", "ялгаа*"),
    "list": ("list*", "enumerat*", "all the", "name all", "which ones", "一覧", "リスト", "列挙", "すべて", "全て",
             "жагсаа*", "бүх", "бүгд*"),
    "summary": ("summar*", "overview", "tl;dr", "tldr", "brief*", "recap", "要約", "まとめ", "概要", "хураангуй*",
                "товч*", "дүгнэ*"),
    "extract": ("extract*", "field*", "value of", "date of", "amount", "invoice number", "fill in", "抽出", "項目",
                "金額", "日付", "番号", "гаргаж ав*", "дүн*", "огноо*", "дугаар*"),
    "complaint": ("complain*", "refund*", "unacceptable", "terrible", "angry", "disappointed", "broken", "苦情",
                  "クレーム", "返金", "不満", "怒", "гомдол*", "буцаан олго*", "сэтгэл ханамжгүй"),
    "legal": ("legal*", "lawsuit*", "sue", "attorney*", "lawyer*", "liabilit*", "court*", "regulator*", "gdpr",
              "breach of contract", "法的", "訴訟", "弁護士", "契約違反", "裁判", "хууль*", "шүүх*", "өмгөөлөгч*",
              "гэрээ зөрчсөн"),
    "personal_data": ("email address", "phone number", "home address", "ssn", "social security", "passport*",
                      "password*", "credit card", "date of birth", "salary", "個人情報", "住所", "電話番号",
                      "メールアドレス", "パスワード", "給与", "生年月日", "マイナンバー", "хувийн мэдээлэл",
                      "утасны дугаар", "нууц үг", "регистрийн дугаар", "цалин*"),
}
_PRONOUNS = ("it", "this", "that", "they", "them", "those", "these", "he", "she", "それ", "これ", "あれ", "その",
             "この", "あの", "энэ", "тэр", "үүнийг", "түүнийг", "тэдгээр")
_QUOTES = re.compile(r"[\"“”„«»「」『』]|(?<!\w)'|'(?!\w)")
_JA_Q_PARTICLE = re.compile(r"(か|かな|ですか|ますか)\s*[?]?\s*$")
_MN_Q_PARTICLE = re.compile(r"(?:^|\s)(вэ|бэ|уу|үү|юу|бол)\s*[?]?\s*$")

QUERY_FEATURES: list[str] = [
    "q_log_chars", "q_log_tokens", "q_log_content_tokens",
    *[f"q_qw_{k}" for k in _QW],
    "q_has_digits", "q_has_quotes", "q_ends_question", "q_question_particle",
    "q_share_latin", "q_share_cjk", "q_share_cyrillic",
    *[f"q_cue_{k}" for k in _CUES],
    "q_has_pronoun", "q_very_short", "q_ambiguous",
]


def _matches(qnorm: str, toks: list[str], cues: tuple[str, ...]) -> bool:
    joined = f" {' '.join(toks)} "
    for c in cues:
        if _CJK.search(c):
            if c in qnorm:
                return True
        elif " " in c or ";" in c:
            if f" {c} " in joined or c in qnorm:
                return True
        elif c.endswith("*"):
            stem = c[:-1]
            if any(t.startswith(stem) for t in toks):
                return True
        elif c in toks:
            return True
    return False


def script_shares(text: str) -> tuple[float, float, float]:
    """(latin, cjk, cyrillic) shares of the letters in `text` (zeros when it has no letters)."""
    lat = cjk = cyr = total = 0
    for ch in unicodedata.normalize("NFKC", str(text or "")):
        if not ch.isalpha():
            continue
        total += 1
        if _is_cjk(ch):
            cjk += 1
        elif "\u0400" <= ch <= "\u04ff":
            cyr += 1
        elif ch.isascii() or "\u00c0" <= ch <= "\u024f":
            lat += 1
    if not total:
        return 0.0, 0.0, 0.0
    return lat / total, cjk / total, cyr / total


def dominant_script(text: str) -> str:
    shares = dict(zip(("latin", "cjk", "cyrillic"), script_shares(text), strict=True))
    best = max(shares, key=lambda k: shares[k])
    return best if shares[best] > 0 else "none"


def query_features(query: str) -> dict[str, float]:
    """Cheap cues about a query, in `QUERY_FEATURES` order; every key is always present (0.0 default)."""
    f = dict.fromkeys(QUERY_FEATURES, 0.0)
    qnorm = _norm(query).strip()
    toks = tokens(query)
    content = content_tokens(query)
    f["q_log_chars"] = math.log1p(len(qnorm))
    f["q_log_tokens"] = math.log1p(len(toks))
    f["q_log_content_tokens"] = math.log1p(len(content))
    for k, cues in _QW.items():
        f[f"q_qw_{k}"] = float(_matches(qnorm, toks, cues))
    f["q_has_digits"] = float(any(ch.isdigit() for ch in qnorm))
    f["q_has_quotes"] = float(bool(_QUOTES.search(qnorm)))
    f["q_ends_question"] = float(qnorm.endswith("?"))
    f["q_question_particle"] = float(bool(_JA_Q_PARTICLE.search(qnorm) or _MN_Q_PARTICLE.search(qnorm)))
    f["q_share_latin"], f["q_share_cjk"], f["q_share_cyrillic"] = script_shares(query)
    for k, cues in _CUES.items():
        f[f"q_cue_{k}"] = float(_matches(qnorm, toks, cues))
    has_pronoun = _matches(qnorm, toks, _PRONOUNS)
    f["q_has_pronoun"] = float(has_pronoun)
    f["q_very_short"] = float(len(toks) <= 2)
    f["q_ambiguous"] = float(len(content) <= 1 or (has_pronoun and len(content) <= 3))
    return f


# ----------------------------------------------------------------------------- pair features

PAIR_FEATURES: list[str] = ["p_coverage", "p_jaccard", "p_bm25", "p_numbers_overlap", "p_query_has_numbers",
                            "p_phrase_hit", "p_log_len_ratio", "p_same_script"]
_QUOTED = re.compile(r"[\"“„«「『]([^\"”»」』]{2,200})[\"”»」』]")


def _ngrams(seq: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)}


def phrase_hit(query: str, text: str) -> float:
    """1.0 when the query's quoted phrases all occur in `text` (normalised); without quotes, when some run of
    min(3, len) consecutive query tokens occurs in `text`'s token sequence."""
    tnorm = _norm(text)
    quoted = [_norm(q).strip() for q in _QUOTED.findall(unicodedata.normalize("NFKC", str(query or "")))]
    quoted = [q for q in quoted if q]
    if quoted:
        return float(all(q in tnorm for q in quoted))
    qt = tokens(query)
    if not qt:
        return 0.0
    n = min(3, len(qt))
    return float(bool(_ngrams(qt, n) & _ngrams(tokens(text), n)))


def pair_features(query: str, text: str, bm25: float | None = None) -> dict[str, float]:
    """(query, text) overlap signals in `PAIR_FEATURES` order. `bm25` is the caller's BM25 score of this text
    within its candidate set (0.0 when not given: BM25 needs a collection to mean anything)."""
    qt, tt = set(tokens(query)), set(tokens(text))
    union = qt | tt
    q_nums = numbers(query)
    lq, lt = len(tokens(query)), len(tokens(text))
    qs, ts = dominant_script(query), dominant_script(text)
    return {
        "p_coverage": coverage(query, text),
        "p_jaccard": len(qt & tt) / len(union) if union else 0.0,
        "p_bm25": float(bm25) if bm25 is not None and np.isfinite(bm25) else 0.0,
        "p_numbers_overlap": number_support(query, text)[0],
        "p_query_has_numbers": float(bool(q_nums)),
        "p_phrase_hit": phrase_hit(query, text),
        "p_log_len_ratio": math.log((1.0 + lt) / (1.0 + lq)),
        "p_same_script": float(qs == ts and qs != "none"),
    }
