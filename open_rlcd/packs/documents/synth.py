"""Deterministic synthetic RAG data with ground-truth labels for all eight documents heads.

    write_synth(home, encoder_config=None, seed=7, n_queries=900) -> dict

writes, under `home`:
  * the corpus of the fictional company in `world.py` (`Corpus.ingest_records`: chunks, vectors, graph, BM25);
  * label items for the four views and their labels with source "synthetic" (trusted) in `LabelStore(home)`.

Queries cover every response type in English, Japanese and Mongolian: single-document lookups (vector),
multi-hop questions that need a product sheet and its team page (graph), comparisons and lists across documents
(hybrid), ambiguous requests (clarify), complaints / legal threats / personal-data requests (escalate) and
unrelated chit-chat (out of scope), plus unanswerable ones (products that do not exist, attributes no document
states). Candidates for every item come from `Corpus.retrieve`, so retrieval features look like serving time;
`search.candidates` (the same call `decide()` makes) adds the documents the top hits link to; a gold chunk the
retriever still missed is appended (counted in the result as `gold_injected`).

Labels:
  * response_type / route / needs_graph from the query family;
  * chunk_relevant: gold chunks true, the other retrieved chunks (similar products, news naming the product)
    false;
  * answerable: gold evidence (+ distractors) true; evidence missing a gold chunk, or any evidence for a question
    about something the corpus does not contain, false;
  * grounded / complete / risk_tier on drafted answers: correct answers composed from the gold facts (grounded,
    complete 3), the same answer with one number / name / value changed (not grounded, still complete 3),
    partial answers (grounded, complete 1-2), answers about another entity (not grounded, complete 0); risk tier
    from the topic (specs 0, internal 1, money 2, legal / safety / medical 3).

Splits: `group` = the surface template (family, language, paraphrase, slot pattern), so the holdout is made of
phrasings the heads never saw; `t` = synthetic timestamps over nine months. Entities recur across templates: the
holdout measures generalisation to new phrasings of known documents, not to a new corpus. Metrics on this data
are a sanity check of the pipeline, not accuracy on anyone's real traffic.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ...features.text import make_encoder
from ...labeling.store import LabelItem, LabelStore
from .corpus import Corpus
from .search import candidates
from .settings import DEFAULT_ENCODER, DEFAULT_RETRIEVE_K
from .tasks import display_for
from .world import (
    DAY,
    LANGS,
    MISSING_NOUNS,
    PRODUCT_NOUNS,
    PRODUCT_RISK,
    T0,
    Doc,
    Fact,
    World,
    build_world,
    fake_product_name,
)

SPAN_DAYS = 270

# ----------------------------------------------------------------------------- templates

PREFIX = {"en": ["", "", "", "", "Hi, ", "Quick question: ", "Hello! ", "Please help: "],
          "ja": ["", "", "", "すみません、", "質問です。"], "mn": ["", "", "", "Сайн байна уу, ", "Асуулт байна: "]}
PROD_ATTR = {"en": ["What is the {a} of the {P}?", "Could you tell me the {a} of {P}?", "{P} {a}?",
                    "I need to know the {a} for the {P}."],
             "ja": ["{P}の{a}は何ですか？", "{P}の{a}を教えてください。"],
             "mn": ["{P}-ийн {a} юу вэ?", "{P}-ийн {a}-г хэлж өгнө үү."]}
PROD_EXTRACT = {"en": ["Give the {f1}, {f2} and {f3} of the {P}.", "Extract these fields for {P}: {f1}, {f2}, {f3}.",
                       "Fill in for the {P}: {f1} / {f2} / {f3}.", "I need the {f1}, {f2} and {f3} of {P}, please."],
                "ja": ["{P}の{f1}、{f2}、{f3}を抽出してください。", "{P}について{f1}・{f2}・{f3}の項目を埋めてください。"],
                "mn": ["{P}-ийн {f1}, {f2}, {f3}-г гаргаж өгнө үү.", "{P}: {f1}, {f2}, {f3} талбаруудыг бөглөнө үү."]}
EXTRACT_SETS = [("sku", "price", "warranty"), ("weight", "launch", "owner"), ("price", "weight", "owner")]
FACT_EXTRACT = {"en": ["From the {S}, extract the {f1}, {f2} and {f3}.",
                       "Fill in the following fields from the {S}: {f1}; {f2}; {f3}.",
                       "Pull out the {f1}, the {f2} and the {f3} stated in the {S}."],
                "ja": ["{S}から{f1}、{f2}、{f3}を抽出してください。", "{S}の{f1}・{f2}・{f3}の項目を教えてください。"],
                "mn": ["{S}-аас {f1}, {f2}, {f3}-г гаргаж өгнө үү.", "{S}-ийн {f1}, {f2}, {f3} талбаруудыг бөглөнө үү."]}
SUMMARIZE = {"en": ["Summarize the {X}.", "Give me a short overview of the {X}.", "What are the key points of the {X}?",
                    "TL;DR of the {X}, please."],
             "ja": ["{X}を要約してください。", "{X}の概要を教えてください。"],
             "mn": ["{X}-г товч хураангуйлна уу.", "{X}-ийн гол санааг товчхон хэлнэ үү."]}
COMPARE = {"en": ["Compare the {P1} and the {P2} on {A}.", "How do {P1} and {P2} differ in {A}?", "{P1} vs {P2}: {A}?",
                  "What is the difference in {A} between {P1} and {P2}?"],
           "ja": ["{P1}と{P2}の{A}を比較してください。", "{P1}と{P2}の{A}の違いは何ですか？"],
           "mn": ["{P1} болон {P2}-ийн {A}-г харьцуулна уу.", "{P1} ба {P2}-ийн {A}-ийн ялгаа юу вэ?"]}
COMPARE_SETS = [("price", "warranty"), ("weight",), ("price",), ("warranty",)]
AND = {"en": " and ", "ja": "と", "mn": " болон "}
LIST_TEAM = {"en": ["List all products owned by the {T} team.", "Which products does the {T} team own?",
                    "Name every product the {T} team is responsible for.", "Give me the full list of {T} products."],
             "ja": ["{T}チームが担当する製品をすべて挙げてください。", "{T}チームの製品一覧を教えてください。"],
             "mn": ["{T} багийн хариуцдаг бүх бүтээгдэхүүнийг жагсаана уу.", "{T} баг ямар ямар бүтээгдэхүүн хариуцдаг вэ?"]}
LIST_DEPT = {"en": ["Which teams belong to the {D} department?", "List the teams of the {D} department.",
                    "Name all teams in {D}."]}
FAKE_TEAMS = ["Harbor Analytics", "Quantum Freight", "Blue Routing", "Delta Packaging"]
MULTIHOP = {
    "lead": {"en": ["Who leads the team that owns the {P}?", "Who is the team lead responsible for {P}?",
                    "Which person heads the team behind the {P}?"],
             "ja": ["{P}を担当するチームのリーダーは誰ですか？", "{P}の担当チームを率いているのは誰ですか？"],
             "mn": ["{P}-г хариуцдаг багийн ахлагч хэн бэ?", "{P}-г хариуцсан багийг хэн удирддаг вэ?"]},
    "office": {"en": ["In which office is the team that owns the {P} based?", "Where does the team behind {P} work from?"],
               "ja": ["{P}を担当するチームの拠点オフィスはどこですか？"],
               "mn": ["{P}-г хариуцдаг баг аль оффист байрладаг вэ?"]},
    "department": {"en": ["Which department does the team that owns the {P} belong to?",
                          "The team responsible for {P} is in which department?"],
                   "ja": ["{P}の担当チームはどの部門に所属していますか？"],
                   "mn": ["{P}-г хариуцдаг баг аль хэлтэст харьяалагддаг вэ?"]},
}
MULTIHOP_EXTRACT = {"en": ["For the {P}, give the owning team, its team lead and its office.",
                           "Extract for {P}: owning team, team lead, office."],
                    "ja": ["{P}について、担当チーム・チームリーダー・拠点オフィスを抽出してください。"],
                    "mn": ["{P}-ийн хариуцдаг баг, багийн ахлагч, байрлах оффисыг гаргаж өгнө үү."]}
MULTIHOP_LIST = {"en": ["List the products owned by the team that {L} leads.",
                        "Which products is the team of {L} responsible for?"],
                 "ja": ["{L}さんが率いるチームの製品をすべて挙げてください。"],
                 "mn": ["{L}-ийн ахалдаг багийн бүх бүтээгдэхүүнийг жагсаана уу."]}
FACT_MISSING = {"en": ["How many days of parental leave do employees get?", "What is the pet policy at the Northgate office?",
                       "How much is the annual bonus for warehouse staff?", "What is the dress code for customer visits?",
                       "Which airline do we use for business travel?", "How many parking spaces does Port Allin have?"],
                "ja": ["育児休暇は何日取れますか？", "社員食堂の営業時間は何時までですか？"],
                "mn": ["Хүүхэд асрах чөлөө хэдэн хоног вэ?", "Ажлын хувцасны дүрэм ямар байдаг вэ?"]}
CLARIFY = {"en": ["What does it cost?", "How long is the warranty?", "Who is in charge of it?",
                  "Can you send me the details?", "What about the other one?", "When was it launched?",
                  "Is that still the policy?", "How heavy is it?"],
           "ja": ["それはいくらですか？", "保証はどのくらいですか？", "担当者は誰ですか？", "詳細を教えて。"],
           "mn": ["Энэ хэд вэ?", "Баталгаа нь хэр удаан бэ?", "Үүнийг хэн хариуцдаг вэ?", "Дэлгэрэнгүйг явуулаач."]}
ESCALATE = {
    "complaint": (2, {"en": ["My {P} arrived broken and I want a refund immediately.",
                             "This is unacceptable: the {P} stopped working after two days. I want to file a complaint.",
                             "I am very disappointed, my delivery is late again and nobody answers."],
                      "ja": ["{P}が壊れて届きました。すぐに返金してください。", "対応がひどすぎます。苦情を申し立てます。"],
                      "mn": ["{P} эвдэрсэн ирсэн, мөнгөө буцааж авмаар байна.", "Танай үйлчилгээнд гомдол гаргамаар байна."]}),
    "legal": (3, {"en": ["Our lawyer says you breached the contract. Who receives the legal notice?",
                         "We are preparing a lawsuit over the lost {P} shipment.",
                         "Is your company liable for the damage your driver caused?"],
                  "ja": ["契約違反の件で弁護士から連絡させます。", "{P}の事故について訴訟を検討しています。"],
                  "mn": ["Гэрээ зөрчсөн тул шүүхэд хандана.", "{P}-ийн асуудлаар өмгөөлөгчөөр дамжуулан хандана."]}),
    "personal": (2, {"en": ["What is the home address of {L}?", "Give me the phone number of {L}.",
                            "How much salary does {L} earn?"],
                     "ja": ["{L}さんの電話番号を教えてください。", "{L}さんの住所はどこですか？"],
                     "mn": ["{L}-ийн утасны дугаарыг өгөөч.", "{L}-ийн цалин хэд вэ?"]}),
}
OUT_OF_SCOPE = {"en": ["What's the weather in Lisbon tomorrow?", "Who won the football match last night?",
                       "Can you recommend a good pasta recipe?", "Tell me a joke about cats.",
                       "How do I fix a squeaky bicycle chain?", "What time is it in Sydney right now?",
                       "Write a short poem about autumn.", "What is the capital of Australia?"],
                "ja": ["明日の東京の天気はどうですか？", "おすすめのラーメン屋を教えてください。", "昨日のサッカーの試合はどちらが勝ちましたか？"],
                "mn": ["Маргааш цаг агаар ямар байх вэ?", "Өчигдрийн хөлбөмбөгийн тоглолтод хэн хожсон бэ?",
                       "Сайн хоолны жор зөвлөөч."]}
FAMILIES = [("prod_attr", 0.20), ("prod_extract", 0.07), ("fact", 0.11), ("fact_missing", 0.03),
            ("fact_extract", 0.04), ("summarize", 0.08), ("compare", 0.08), ("list", 0.07), ("multihop", 0.09),
            ("multihop_extract", 0.03), ("multihop_list", 0.03), ("clarify", 0.06), ("escalate", 0.06),
            ("out_of_scope", 0.05)]
QUERY_LANG = {"en": 0.6, "ja": 0.2, "mn": 0.2}

# answer sentences
FACT_SENT = {"en": ["The {noun} of {s} is {v}.", "For {s}, the {noun} is {v}.", "{s}: {noun} {v}.",
                    "According to the documents, the {noun} of {s} is {v}."],
             "ja": ["{s}の{noun}は{v}です。", "{s}について、{noun}は{v}です。", "{s}：{noun} {v}。"],
             "mn": ["{s}-ийн {noun} нь {v}.", "{s}: {noun} {v}.", "Баримтын дагуу {s}-ийн {noun} нь {v}."]}
LIST_SENT = {"en": ["The products owned by {T} are {items}.", "{T} owns {items}."],
             "ja": ["{T}が担当する製品は{items}です。", "{T}の製品：{items}。"],
             "mn": ["{T}-ийн хариуцдаг бүтээгдэхүүнүүд: {items}.", "{T} баг {items}-г хариуцдаг."]}
DEPT_SENT = {"en": "The {D} department has these teams: {items}."}
LEAD_SENT = {"en": "{L} leads the {T} team.", "ja": "{L}は{T}のチームリーダーです。", "mn": "{L} нь {T} багийн ахлагч."}


def _join(items: list[str], lang: str) -> str:
    if lang == "ja":
        return "、".join(items)
    if lang == "mn":
        return ", ".join(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


# ----------------------------------------------------------------------------- one query

@dataclass
class Q:
    text: str
    lang: str
    family: str
    template: str
    response_type: str
    route: str
    risk: int
    gold: list[str] = field(default_factory=list)          # chunk ids the full answer needs
    answerable: bool = False                               # the gold, when all present, answers it
    facts: list[tuple[str, Fact]] = field(default_factory=list)   # (subject, fact) the full answer states
    items: list[str] = field(default_factory=list)         # list answers: the items
    item_kind: str = ""                                    # list answers: "team" (products of T) | "dept" | ""
    list_subject: str = ""
    intro: tuple[str, str] | None = None                   # multihop list: (lead, team)

    @property
    def needs_graph(self) -> bool:
        return self.route == "graph"


def _noun(f: Fact, lang: str) -> str:
    return f.noun.get(lang) or next(iter(f.noun.values()))


def _fill(rng: random.Random, lang: str, template: str, **kw: str) -> str:
    text = template.format(**kw)
    pre = rng.choice(PREFIX[lang])
    if pre and lang == "en" and not template.startswith("{"):
        text = text[0].lower() + text[1:]
    return pre + text


class Generator:
    def __init__(self, world: World, seed: int):
        self.w = world
        self.rng = random.Random(f"queries:{seed}")
        self.taken = set(world.names["product"]) | set(world.names["team"])
        self.by_name = {d.subject: d for d in world.products}
        self.team_doc = {d.subject: d for d in world.teams}

    def lang(self, allowed: tuple[str, ...] = LANGS) -> str:
        langs = [x for x in QUERY_LANG if x in allowed]
        return self.rng.choices(langs, weights=[QUERY_LANG[x] for x in langs])[0]

    def query(self) -> Q:
        fam = self.rng.choices([f for f, _ in FAMILIES], weights=[w for _, w in FAMILIES])[0]
        return getattr(self, f"_{fam}")()

    # ------------------------------------------------------------------ families
    def _prod_attr(self) -> Q:
        r = self.rng
        lang = self.lang()
        pi = r.randrange(len(PROD_ATTR[lang]))
        missing = r.random() < 0.12
        fake = not missing and r.random() < 0.12
        attr = r.choice(list(MISSING_NOUNS)) if missing else r.choice(list(PRODUCT_NOUNS))
        noun = (MISSING_NOUNS if missing else PRODUCT_NOUNS)[attr][lang]
        p = fake_product_name(r, self.taken) if fake else r.choice(self.w.products).subject
        q = Q(_fill(r, lang, PROD_ATTR[lang][pi], a=noun, P=p), lang, "prod_attr", f"prod_attr:{lang}:{pi}:{attr}",
              "direct_answer", "vector", 0 if missing else PRODUCT_RISK[attr])
        if not (fake or missing):
            d = self.by_name[p]
            q.gold, q.answerable, q.facts = [d.chunk_id], True, [(p, d.facts[attr])]
        return q

    def _prod_extract(self) -> Q:
        r = self.rng
        lang = self.lang()
        pi, si = r.randrange(len(PROD_EXTRACT[lang])), r.randrange(len(EXTRACT_SETS))
        fields = list(EXTRACT_SETS[si])
        fake = r.random() < 0.12
        missing = not fake and r.random() < 0.12
        nouns = [PRODUCT_NOUNS[f][lang] for f in fields]
        if missing:
            nouns[r.randrange(3)] = MISSING_NOUNS[r.choice(list(MISSING_NOUNS))][lang]
        p = fake_product_name(r, self.taken) if fake else r.choice(self.w.products).subject
        q = Q(_fill(r, lang, PROD_EXTRACT[lang][pi], P=p, f1=nouns[0], f2=nouns[1], f3=nouns[2]), lang,
              "prod_extract", f"prod_extract:{lang}:{pi}:{si}", "extract_fields", "vector",
              max(PRODUCT_RISK[f] for f in fields))
        if not (fake or missing):
            d = self.by_name[p]
            q.gold, q.answerable, q.facts = [d.chunk_id], True, [(p, d.facts[f]) for f in fields]
        return q

    def _fact_doc(self) -> Doc:
        # legal / safety / medical documents are rarer in a corpus but matter most: ask about them twice as often
        docs = self.w.fact_docs
        return self.rng.choices(docs, weights=[2.0 if d.risk == 3 else 1.0 for d in docs])[0]

    def _fact(self) -> Q:
        r = self.rng
        d = self._fact_doc()
        f = d.facts[r.choice(list(d.facts))]
        qi = r.randrange(len(f.questions[d.lang]))
        q = Q(_fill(r, d.lang, f.questions[d.lang][qi]), d.lang, "fact", f"fact:{d.id}:{f.key}:{qi}",
              "direct_answer", "vector", d.risk, [d.chunk_id], True, [(d.subject, f)])
        return q

    def _fact_missing(self) -> Q:
        r = self.rng
        lang = self.lang()
        i = r.randrange(len(FACT_MISSING[lang]))
        return Q(_fill(r, lang, FACT_MISSING[lang][i]), lang, "fact_missing", f"fact_missing:{lang}:{i}",
                 "direct_answer", "vector", 1)

    def _fact_extract(self) -> Q:
        r = self.rng
        d = self._fact_doc()
        keys = list(d.facts)[:3]
        if len(keys) < 3:
            return self._fact()
        pi = r.randrange(len(FACT_EXTRACT[d.lang]))
        nouns = [_noun(d.facts[k], d.lang) for k in keys]
        return Q(_fill(r, d.lang, FACT_EXTRACT[d.lang][pi], S=d.subject, f1=nouns[0], f2=nouns[1], f3=nouns[2]),
                 d.lang, "fact_extract", f"fact_extract:{d.lang}:{pi}", "extract_fields", "vector", d.risk,
                 [d.chunk_id], True, [(d.subject, d.facts[k]) for k in keys])

    def _summarize(self) -> Q:
        r = self.rng
        kind = r.choice(["product", "team", "policy"])
        if kind == "product":
            d = r.choice(self.w.products)
            lang = self.lang()
            x, keys, risk = d.subject, ["price", "warranty", "owner"], 0
        elif kind == "team":
            d = r.choice(self.w.teams)
            lang = self.lang(("en", d.lang))
            x = {"en": f"{d.subject} team", "ja": f"{d.subject}チーム", "mn": f"{d.subject} баг"}[lang]
            keys, risk = ["department", "office", "lead"], 1
        else:
            d = self._fact_doc()
            lang, x, keys, risk = d.lang, d.subject, list(d.facts)[:3], d.risk
        pi = r.randrange(len(SUMMARIZE[lang]))
        return Q(_fill(r, lang, SUMMARIZE[lang][pi], X=x), lang, "summarize", f"summarize:{lang}:{pi}:{kind}",
                 "summarize", "vector", risk, [d.chunk_id], True, [(d.subject, d.facts[k]) for k in keys])

    def _compare(self) -> Q:
        r = self.rng
        lang = self.lang()
        pi, si = r.randrange(len(COMPARE[lang])), r.randrange(len(COMPARE_SETS))
        attrs = COMPARE_SETS[si]
        a, b = r.sample(self.w.products, 2)
        fake = r.random() < 0.12
        p2 = fake_product_name(r, self.taken) if fake else b.subject
        phrase = AND[lang].join(PRODUCT_NOUNS[x][lang] for x in attrs)
        q = Q(_fill(r, lang, COMPARE[lang][pi], P1=a.subject, P2=p2, A=phrase), lang, "compare",
              f"compare:{lang}:{pi}:{si}", "compare", "hybrid", max(PRODUCT_RISK[x] for x in attrs))
        if not fake:
            q.gold, q.answerable = [a.chunk_id, b.chunk_id], True
            q.facts = [(a.subject, a.facts[x]) for x in attrs] + [(b.subject, b.facts[x]) for x in attrs]
        return q

    def _list(self) -> Q:
        r = self.rng
        if r.random() < 0.3:
            dept = r.choice([d for d, ts in self.w.departments.items() if len(ts) >= 2])
            pi = r.randrange(len(LIST_DEPT["en"]))
            teams = self.w.departments[dept]
            return Q(_fill(r, "en", LIST_DEPT["en"][pi], D=dept), "en", "list", f"list_dept:en:{pi}", "list", "hybrid",
                     1, [self.team_doc[t].chunk_id for t in teams], True, items=list(teams), item_kind="dept",
                     list_subject=dept)
        lang = self.lang()
        pi = r.randrange(len(LIST_TEAM[lang]))
        fake = r.random() < 0.1
        team = r.choice(FAKE_TEAMS) if fake else r.choice([t for t, ps in self.w.team_products.items() if ps])
        q = Q(_fill(r, lang, LIST_TEAM[lang][pi], T=team), lang, "list", f"list_team:{lang}:{pi}", "list", "hybrid", 1)
        if not fake:
            names = self.w.team_products[team]
            q.gold, q.answerable = [self.by_name[n].chunk_id for n in names], True
            q.items, q.item_kind, q.list_subject = list(names), "team", team
        return q

    def _multihop(self) -> Q:
        r = self.rng
        rel = r.choice(list(MULTIHOP))
        lang = self.lang()
        pi = r.randrange(len(MULTIHOP[rel][lang]))
        fake = r.random() < 0.12
        d = None if fake else r.choice(self.w.products)
        p = fake_product_name(r, self.taken) if fake else d.subject
        q = Q(_fill(r, lang, MULTIHOP[rel][lang][pi], P=p), lang, "multihop", f"multihop:{lang}:{pi}:{rel}",
              "direct_answer", "graph", 0 if rel == "office" else 1)
        if d is not None:
            t = self.team_doc[d.meta["team"]]
            q.gold, q.answerable = [d.chunk_id, t.chunk_id], True
            q.facts = [(p, d.facts["owner"]), (t.subject, t.facts[rel])]
        return q

    def _multihop_extract(self) -> Q:
        r = self.rng
        lang = self.lang()
        pi = r.randrange(len(MULTIHOP_EXTRACT[lang]))
        d = r.choice(self.w.products)
        t = self.team_doc[d.meta["team"]]
        return Q(_fill(r, lang, MULTIHOP_EXTRACT[lang][pi], P=d.subject), lang, "multihop_extract",
                 f"multihop_extract:{lang}:{pi}", "extract_fields", "graph", 1, [d.chunk_id, t.chunk_id], True,
                 [(d.subject, d.facts["owner"]), (t.subject, t.facts["lead"]), (t.subject, t.facts["office"])])

    def _multihop_list(self) -> Q:
        r = self.rng
        lang = self.lang()
        pi = r.randrange(len(MULTIHOP_LIST[lang]))
        team = r.choice([t for t, ps in self.w.team_products.items() if ps])
        lead = self.w.leads[team]
        names = self.w.team_products[team]
        return Q(_fill(r, lang, MULTIHOP_LIST[lang][pi], L=lead), lang, "multihop_list", f"multihop_list:{lang}:{pi}",
                 "list", "graph", 1, [self.team_doc[team].chunk_id, *(self.by_name[n].chunk_id for n in names)], True,
                 items=list(names), item_kind="team", list_subject=team, intro=(lead, team))

    def _clarify(self) -> Q:
        lang = self.lang()
        i = self.rng.randrange(len(CLARIFY[lang]))
        return Q(_fill(self.rng, lang, CLARIFY[lang][i]), lang, "clarify", f"clarify:{lang}:{i}", "clarify_question",
                 "none", 0)

    def _escalate(self) -> Q:
        r = self.rng
        kind = r.choice(list(ESCALATE))
        risk, temps = ESCALATE[kind]
        lang = self.lang()
        i = r.randrange(len(temps[lang]))
        text = _fill(r, lang, temps[lang][i], P=r.choice(self.w.products).subject,
                     L=r.choice(self.w.names["person"]))
        return Q(text, lang, "escalate", f"escalate:{lang}:{kind}:{i}", "escalate_human", "none", risk)

    def _out_of_scope(self) -> Q:
        lang = self.lang()
        i = self.rng.randrange(len(OUT_OF_SCOPE[lang]))
        return Q(_fill(self.rng, lang, OUT_OF_SCOPE[lang][i]), lang, "out_of_scope", f"oos:{lang}:{i}",
                 "out_of_scope", "none", 0)


# ----------------------------------------------------------------------------- answers

def fact_sentence(subject: str, f: Fact, lang: str, value: str | None = None, style: int = 0) -> str:
    temps = FACT_SENT[lang]
    return temps[style % len(temps)].format(noun=_noun(f, lang), s=subject, v=f.value[lang] if value is None else value)


def corrupt_value(f: Fact, lang: str, rng: random.Random, world: World, avoid: str) -> str:
    """The fact's value with one thing changed so that the evidence no longer supports it."""
    v = f.value[lang]
    if f.kind in ("name",) and f.entity_kind:
        pool = [n for n in world.names[f.entity_kind] if n != v and n not in avoid]
        if pool:
            return rng.choice(pool)
    if f.kind == "text" and f.alt:
        return rng.choice(f.alt)
    if f.kind == "code":
        return v[:-3] + "".join(rng.choice("0123456789") for _ in range(3)) if v[-3:] != "000" else v[:-3] + "111"
    if f.kind == "date":
        y, m, d = v.split("-")
        nd = (int(d) + rng.randint(3, 15)) % 28 + 1
        nd = nd if nd != int(d) else (nd % 28) + 1
        return f"{y}-{m}-{nd:02d}"
    # numbers (with units, thousands separators, decimals, %)
    m = re.search(r"\d[\d,]*(?:\.\d+)?", v)
    if not m:
        return v + " (approx.)"
    raw = m.group(0)
    num = float(raw.replace(",", ""))
    if "." in raw:
        new = round(num + rng.choice([-1, 1]) * rng.uniform(0.4, 2.5), 1)
        new = new if new > 0 and new != num else round(num + 1.3, 1)
        txt = f"{new}"
    else:
        delta = max(1, int(round(num * rng.uniform(0.12, 0.4))))
        new_i = int(num) + (delta if rng.random() < 0.6 or num - delta <= 0 else -delta)
        txt = f"{new_i:,}" if "," in raw else str(new_i)
    return v[: m.start()] + txt + v[m.end():]


def compose(q: Q, lang: str, keep: list[int] | None = None, corrupt: int | None = None, rng: random.Random | None = None,
            world: World | None = None, avoid: str = "", style: int = 0) -> str:
    """The answer text: all facts / items (or only `keep`), with fact or item number `corrupt` changed, in wording
    number `style`."""
    parts = []
    if q.items:
        items = list(q.items)
        idx = list(range(len(items))) if keep is None else keep
        chosen = [items[i] for i in idx]
        if corrupt is not None and rng is not None and world is not None:
            kind = "team" if q.item_kind == "dept" else "product"
            pool = [n for n in world.names[kind] if n not in items and n not in avoid]
            chosen[corrupt % len(chosen)] = rng.choice(pool)
        if q.intro:
            parts.append(LEAD_SENT[lang].format(L=q.intro[0], T=q.intro[1]))
        if q.item_kind == "dept":
            parts.append(DEPT_SENT["en"].format(D=q.list_subject, items=_join(chosen, "en")))
        else:
            temps = LIST_SENT[lang]
            parts.append(temps[style % len(temps)].format(T=q.list_subject, items=_join(chosen, lang)))
        return " ".join(parts) if lang == "en" or lang == "mn" else "".join(parts)
    idx = list(range(len(q.facts))) if keep is None else keep
    for i in idx:
        subject, f = q.facts[i]
        value = None
        if corrupt == i and rng is not None and world is not None:
            value = corrupt_value(f, lang, rng, world, avoid)
        parts.append(fact_sentence(subject, f, lang, value, style))
    return ("" if lang == "ja" else " ").join(parts)


def complete_level(k: int, n: int) -> int:
    if n <= 0 or k <= 0:
        return 0
    if k >= n:
        return 3
    return 2 if k / n >= 2 / 3 else 1


# ----------------------------------------------------------------------------- candidates and evidence

def _inject(corpus: Corpus, query: str, cands: list[dict], gold: list[str]) -> tuple[list[dict], int]:
    """Append gold chunks the retriever missed (scored by cosine like every candidate); returns (list, n added)."""
    have = {c["id"] for c in cands}
    missing = [g for g in gold if g not in have and corpus.chunk(g) is not None]
    if not missing:
        return cands, 0
    qv = np.asarray(corpus.encoder.encode([query], kind="query"), np.float32)[0]
    qn = qv / (np.linalg.norm(qv) or 1.0)
    out = list(cands)
    for g in missing:
        ch = corpus.chunk(g)
        s = float(corpus.vectors.vectors([g])[0] @ qn)
        out.append({"id": g, "doc_id": ch["doc_id"], "title": ch.get("title") or "", "text": ch["text"],
                    "score": s, "source": "vector", "sources": ["vector"], "rrf": 0.0, "lexical_score": None,
                    "updated_at": ch.get("updated_at"), "authority": ch.get("authority"), "rank": len(out) + 1})
    return out, len(missing)


# ----------------------------------------------------------------------------- writing

def _iso(t: int) -> str:
    return datetime.fromtimestamp(t, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_synth(home: Path, encoder_config: dict | None = None, seed: int = 7, n_queries: int = 900) -> dict:
    """Write the synthetic corpus and labelled items into `home` (an RLCD_HOME); returns counts per view, head and
    class. `encoder_config` must be the encoder the project trains with (pack_config.encoder): None means the
    pack default (ONNX multilingual-e5-small, downloaded once); offline use {"kind": "hashing", "dim": 256}.
    Re-running with the same seed adds nothing new (ids are deterministic; labels are re-appended, latest wins)."""
    home = Path(home)
    world = build_world(seed)
    encoder = make_encoder(encoder_config or DEFAULT_ENCODER, home)
    corpus = Corpus(home, encoder, {"graph": {"kind": "local", "terms": world.terms}})
    ingest = corpus.ingest_records([d.record() for d in world.docs.values()])
    gen = Generator(world, seed)
    rng = random.Random(f"labels:{seed}")
    t_rng = random.Random(f"time:{seed}")
    items: list[LabelItem] = []
    labels: list[tuple[str, str, Any]] = []
    langs: dict[str, int] = {}
    injected = 0
    for i in range(int(n_queries)):
        q = gen.query()
        qid = f"syn{seed}-{i:05d}"
        t = T0 + int(t_rng.uniform(0, SPAN_DAYS) * DAY)
        now = _iso(t)
        langs[q.lang] = langs.get(q.lang, 0) + 1
        raw = candidates(corpus, q.text, DEFAULT_RETRIEVE_K)

        def add(suffix: str, view: str, state: dict, heads: dict[str, Any], _q: Q = q, _t: int = t,
                _qid: str = qid) -> None:
            it = LabelItem(id=f"{_qid}:{suffix}", view=view, state=state, display=display_for(view, state),
                           group=_q.template, t=_t, meta={"family": _q.family, "lang": _q.lang, "synthetic": True})
            items.append(it)
            labels.extend((it.id, h, v) for h, v in heads.items())

        add("q", "query", {"query": q.text, "candidates": raw, "now": now},
            {"response_type": q.response_type, "route": q.route, "needs_graph": q.needs_graph})
        cands, n_inj = _inject(corpus, q.text, raw, q.gold) if q.gold else (raw, 0)
        injected += n_inj
        gold = set(q.gold)
        others = [c for c in cands if c["id"] not in gold]
        # ---- chunk_relevant: every gold chunk + the strongest look-alikes (+ one from further down)
        if q.family not in ("escalate",) and (q.family not in ("clarify", "out_of_scope") or rng.random() < 0.5):
            pick = [c for c in cands if c["id"] in gold]
            negs = others[:3] if q.gold else others[:2]
            rest = others[3:]
            if rest and rng.random() < 0.5:
                negs = [*negs, rng.choice(rest)]
            for c in [*pick, *negs]:
                add(f"c:{c['id']}", "query_chunk", {"query": q.text, "chunk": c, "candidates": cands, "now": now},
                    {"chunk_relevant": c["id"] in gold})
        # ---- answerable
        good_ev: list[dict] = []
        if q.answerable:
            k = rng.randint(0, 3)
            good_ev = sorted([c for c in cands if c["id"] in gold] + others[:k], key=lambda c: c["rank"])
            add("e:good", "query_evidence", {"query": q.text, "evidence": good_ev, "candidates": cands, "now": now},
                {"answerable": True})
            if rng.random() < 0.7:
                if len(q.gold) >= 2:
                    drop = rng.choice(q.gold)
                    bad = [c for c in good_ev if c["id"] != drop]
                else:
                    bad = others[: rng.randint(1, 4)]
                add("e:bad", "query_evidence", {"query": q.text, "evidence": bad, "candidates": cands, "now": now},
                    {"answerable": False})
        elif q.family != "escalate" and (q.family not in ("clarify", "out_of_scope") or rng.random() < 0.5):
            ev = cands[: rng.randint(1, 5)]
            add("e:none", "query_evidence", {"query": q.text, "evidence": ev, "candidates": cands, "now": now},
                {"answerable": False})
        # ---- evaluator rows on drafted answers
        if q.answerable and good_ev:
            ev_text = " ".join(c.get("text") or "" for c in good_ev)
            n = len(q.items) if q.items else len(q.facts)
            drafts = [("full", compose(q, q.lang, style=rng.randrange(4)), True, 3)]
            if rng.random() < 0.85:
                drafts.append(("corrupt", compose(q, q.lang, corrupt=rng.randrange(n), rng=rng, world=world,
                                                  avoid=ev_text, style=rng.randrange(4)), False, 3))
            if n >= 2 and rng.random() < 0.6:
                k = rng.randint(1, n - 1)
                keep = sorted(rng.sample(range(n), k))
                drafts.append(("partial", compose(q, q.lang, keep=keep, style=rng.randrange(4)), True,
                               complete_level(k, n)))
            if rng.random() < 0.3:
                off = _off_topic(world, q, good_ev, rng)
                if off:
                    drafts.append(("off", off, False, 0))
            for name, text, grounded, level in drafts:
                add(f"a:{name}", "query_evidence_answer", {"query": q.text, "evidence": good_ev, "answer": text},
                    {"grounded": grounded, "complete": level, "risk_tier": q.risk})
    store = LabelStore(home)
    added = store.add_items(items)
    for item_id, head, value in labels:
        store.add_human(item_id, head, value, source="synthetic", annotator=f"synth:{seed}")
    counts: dict[str, dict] = {}
    for _, head, value in labels:
        c = counts.setdefault(head, {"rows": 0, "classes": {}})
        c["rows"] += 1
        key = str(value).lower() if isinstance(value, bool) else str(value)
        c["classes"][key] = c["classes"].get(key, 0) + 1
    by_view: dict[str, int] = {}
    for it in items:
        by_view[it.view] = by_view.get(it.view, 0) + 1
    return {"home": str(home), "seed": seed, "n_queries": int(n_queries), "encoder": encoder.id,
            "corpus": {"docs": ingest["docs_total"], "chunks": ingest["chunks_total"]}, "items": by_view,
            "items_added": added, "labels": counts, "languages": langs, "gold_injected": injected,
            "templates": len({it.group for it in items})}


def _off_topic(world: World, q: Q, evidence: list[dict], rng: random.Random) -> str | None:
    """A fluent answer about some other entity whose document is not in the evidence: not grounded, complete 0."""
    ids = {c["id"] for c in evidence}
    pool = [d for d in [*world.products, *world.fact_docs] if d.chunk_id not in ids and d.facts]
    if not pool:
        return None
    d = rng.choice(pool)
    keys = [k for k, f in d.facts.items() if f.kind in ("number", "months", "date", "text")]
    if not keys:
        return None
    f = d.facts[rng.choice(keys)]
    lang = q.lang if (d.kind == "product" or d.lang == q.lang) else d.lang
    return fact_sentence(d.subject, f, lang, style=rng.randrange(4))
