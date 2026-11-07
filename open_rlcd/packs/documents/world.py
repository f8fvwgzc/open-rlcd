"""The fictional company behind the synthetic RAG data: Kestrovia Logistics (invented; any resemblance to a real
organisation, product or person is accidental).

    departments -> teams (department, office, lead) -> products (SKU, price, warranty, weight, launch, team)
    policies and offices (facts with numbers and dates), announcements (hard negatives: they name products
    without stating any of their facts)

Documents are mostly English; the teams based in Osaka write Japanese pages, the teams in Ulaanbaatar write
Mongolian (Cyrillic) pages, and one policy and one office page exist in each of those languages. Product, team
and code names stay in Latin script in every language (as they do in real multilingual companies), which is what
lets a query in one language find a document in another through names and codes.

The entity links that make multi-hop questions: a product sheet names its owning team (and links to the team
page); the team page names the department, the office and the team lead, but never the products. So "who leads
the team that owns X?" needs two documents.

Everything here is a deterministic function of the seed (`build_world(seed)`): prices, weights, dates, which
product belongs to which team and the order of everything.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import UTC, datetime

COMPANY = "Kestrovia Logistics"
LANGS = ("en", "ja", "mn")
T0 = int(datetime(2025, 1, 6, tzinfo=UTC).timestamp())
DAY = 86400


@dataclass
class Fact:
    """One answerable fact of one document. `value` is the value as an answer in each language states it: the
    document's own wording for names and text, the asker's unit words for months / kilograms (the number is what
    must match the document)."""
    key: str
    noun: dict[str, str]
    value: dict[str, str]
    kind: str                       # number | months | date | name | code | text
    risk: int
    questions: dict[str, list[str]] = field(default_factory=dict)
    alt: list[str] = field(default_factory=list)     # wrong values for `text` facts (corrupted answers)
    entity_kind: str | None = None  # name facts: team | person | office | department


@dataclass
class Doc:
    id: str
    lang: str
    kind: str                       # product | team | policy | office | announcement
    title: str
    text: str
    subject: str                    # the entity the document is about (as it is named in queries)
    facts: dict[str, Fact] = field(default_factory=dict)
    links: list[str] = field(default_factory=list)
    authority: float = 0.7
    updated_at: int = 0
    risk: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        return f"{self.id}#0"       # every synthetic document is short enough to be one chunk

    def record(self) -> dict:
        return {"id": self.id, "title": self.title, "text": self.text, "updated_at": self.updated_at,
                "authority": self.authority, "links": self.links}


@dataclass
class World:
    docs: dict[str, Doc]
    products: list[Doc]
    teams: list[Doc]
    fact_docs: list[Doc]            # policies and offices
    names: dict[str, list[str]]     # entity kind -> names (for corrupted answers)
    leads: dict[str, str]           # team -> lead
    team_products: dict[str, list[str]]
    departments: dict[str, list[str]]  # department (English) -> teams
    terms: list[str]                # graph terms (kanji names have no capitals to find them by)


# ----------------------------------------------------------------------------- vocabulary

MONTHS = {"en": "{n} months", "ja": "{n}か月", "mn": "{n} сар"}
PRODUCT_NOUNS = {
    "price": {"en": "price", "ja": "価格", "mn": "үнэ"},
    "warranty": {"en": "warranty period", "ja": "保証期間", "mn": "баталгаат хугацаа"},
    "weight": {"en": "weight", "ja": "重さ", "mn": "жин"},
    "launch": {"en": "launch date", "ja": "発売日", "mn": "худалдаанд гарсан огноо"},
    "sku": {"en": "SKU", "ja": "SKU", "mn": "SKU код"},
    "owner": {"en": "owning team", "ja": "担当チーム", "mn": "хариуцдаг баг"},
}
PRODUCT_RISK = {"price": 2, "warranty": 2, "weight": 0, "launch": 0, "sku": 0, "owner": 1}
MISSING_NOUNS = {   # attributes no product sheet states: questions about them are unanswerable
    "battery": {"en": "battery life", "ja": "バッテリー駆動時間", "mn": "батерейн хугацаа"},
    "temperature": {"en": "maximum operating temperature", "ja": "最高動作温度", "mn": "ажиллах дээд температур"},
    "origin": {"en": "country of manufacture", "ja": "製造国", "mn": "үйлдвэрлэсэн улс"},
}
TEAM_NOUNS = {
    "department": {"en": "department", "ja": "所属部門", "mn": "харьяалах хэлтэс"},
    "office": {"en": "office", "ja": "拠点オフィス", "mn": "байрлах оффис"},
    "lead": {"en": "team lead", "ja": "チームリーダー", "mn": "багийн ахлагч"},
}
DEPARTMENTS = {
    "Engineering": {"ja": "エンジニアリング部門", "mn": "Инженерийн хэлтэс"},
    "Operations": {"ja": "オペレーション部門", "mn": "Үйл ажиллагааны хэлтэс"},
    "Sales": {"ja": "営業部門", "mn": "Борлуулалтын хэлтэс"},
    "Finance": {"ja": "財務部門", "mn": "Санхүүгийн хэлтэс"},
    "People": {"ja": "人事部門", "mn": "Хүний нөөцийн хэлтэс"},
    "Safety": {"ja": "安全部門", "mn": "Аюулгүй ажиллагааны хэлтэс"},
}
OFFICES = {"Northgate": {"ja": "Northgate", "mn": "Northgate"}, "Port Allin": {"ja": "Port Allin", "mn": "Port Allin"},
           "Osaka": {"ja": "大阪", "mn": "Осака"}, "Ulaanbaatar": {"ja": "ウランバートル", "mn": "Улаанбаатар"}}
# team, department, office, page language, what the team does (English), owns products
TEAMS = [
    ("Parcel Systems", "Engineering", "Northgate", "en", "designs the parcel lockers and docks", True),
    ("Fleet Telemetry", "Engineering", "Northgate", "en", "builds the trackers and sensors fitted to vehicles", True),
    ("Robotics Lab", "Engineering", "Osaka", "ja", "倉庫ロボットと搬送機器を開発しています", True),
    ("Cold Chain", "Operations", "Port Allin", "en", "runs temperature-controlled storage and transport", True),
    ("Last Mile", "Operations", "Ulaanbaatar", "mn", "хотын доторх хүргэлтийг зохион байгуулдаг", True),
    ("Customs Desk", "Operations", "Port Allin", "en", "clears cross-border shipments and their paperwork", True),
    ("Key Accounts", "Sales", "Northgate", "en", "looks after the largest customers", False),
    ("Partner Sales", "Sales", "Osaka", "ja", "販売代理店との取引を担当しています", False),
    ("Billing Ops", "Finance", "Northgate", "en", "issues invoices and chases payments", False),
    ("Payroll Services", "Finance", "Ulaanbaatar", "mn", "ажилчдын цалинг тооцож олгодог", False),
    ("Talent Partners", "People", "Northgate", "en", "hires and onboards new staff", False),
    ("Site Safety", "Safety", "Port Allin", "en", "audits warehouses and trains staff on safe work", False),
]
LEADS = {"en": ["Mara Okonkwo", "Daniel Reyes", "Priya Natarajan", "Tomas Lindqvist", "Grace Mwangi",
                "Owen Castellanos", "Lena Hartmann", "Samir Haddad", "Ingrid Solberg"],
         "ja": ["田中 美咲", "佐藤 健太"], "mn": ["Бат-Эрдэнэ Ганбаатар", "Сарантуяа Дорж"]}
STAFF = {"en": ["Hugo Pereira", "Amelia Brandt", "Kofi Asante"], "ja": ["鈴木 陽菜"], "mn": ["Тэмүүлэн Батболд"]}
PRODUCT_NAMES = ["Velobox 300", "Arcternix Hub", "Quillomar Scanner", "Tessavio Dock", "Brimwyck Tracker",
                 "Corvellan Printer", "Nyvessa Sensor", "Halvarne Cart", "Pellowyn Lift", "Sorrelith Gate",
                 "Ivaquor Beacon", "Marloveth Scale", "Fenvoria Locker", "Graylmont Router", "Zephlinor Tag",
                 "Ondrevik Cooler", "Truvallis Truck", "Kelvadyn Monitor"]
CATEGORIES = {"Parcel Systems": ("parcel locker", "宅配ロッカー", "илгээмжийн шүүгээ"),
              "Fleet Telemetry": ("vehicle tracker", "車両トラッカー", "тээврийн хэрэгслийн төхөөрөмж"),
              "Robotics Lab": ("warehouse robot", "倉庫ロボット", "агуулахын робот"),
              "Cold Chain": ("refrigerated container", "冷蔵コンテナ", "хөргөлттэй сав"),
              "Last Mile": ("delivery cart", "配送カート", "хүргэлтийн тэрэг"),
              "Customs Desk": ("document scanner", "書類スキャナー", "баримт уншигч")}
SYLLABLES = ["mor", "vax", "el", "quen", "ta", "ro", "bel", "vorn", "saf", "frel", "dax", "ilo", "nur", "pex", "ost",
             "ra", "vin", "cul", "ber", "tho", "lan", "zef", "ka", "mir"]
SUFFIXES = ["Hub", "Scanner", "Tracker", "Dock", "Locker", "Sensor", "Cart", "Gate", "Beacon"]


def fake_product_name(rng: random.Random, taken: set[str]) -> str:
    """A product name in the corpus's style that no document mentions (unanswerable questions)."""
    while True:
        word = "".join(rng.choice(SYLLABLES) for _ in range(rng.randint(2, 3))).capitalize()
        name = f"{word} {rng.choice(SUFFIXES)}" if rng.random() < 0.7 else f"{word} {rng.randint(2, 9)}00"
        if name not in taken and word.lower() not in " ".join(taken).lower():
            return name


def _date(ts: int) -> str:
    return datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%d")


def _money(v: int) -> str:
    return f"{v:,} USD"


# ----------------------------------------------------------------------------- documents

def _product_doc(rng: random.Random, name: str, team: tuple, idx: int, others: list[str]) -> Doc:
    tname, _, _, lang, _, _ = team
    cat_en, cat_ja, cat_mn = CATEGORIES[tname]
    sku = f"KV-{rng.randint(1000, 9899)}"
    price = rng.choice([180, 240, 320, 450, 640, 890, 1250, 1480, 2100, 2750, 3900])
    price += rng.choice([0, 0, 15, 30, 45])
    months = rng.choice([6, 12, 18, 24, 36])
    kg = round(rng.uniform(0.3, 48.0), 1)
    launch = _date(T0 - rng.randint(200, 1500) * DAY)
    facts = {
        "price": Fact("price", PRODUCT_NOUNS["price"], dict.fromkeys(LANGS, _money(price)), "number", 2),
        "warranty": Fact("warranty", PRODUCT_NOUNS["warranty"], {k: v.format(n=months) for k, v in MONTHS.items()},
                         "months", 2),
        "weight": Fact("weight", PRODUCT_NOUNS["weight"], {"en": f"{kg} kg", "ja": f"{kg} kg", "mn": f"{kg} кг"},
                       "number", 0),
        "launch": Fact("launch", PRODUCT_NOUNS["launch"], dict.fromkeys(LANGS, launch), "date", 0),
        "sku": Fact("sku", PRODUCT_NOUNS["sku"], dict.fromkeys(LANGS, sku), "code", 0),
        "owner": Fact("owner", PRODUCT_NOUNS["owner"], dict.fromkeys(LANGS, tname), "name", 1, entity_kind="team"),
    }
    extra = ""
    if others and rng.random() < 0.35:
        mate = rng.choice(others)
        extra = {"en": f" It is often installed together with the {mate}.",
                 "ja": f" {mate}と組み合わせて使われることが多い製品です。",
                 "mn": f" Ихэвчлэн {mate}-тэй хамт суурилуулдаг."}[lang]
    # two or three wordings per language, so no head can learn one sheet layout instead of the facts
    style = rng.randrange(3 if lang == "en" else 2)
    m = _money(price)
    if lang == "en":
        text = [f"The {name} (SKU {sku}) is a {cat_en}. It sells for {m} and comes with a {months}-month warranty. "
                f"The unit weighs {kg} kg and was launched on {launch}. The product is owned by the {tname} team.",
                f"{name} ({cat_en}, SKU {sku}): list price {m}, warranty {months} months, weight {kg} kg, launched "
                f"{launch}, owned by the {tname} team.",
                f"The {cat_en} {name} (SKU {sku}) has a price of {m} and a warranty of {months} months. It weighs "
                f"{kg} kg and launched on {launch}; the {tname} team owns it."][style] + extra
    elif lang == "ja":
        text = [f"{name}（SKU {sku}）は{cat_ja}です。価格は{m}で、保証期間は{months}か月です。"
                f"重さは{kg} kg、発売日は{launch}です。担当チームは{tname}です。",
                f"{name}（{cat_ja}、SKU {sku}）の価格は{m}、保証期間は{months}か月、重さは{kg} kgです。"
                f"{launch}に発売され、{tname}チームが担当しています。"][style] + extra
    else:
        text = [f"{name} (SKU {sku}) нь {cat_mn} юм. Үнэ нь {m}, баталгаат хугацаа нь {months} сар. "
                f"Жин нь {kg} кг бөгөөд {launch}-нд худалдаанд гарсан. Хариуцдаг баг нь {tname}.",
                f"{name} ({cat_mn}, SKU {sku}) нь {m} үнэтэй, {months} сарын баталгаатай, {kg} кг жинтэй. "
                f"{launch}-нд худалдаанд гарсан бөгөөд {tname} баг хариуцдаг."][style] + extra
    title = {"en": f"{name} product sheet", "ja": f"{name} 製品情報", "mn": f"{name} бүтээгдэхүүний мэдээлэл"}[lang]
    return Doc(id=f"products/{name.lower().replace(' ', '-')}", lang=lang, kind="product", title=title, text=text,
               subject=name, facts=facts, links=[f"teams/{tname.lower().replace(' ', '-')}"], authority=0.8,
               updated_at=T0 - rng.randint(20, 420) * DAY, meta={"team": tname, "price": price, "months": months,
                                                                  "kg": kg, "idx": idx})


def _team_doc(rng: random.Random, team: tuple, lead: str) -> Doc:
    tname, dept, office, lang, what, _ = team
    dept_v = {"en": dept, "ja": DEPARTMENTS[dept]["ja"], "mn": DEPARTMENTS[dept]["mn"]}
    office_v = {"en": office, "ja": OFFICES[office]["ja"], "mn": OFFICES[office]["mn"]}
    size = rng.randint(5, 40)
    facts = {
        "department": Fact("department", TEAM_NOUNS["department"], dict.fromkeys(LANGS, dept_v[lang]), "name", 1,
                           entity_kind="department"),
        "office": Fact("office", TEAM_NOUNS["office"], dict.fromkeys(LANGS, office_v[lang]), "name", 0,
                       entity_kind="office"),
        "lead": Fact("lead", TEAM_NOUNS["lead"], dict.fromkeys(LANGS, lead), "name", 1, entity_kind="person"),
    }
    if lang == "en":
        text = (f"The {tname} team is part of the {dept} department and works from the {office} office. "
                f"Its team lead is {lead}. The team {what} and has {size} members.")
    elif lang == "ja":
        text = (f"{tname}チームは{dept_v['ja']}に所属し、{office_v['ja']}オフィスを拠点としています。"
                f"チームリーダーは{lead}です。{what}。メンバーは{size}名です。")
    else:
        text = (f"{tname} баг нь {dept_v['mn']}т харьяалагддаг бөгөөд {office_v['mn']} оффист байрладаг. "
                f"Багийн ахлагч нь {lead}. Тус баг {what}. {size} гишүүнтэй.")
    title = {"en": f"{tname} team", "ja": f"{tname}チーム紹介", "mn": f"{tname} багийн танилцуулга"}[lang]
    return Doc(id=f"teams/{tname.lower().replace(' ', '-')}", lang=lang, kind="team", title=title, text=text,
               subject=tname, facts=facts, authority=0.7, updated_at=T0 - rng.randint(20, 300) * DAY, risk=1,
               meta={"department": dept, "office": office, "lead": lead, "size": size})


def _fact(key: str, noun: dict, value: str, kind: str, risk: int, questions: dict, alt: list[str] | None = None,
          entity_kind: str | None = None) -> Fact:
    return Fact(key, noun, dict.fromkeys(LANGS, value), kind, risk, questions, alt or [], entity_kind)


def _policy_docs(rng: random.Random) -> list[Doc]:
    """Policies and office pages: each states three or four facts with numbers / dates, filled into the text."""
    r = rng.randint
    specs = []
    # ---- refund (en, money)
    v = {"window": f"{rng.choice([14, 30, 45])} days", "credit": f"{rng.choice([40, 50, 60])}%",
         "processing": f"{r(5, 10)} business days", "effective": _date(T0 - r(250, 500) * DAY)}
    specs.append(("policies/refund", "en", 2, "Refund policy", "Refund policy",
                  f"Refund policy, effective {v['effective']}. Customers may return an item within {v['window']} of "
                  f"delivery for a full refund. Items returned later, up to 90 days after delivery, receive "
                  f"{v['credit']} store credit instead. Approved refunds are paid out within {v['processing']}.", [
                      ("window", "return window for a full refund", v["window"], "number",
                       ["How many days do customers have to return an item for a full refund?",
                        "Within what period can a customer return something and get all their money back?"]),
                      ("credit", "store credit share for late returns", v["credit"], "number",
                       ["How much store credit do late returns receive?",
                        "What share of the price comes back as store credit for a late return?"]),
                      ("processing", "refund processing time", v["processing"], "number",
                       ["How quickly are approved refunds paid out?",
                        "How long does it take to receive an approved refund?"]),
                      ("effective", "effective date", v["effective"], "date",
                       ["Since when is the current refund policy in effect?",
                        "What is the effective date of the refund rules?"])]))
    # ---- shipping (en, money)
    v = {"fee": f"{r(8, 15)} USD", "free": f"{rng.choice([250, 300, 400, 500])} USD", "express": f"{r(25, 45)} USD",
         "days": f"{r(2, 4)} business days"}
    specs.append(("policies/shipping", "en", 2, "Shipping policy", "Shipping policy",
                  f"Shipping policy. Standard delivery costs {v['fee']} per order and arrives within {v['days']}. "
                  f"Orders above {v['free']} ship free of charge. Express delivery costs {v['express']} and arrives "
                  f"the next business day.", [
                      ("fee", "standard shipping fee", v["fee"], "number",
                       ["How much does standard shipping cost?", "What is the fee for standard delivery of an order?"]),
                      ("free", "free shipping threshold", v["free"], "number",
                       ["Above what order value is shipping free?", "From which order amount do we ship for free?"]),
                      ("express", "express delivery fee", v["express"], "number",
                       ["What does express delivery cost?", "How much do customers pay for next-day express?"]),
                      ("days", "standard delivery time", v["days"], "number",
                       ["How long does standard delivery take?", "Within how many days does a standard order arrive?"])]))
    # ---- data retention (en, legal)
    v = {"years": f"{rng.choice([5, 7, 10])} years", "deletion": f"{rng.choice([14, 30])} days",
         "backups": f"{rng.choice([60, 90, 120])} days"}
    specs.append(("policies/data-retention", "en", 3, "Data retention policy", "Data retention policy",
                  f"Data retention policy, required by law. Customer records are retained for {v['years']} after the "
                  f"last order. Verified deletion requests are completed within {v['deletion']}. Backups containing "
                  f"personal data are purged after {v['backups']}. Breaches must be reported to the regulator.", [
                      ("years", "retention period for customer records", v["years"], "number",
                       ["How long are customer records retained?",
                        "For how many years must we keep a customer's records after the last order?"]),
                      ("deletion", "deadline for deletion requests", v["deletion"], "number",
                       ["How fast must a verified deletion request be completed?",
                        "What is the deadline for completing a data deletion request?"]),
                      ("backups", "purge period for backups with personal data", v["backups"], "number",
                       ["When are backups with personal data purged?",
                        "After how many days are backups that contain personal data deleted?"])]))
    # ---- hazardous goods (en, safety)
    v = {"ppe": "chemical-resistant gloves, safety goggles and a flame-resistant apron", "lift": f"{r(20, 25)} kg",
         "spill": f"{rng.choice([10, 15, 30])} minutes", "training": f"{rng.choice([12, 24])} months"}
    specs.append(("policies/hazardous-goods", "en", 3, "Hazardous goods handling", "hazardous goods procedure",
                  f"Hazardous goods handling procedure. Staff handling hazardous goods must wear {v['ppe']}. No person "
                  f"may lift more than {v['lift']} of hazardous material by hand. Any spill must be reported to Site "
                  f"Safety within {v['spill']}. Refresher training is mandatory every {v['training']}.", [
                      ("ppe", "required protective equipment", v["ppe"], "text",
                       ["What protective equipment is required when handling hazardous goods?",
                        "What must staff wear when they handle hazardous material?"],
                       ["a hard hat and ear defenders", "a high-visibility vest only"]),
                      ("lift", "manual lifting limit for hazardous material", v["lift"], "number",
                       ["What is the maximum weight of hazardous material one person may lift by hand?",
                        "How heavy can a manual lift of hazardous goods be?"]),
                      ("spill", "spill reporting deadline", v["spill"], "number",
                       ["How quickly must a hazardous spill be reported?",
                        "Within how many minutes do we report a chemical spill?"]),
                      ("training", "refresher training interval", v["training"], "number",
                       ["How often is hazardous goods training repeated?",
                        "What is the refresher interval for hazardous goods training?"])]))
    # ---- workplace injuries (en, medical)
    v = {"report": f"{rng.choice([12, 24, 48])} hours", "kits": f"every {rng.choice([1, 3])} months",
         "hospital": "St. Aldric Clinic", "aiders": f"{r(2, 4)} trained first aiders"}
    specs.append(("policies/injury", "en", 3, "Workplace injury and first aid", "workplace injury procedure",
                  f"Workplace injury and first aid. Every injury, however small, must be reported within "
                  f"{v['report']} using the incident form. First aid kits are checked {v['kits']}. Each warehouse "
                  f"shift has at least {v['aiders']}. Serious injuries go to the {v['hospital']}.", [
                      ("report", "injury reporting deadline", v["report"], "number",
                       ["How soon must a workplace injury be reported?",
                        "What is the deadline for filing an incident form after an injury?"]),
                      ("kits", "first aid kit check interval", v["kits"], "number",
                       ["How often are first aid kits checked?", "At what interval do we inspect the first aid kits?"]),
                      ("hospital", "hospital for serious injuries", v["hospital"], "text",
                       ["Where are seriously injured workers taken?",
                        "Which hospital handles serious workplace injuries?"],
                       ["the Northgate General Hospital", "the Harbour Medical Centre"]),
                      ("aiders", "first aiders per shift", v["aiders"], "number",
                       ["How many first aiders must be on each shift?",
                        "What is the minimum number of trained first aiders per warehouse shift?"])]))
    # ---- information security (en, internal)
    v = {"rotation": f"{rng.choice([60, 90, 180])} days", "mfa": "all remote logins",
         "lock": f"{rng.choice([5, 10, 15])} minutes"}
    specs.append(("policies/security", "en", 1, "Information security policy", "information security policy",
                  f"Information security policy. Passwords are rotated every {v['rotation']}. Multi-factor "
                  f"authentication is required for {v['mfa']}. Laptops lock automatically after {v['lock']} of "
                  f"inactivity and their disks are encrypted.", [
                      ("rotation", "password rotation interval", v["rotation"], "number",
                       ["How often do passwords have to be changed?", "What is the password rotation interval?"]),
                      ("mfa", "logins that require multi-factor authentication", v["mfa"], "text",
                       ["When is multi-factor authentication required?", "Which logins need MFA?"],
                       ["office logins only", "administrator accounts only"]),
                      ("lock", "automatic screen lock time", v["lock"], "number",
                       ["After how long do laptops lock automatically?",
                        "How many idle minutes before a laptop screen locks?"])]))
    # ---- travel expenses (ja, internal)
    v = {"per_diem": f"{rng.choice([4000, 5000, 6000]):,}円", "hotel": f"{rng.choice([12000, 15000]):,}円",
         "approval": f"{rng.choice([5, 7, 10])}日前"}
    specs.append(("policies/travel-ja", "ja", 1, "出張旅費規程", "出張旅費規程",
                  f"出張旅費規程。国内出張の日当は{v['per_diem']}です。宿泊費の上限は1泊{v['hotel']}です。"
                  f"出張は出発の{v['approval']}までに上長の承認が必要です。", [
                      ("per_diem", "国内出張の日当", v["per_diem"], "number",
                       ["国内出張の日当はいくらですか？", "出張したときの1日の手当は？"]),
                      ("hotel", "宿泊費の上限", v["hotel"], "number",
                       ["出張の宿泊費の上限はいくらですか？", "ホテル代は1泊いくらまで認められますか？"]),
                      ("approval", "出張承認の期限", v["approval"], "number",
                       ["出張の承認はいつまでに必要ですか？", "上長の承認は出発の何日前までに取りますか？"])]))
    # ---- leave (mn, internal)
    v = {"annual": f"{r(15, 22)} хоног", "sick": f"{r(7, 12)} хоног", "carry": f"{r(3, 6)} хоног"}
    specs.append(("policies/leave-mn", "mn", 1, "Ээлжийн амралтын журам", "ээлжийн амралтын журам",
                  f"Ээлжийн амралтын журам. Ажилтан жилд {v['annual']} ээлжийн амралт эдэлнэ. Өвчний чөлөө жилд "
                  f"{v['sick']} байна. Ашиглаагүй амралтаас дараа жил рүү {v['carry']} шилжүүлж болно.", [
                      ("annual", "жилийн ээлжийн амралт", v["annual"], "number",
                       ["Жилд хэдэн хоног ээлжийн амралт авах вэ?", "Ээлжийн амралт хэдэн хоног байдаг вэ?"]),
                      ("sick", "өвчний чөлөө", v["sick"], "number",
                       ["Өвчний чөлөө жилд хэдэн хоног вэ?", "Өвдсөн үед хэдэн хоног чөлөө авч болох вэ?"]),
                      ("carry", "дараа жил рүү шилжүүлэх амралт", v["carry"], "number",
                       ["Ашиглаагүй амралтаас хэдэн хоногийг дараа жил шилжүүлж болох вэ?",
                        "Амралтын хэдэн хоногийг дараа жил рүү шилжүүлж болдог вэ?"])]))
    # ---- offices
    for oid, lang, office, street in (("offices/northgate", "en", "Northgate", "14 Larkspur Row"),
                                      ("offices/port-allin", "en", "Port Allin", "2 Quayside Terrace")):
        v = {"address": street, "opened": str(r(2008, 2020)), "staff": f"{r(60, 400)} employees"}
        specs.append((oid, lang, 0, f"{office} office", f"{office} office",
                      f"The {office} office is at {v['address']}. It opened in {v['opened']} and today hosts "
                      f"{v['staff']}. Visitors sign in at the front desk.", [
                          ("address", "address", v["address"], "text",
                           [f"What is the address of the {office} office?", f"Where exactly is the {office} office?"],
                           ["7 Mill Lane", "88 Harbour Road"]),
                          ("opened", "opening year", v["opened"], "number",
                           [f"When did the {office} office open?", f"In which year was the {office} office opened?"]),
                          ("staff", "number of employees", v["staff"], "number",
                           [f"How many people work at the {office} office?",
                            f"What is the headcount of the {office} office?"])]))
    v = {"staff": f"{r(30, 120)}名", "opened": f"{r(2012, 2021)}年"}
    specs.append(("offices/osaka-ja", "ja", 0, "大阪オフィス", "大阪オフィス",
                  f"大阪オフィスは{v['opened']}に開設され、現在{v['staff']}の社員が勤務しています。受付は1階です。", [
                      ("staff", "社員数", v["staff"], "number",
                       ["大阪オフィスには何人の社員がいますか？", "大阪オフィスの社員数を教えてください。"]),
                      ("opened", "開設年", v["opened"], "number",
                       ["大阪オフィスはいつ開設されましたか？", "大阪オフィスの開設年は？"])]))
    v = {"staff": f"{r(30, 120)} ажилтан", "opened": f"{r(2012, 2021)} он"}
    specs.append(("offices/ulaanbaatar-mn", "mn", 0, "Улаанбаатар оффис", "Улаанбаатар оффис",
                  f"Улаанбаатар оффис {v['opened']}д нээгдсэн бөгөөд одоо {v['staff']} ажилладаг.", [
                      ("staff", "ажилчдын тоо", v["staff"], "number",
                       ["Улаанбаатар оффист хэдэн ажилтан ажилладаг вэ?", "Улаанбаатар оффисын ажилчдын тоо хэд вэ?"]),
                      ("opened", "нээгдсэн он", v["opened"], "number",
                       ["Улаанбаатар оффис хэдэн онд нээгдсэн бэ?", "Улаанбаатар оффис хэзээ нээгдсэн бэ?"])]))
    docs = []
    for did, lang, risk, title, subject, text, facts in specs:
        fd = {}
        for f in facts:
            key, noun, value, kind, questions = f[:5]
            alt = f[5] if len(f) > 5 else None
            fd[key] = _fact(key, {lang: noun}, value, kind, risk, {lang: questions}, alt)
        docs.append(Doc(id=did, lang=lang, kind="policy" if did.startswith("policies/") else "office", title=title,
                        text=text, subject=subject, facts=fd, risk=risk,
                        authority=1.0 if did.startswith("policies/") else 0.7,
                        updated_at=T0 - rng.randint(30, 500) * DAY))
    return docs


def _announcements(rng: random.Random, products: list[str]) -> list[Doc]:
    """News items that name products (and numbers) without stating any product fact: hard negatives."""
    out = []
    events = ["spring town hall", "autumn customer day", "logistics trade fair", "warehouse open day",
              "partner summit", "innovation week", "year-end review", "safety week"]
    for ev in events:
        names = rng.sample(products, 3)
        people = rng.randint(80, 600)
        out.append(Doc(id=f"news/{ev.replace(' ', '-')}", lang="en", kind="announcement", title=f"News: {ev}",
                       text=f"About {people} guests attended the {ev} of {COMPANY}. Visitors tried the {names[0]} and "
                            f"the {names[1]}, and a preview of the next {names[2]} drew long queues. Photos are on "
                            f"the intranet.",
                       subject=ev, authority=0.4, updated_at=T0 - rng.randint(5, 360) * DAY))
    return out


def build_world(seed: int = 7) -> World:
    rng = random.Random(f"kestrovia:{seed}")
    leads: dict[str, str] = {}
    pools = {k: list(v) for k, v in LEADS.items()}
    for k in pools:
        rng.shuffle(pools[k])
    teams = []
    for t in TEAMS:
        lead = pools[t[3]].pop()
        leads[t[0]] = lead
        teams.append(_team_doc(rng, t, lead))
    names = list(PRODUCT_NAMES)
    rng.shuffle(names)
    owners = [t for t in TEAMS if t[5]]
    products, team_products = [], {t[0]: [] for t in TEAMS}
    for i, name in enumerate(names):
        team = owners[i % len(owners)]
        others = [n for n in names if n != name]
        d = _product_doc(rng, name, team, i, others)
        products.append(d)
        team_products[team[0]].append(name)
    fact_docs = _policy_docs(rng)
    news = _announcements(rng, names)
    docs = {d.id: d for d in [*products, *teams, *fact_docs, *news]}
    departments: dict[str, list[str]] = {}
    for t in TEAMS:
        departments.setdefault(t[1], []).append(t[0])
    people = [*leads.values(), *(n for v in STAFF.values() for n in v)]
    entity_names = {"team": [t[0] for t in TEAMS], "person": people, "office": list(OFFICES),
                    "department": list(DEPARTMENTS), "product": names}
    terms = [n for n in people if any("一" <= ch <= "鿿" for ch in n)]
    return World(docs=docs, products=products, teams=teams, fact_docs=fact_docs, names=entity_names, leads=leads,
                 team_products=team_products, departments=departments, terms=terms)
