"""Synthetic support tickets with known answers, in English, Japanese and Mongolian.

Each ticket is drawn from one queue's phrases; about one in ten says nothing specific (its queue cannot be read
from the text), and the `urgent` label follows the urgency wording with some noise. So a trained head should
find skill, and should not be certain. The labels are synthetic: they show that the pack works, not how it
would do on your tickets.
"""
from __future__ import annotations

import json
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

QUEUES = ("billing", "technical", "account", "shipping", "other")

EN = {
    "billing": (["Charged twice", "Wrong invoice amount", "Refund request", "Payment failed", "Question about my bill"],
                ["I was charged twice for my subscription this month.", "The invoice total does not match my plan.",
                 "Please refund the last payment to my card.", "My credit card was declined but the money left my bank.",
                 "Can you send a corrected invoice with our tax number?", "I cancelled and I am still being billed."]),
    "technical": (["App crashes on start", "Error when saving", "Very slow dashboard", "Export does not work",
                   "Bug in the mobile app"],
                  ["The app crashes as soon as I open it.", "I get an error message every time I save a report.",
                   "The dashboard takes a minute to load since the update.", "The export button does nothing.",
                   "Sync stopped working between my phone and laptop.", "The page shows a blank screen after login."]),
    "account": (["Cannot log in", "Password reset", "Change my email", "Locked out of my account", "Add a team member"],
                ["I forgot my password and the reset link never arrives.", "I cannot log in to my account anymore.",
                 "Please change the email address on my profile.", "My account is locked after too many attempts.",
                 "How do I add a colleague to our workspace?", "Two-factor codes are not accepted at sign-in."]),
    "shipping": (["Order not delivered", "Where is my package", "Damaged parcel", "Wrong delivery address",
                  "Late delivery"],
                 ["My order has not arrived and tracking has not moved.", "The parcel arrived damaged and the box was open.",
                  "I need to change the delivery address of my order.", "The courier says delivered but nothing came.",
                  "When will my package ship?", "I received the wrong item in my delivery."]),
    "other": (["Opening hours", "Partnership proposal", "Thank you", "General question", "Feedback"],
              ["What are your opening hours during the holidays?", "We would like to propose a partnership.",
               "Thanks to your team for the great service.", "Do you have an office in Europe?",
               "Where can I read your privacy policy?", "I would like to give some feedback on the website."]),
}
JA = {
    "billing": ["請求が二重に引き落とされています。", "請求書の金額が間違っています。", "返金をお願いします。"],
    "technical": ["アプリが起動しません。", "保存するとエラーが表示されます。", "システムの動作が非常に遅いです。"],
    "account": ["パスワードを忘れました。", "アカウントにログインできません。", "メールアドレスを変更したいです。"],
    "shipping": ["注文した商品が届きません。", "配送はいつ届きますか。", "荷物が破損していました。"],
    "other": ["営業時間を教えてください。", "提携のご提案があります。", "素晴らしい対応をありがとうございました。"],
}
MN = {
    "billing": ["Төлбөр хоёр удаа хасагдсан байна.", "Нэхэмжлэхийн дүн буруу байна.", "Мөнгөө буцааж авмаар байна."],
    "technical": ["Апп нээгдэхгүй байна.", "Хадгалах үед алдаа гараад байна.", "Систем маш удаан ажиллаж байна."],
    "account": ["Нууц үгээ мартсан.", "Бүртгэлдээ нэвтэрч чадахгүй байна.", "Имэйл хаягаа солих хүсэлтэй байна."],
    "shipping": ["Захиалга маань ирээгүй байна.", "Хүргэлт хэзээ ирэх вэ?", "Илгээмж гэмтсэн ирлээ."],
    "other": ["Ажлын цагийн хуваарийг мэдмээр байна.", "Хамтран ажиллах санал байна.", "Сайн үйлчилгээнд баярлалаа."],
}
VAGUE = {"en": (["Question", "Help needed", "Hello"], ["I have a question about your service.", "Could someone contact me?",
                                                        "I need some help, please."]),
         "ja": (["お問い合わせ"], ["サービスについて質問があります。", "ご連絡をお願いします。"]),
         "mn": (["Асуулт"], ["Үйлчилгээний талаар асуух зүйл байна.", "Надтай холбогдоно уу."])}
URGENT = {"en": ["This is urgent, we are blocked.", "Please answer today, our customers are waiting.",
                 "We are losing money every hour, please help immediately."],
          "ja": ["至急対応をお願いします。", "緊急です。本日中に解決してください。"],
          "mn": ["Маш яаралтай, өнөөдөр шийдэж өгнө үү.", "Яаралтай байна, бид ажиллаж чадахгүй байна."]}
CALM = {"en": ["No rush.", "Whenever you have time.", "Thank you."], "ja": ["お手すきの際にお願いします。"],
        "mn": ["Яарах зүйлгүй."]}


def tickets(n: int = 3000, seed: int = 7) -> list[dict]:
    """`n` labelled tickets {id, subject, body, lang, created_at, queue, urgent, source}, oldest first."""
    rng = random.Random(seed)
    start = datetime(2026, 1, 5, 8, 0, tzinfo=UTC)
    rows = []
    for i in range(n):
        lang = rng.choices(["en", "ja", "mn"], weights=[70, 15, 15])[0]
        queue = rng.choice(QUEUES)
        vague = rng.random() < 0.10
        if vague:
            subjects, bodies = VAGUE[lang]
            subject, body = rng.choice(subjects), [rng.choice(bodies)]
        elif lang == "en":
            subject, body = rng.choice(EN[queue][0]), rng.sample(EN[queue][1], rng.choice([1, 2, 2]))
        else:
            phrases = (JA if lang == "ja" else MN)[queue]
            subject, body = rng.choice(phrases).rstrip("。.?"), rng.sample(phrases, rng.choice([1, 2]))
        pressed = rng.random() < 0.25
        if pressed:
            body.append(rng.choice(URGENT[lang]))
        elif rng.random() < 0.4:
            body.append(rng.choice(CALM[lang]))
        urgent = rng.random() < (0.9 if pressed else 0.04)
        if urgent and pressed and lang == "en" and rng.random() < 0.5:
            subject = "URGENT: " + subject
        rows.append({"id": f"T{i:05d}", "subject": subject, "body": " ".join(body), "lang": lang,
                     "created_at": (start + timedelta(minutes=47 * i)).isoformat(), "queue": queue,
                     "urgent": urgent, "source": "synthetic"})
    return rows


def write(path: str | Path, n: int = 3000, seed: int = 7) -> dict:
    rows = tickets(n, seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    def count(key: str) -> dict:
        return {v: sum(1 for r in rows if r[key] == v) for v in sorted({r[key] for r in rows}, key=str)}

    return {"path": str(path), "rows": len(rows), "queues": count("queue"), "urgent": count("urgent"),
            "languages": count("lang")}
