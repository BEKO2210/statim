"""Golden reference outputs from the official Laya Python package (CPU, fp32).

Writes tests/data/golden_<model>.jsonl: per (state, question) item the token ids, marker
positions, raw logits, act probabilities, plus the public answer dict from system_one().
Statim's C++ engine must reproduce these.
"""
import argparse, json, os, sys
import numpy as np
import torch

import laya
from laya.common import collate_items

STATES = [
    "I was charged twice for March, please refund the duplicate today or we cancel.",
    {"from": "user@acme.com", "subject": "Duplicate charge on invoice #4411",
     "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."},
    "Mein Konto wurde zweimal belastet. Bitte erstatten Sie den Betrag sofort, sonst kündige ich.",
    "La aplicación se cierra cada vez que intento subir una foto. Es muy frustrante.",
    "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।",
    "我们的服务器从昨晚开始宕机，所有客户都无法登录，请立即处理！",
    "サーバーが落ちています。至急対応をお願いします。",
    "Спасибо за быструю помощь, всё работает отлично!",
    "هل يمكنكم إرسال عرض سعر لخمسين ترخيصاً؟",
    "Can you send me a quote for 50 seats of the enterprise plan?",
    "The build fails with a segmentation fault in libfoo.so after upgrading to v2.3.",
    [{"role": "user", "content": "Hi, my package never arrived."},
     {"role": "assistant", "content": "Sorry to hear that! Can you share the order number?"},
     {"role": "user", "content": "It's #99812. I want my money back, this is the third time."}],
    "Great product, would buy again. Delivery was fast.",
    "Worst experience ever. The support agent hung up on me twice.",
    "Please delete my account and all my personal data under GDPR Article 17.",
    "Merci beaucoup pour votre aide, le problème est résolu.",
    "Il pagamento non va a buon fine con la carta Visa, errore 402.",
    "O sistema está lento desde a atualização de ontem.",
    "Toplantıyı yarın saat 10'a erteleyebilir miyiz?",
    "Czy mogę zmienić adres dostawy dla zamówienia 5521?",
    "Refactor this service to use dependency injection and add unit tests.",
    {"ticket": 7781, "priority": None, "tags": ["login", "sso"], "text": "SSO login loops forever after the Okta change.", "nested": {"a": 1.5, "b": True}},
    "Our CEO said the Q3 numbers look great 🚀🔥 and we're hiring 5 engineers!",
    "Wir brauchen bis Freitag ein Angebot für 200 Lizenzen, sonst gehen wir zur Konkurrenz.",
    "",
    "ok",
    "URGENT!!! Production database is down, all customers affected, revenue loss every minute.",
    "Could you tell me what your office hours are?",
    "The invoice total is wrong: 1.299,00 € instead of 1.199,00 €.",
    "x " * 700,  # forces truncation at max_len
]

QUESTIONS = {
    "department": {"type": "choice", "instructions": "Which department should handle this request?",
                   "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages, system errors",
                                "sales": "pricing, new contracts", "other": "everything else"}},
    "urgency": {"type": "score", "instructions": "How urgent is this request?",
                "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]},
    "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"},
    "sentiment": {"type": "choice", "instructions": "What is the sentiment of the message?",
                  "criteria": ["positive", "neutral", "negative"]},
    "refund": {"type": "noul", "instructions": "Does the user explicitly request a refund?",
               "criteria": {"true": "the user asks for money back", "false": "no refund request"}},
    "lang_de": {"type": "noul", "instructions": "Ist die Nachricht auf Deutsch geschrieben?",
                "labels": {"false": "nein", "true": "ja"}},
    "severity": {"type": "score", "instructions": "Rate the severity from 0 to 4.",
                 "criteria": ["none", "low", "medium", "high", "critical"]},
    "intent20": {"type": "choice", "instructions": "What is the user's intent?",
                 "criteria": ["refund", "cancel", "complaint", "praise", "quote", "bug_report", "outage",
                              "account_deletion", "shipping", "payment_failure", "scheduling", "address_change",
                              "feature_request", "question", "hiring", "invoice_error", "login_issue",
                              "performance", "thanks", "other"]},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="laya-multilingual")
    ap.add_argument("--hidden", type=int, default=2, help="dump encoder hidden states for first N items")
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count())
    root = os.path.join(os.path.dirname(__file__), "..")
    agent = laya.load(os.path.abspath(os.path.join(root, "models", a.model)), device="cpu")
    assert agent.dtype == torch.float32 and not agent.amp_enabled
    ids = list(QUESTIONS)
    internal = {q: agent._to_internal(QUESTIONS[q]) for q in ids}
    out_path = os.path.join(root, "tests", "data", "golden_%s.jsonl" % a.model)
    n = 0
    with open(out_path, "w") as f:
        for si, state in enumerate(STATES):
            items = agent._encode_state(state, ids, internal)
            answers = agent.system_one(state, QUESTIONS)["answers"]
            for j, qid in enumerate(ids):
                it = items[j]
                b = collate_items([[it]], agent.tok.pad_token_id)
                with torch.no_grad():
                    logits, act = agent.model(b["input_ids"], b["attention_mask"], b["marker_pos"],
                                              b["marker_mask"], b["qtype"])
                    rec = {"state_index": si, "question": qid, "ids": it["ids"], "markers": it["markers"],
                           "qtype": it["qtype"], "logits": logits[0, :len(it["markers"])].tolist(),
                           "act": torch.softmax(act.float(), -1)[0].tolist(), "answer": answers[qid]}
                    if n < a.hidden:
                        h = agent.model.encoder(input_ids=b["input_ids"],
                                                attention_mask=b["attention_mask"]).last_hidden_state
                        rec["encoder_last_hidden"] = h[0].tolist()
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    json.dump({"states": STATES, "questions": QUESTIONS}, open(os.path.join(root, "tests", "data", "golden_inputs.json"), "w"),
              ensure_ascii=False, indent=1)
    print("wrote", n, "items to", out_path)


if __name__ == "__main__":
    main()
