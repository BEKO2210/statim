import json, collections, sys
sys.path.insert(0, "tools/finetune")
import check_sources
D = "tools/finetune/sources/"
p1 = json.load(open(D + "evidence-2026-10-03.json")); p2 = json.load(open(D + "evidence-2026-10-03-pass2.json"))
final = {s: {**p1[s], **p2.get(s, {})} for s in ("v6", "v5")}
VERIFY = "independent verification 2026-10-03: "
# independent verification (Claude agents) overrides
fail = {
 ("v6", "maximoss/mnli-nineeleven-fr::default"): "FAIL: rule 1 — " + VERIFY + "the bsd-2-clause tag has no upstream basis; MultiNLI text is under the OANC licence, not allowlisted",
 ("v6", "gretelai/gretel-pii-masking-en-v1::default"): "FAIL: rule 3 — " + VERIFY + "'mistral-nemo-2407' through Gretel Navigator, a compound system; no exact model id",
 ("v6", "alaminxpro/university-students-complaints::default"): "FAIL: rule 2/3 — " + VERIFY + "who wrote the paraphrased variants is undocumented, no grant from the students",
 ("v6", "ai4bharat/IndicSentiment::default"): "UNVERIFIED: licence_spdx — " + VERIFY + "no licence on the HF card; IndicBERT README only says datasets 'will be released under a CC-0 license'",
 ("v5", "FOL-nli"): "FAIL: rule 3 — " + VERIFY + "arXiv 2406.11035: 'We prompted GPT-4 (May version) to Generate 150 predicates'",
 ("v5", "cladder"): "FAIL: rule 3 — " + VERIFY + "arXiv 2312.04350 A.6: 'we use GPT-4 to generate some meaningless words'",
 ("v5", "PARARULE-Plus"): "FAIL: rule 1 — " + VERIFY + "License_PARARULE_Plus.txt is MIT plus a credit clause (GitHub: NOASSERTION), not allowlisted",
 ("v5", "social_i_qa"): "UNVERIFIED: licence_spdx — " + VERIFY + "tasksource card says license: unknown; allenai card has no licence field",
}
fixes = {
 "babi_nli/": ("CC-BY-3.0", ["https://huggingface.co/datasets/facebook/babi_qa/blob/021d7aeb7307/README.md"]),
 "SpaceNLI": ("MIT", ["https://github.com/kovvalsky/SpaceNLI/blob/8c10e94d2387/LICENSE"]),
 "temporal-nli": ("CC0-1.0", ["https://github.com/kunalkukreja21/temporal-expressions-evaluation-lm/blob/384ebf53ed5b/LICENSE"]),
 "clcd-english": ("MIT", ["https://github.com/felipessalvatore/CLCD/blob/422f9e93d49e/LICENSE"]),
}
ALIAS = {"Gemma-4-26B-A4B-it": "google/gemma-4-26B-A4B-it", "phi3.5-mini-instruct-q8_0": "microsoft/Phi-3.5-mini-instruct",
         "Qwen3.8-27B": "Qwen/Qwen3.8-27B"}
NEW_GENS = [("google/gemma-4-26B-A4B-it", "Apache-2.0", "https://huggingface.co/google/gemma-4-26B-A4B-it"),
            ("microsoft/Phi-3.5-mini-instruct", "MIT", "https://huggingface.co/microsoft/Phi-3.5-mini-instruct"),
            ("mistralai/Mistral-7B-v0.1", "Apache-2.0", "https://huggingface.co/mistralai/Mistral-7B-v0.1"),
            ("mistralai/Mistral-Nemo-Instruct-2407", "Apache-2.0", "https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407"),
            ("Helsinki-NLP/opus-mt-tc-big-en-fr", "CC-BY-4.0", "https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-en-fr")]
pol = json.load(open(D + "policy.json"))
pol["generators"] = [g for g in pol["generators"] if g["model"] != "Gemma-4-26B-A4B-it"]
have = {g["model"] for g in pol["generators"]}
for m, l, u in NEW_GENS:
    if m not in have:
        pol["generators"].append({"model": m, "licence_spdx": l, "terms_url": u, "allows_training": True})
def norm_gen(g):
    if isinstance(g, list):
        out = [dict(x, model=ALIAS.get(x["model"], x["model"])) for x in g if x.get("model") not in ("human", "template")]
        return out or "human"
    return g
def cls(v):
    if v.startswith("PASS"): return "pass"
    if v.startswith("UNVERIFIED"): return "unverified"
    return "model-output" if "rule 3" in v[:20] else "breaks-policy"
FIELDS = ("licence_spdx", "licence_evidence", "provenance", "text_licence_spdx", "generator")
stats = collections.Counter(); excl = []
def apply(sec, key, rec, target):
    v = fail.get((sec, key), rec.get("verdict", "UNVERIFIED: no record"))
    if sec == "v5" and key in fixes:
        rec = dict(rec); rec["licence_spdx"], rec["licence_evidence"] = fixes[key]
        if rec.get("text_licence_spdx") in (None, "TODO"): rec["text_licence_spdx"] = fixes[key][0]
    c = cls(v)
    if c == "pass":
        new = {f: rec[f] for f in FIELDS}; new["generator"] = norm_gen(new["generator"])
        err = check_sources.generator_error(new["generator"], pol)
        if err:
            c, v = "model-output", "FAIL: rule 3 — " + err
    if c == "pass":
        target.update(new); target["checked"] = "2026-10-03"; target["use"] = True
        target.pop("excluded_reason", None); stats[sec, "pass"] += 1
    else:
        target["use"] = False; target["excluded_reason"] = "licence audit 2026-10-03, see policy.json"
        stats[sec, c] += 1
        excl.append((sec, key, c, v))
    return c
v6 = json.load(open(D + "v6-keep.json"))
passed_ids = collections.defaultdict(list)
for e in v6:
    if not e.get("use"): continue
    key = "%s::%s" % (e["id"], e.get("config", "default"))
    rec = final["v6"].get(key)
    if rec is None: print("MISSING v6", key); continue
    c = apply("v6", key, rec, e)
    if c == "pass": passed_ids[e["id"]].append(e.get("config", "default"))
audit = json.load(open("tools/finetune/licence_audit.json"))
for fam, rec in final["v5"].items():
    t = audit["keep_families"].get(fam)
    if t is None: print("MISSING v5", fam); continue
    apply("v5", fam, rec, t)
# commonsense_qa_2.0 slipped through the commonsense_qa prefix without evidence
excl.append(("v5", "commonsense_qa_2.0", "unverified", "UNVERIFIED: " + VERIFY + "admitted only through the commonsense_qa prefix; CSQA 2 has no evidence record"))
# temporal-nli is now verified at the source (CC0-1.0): lift its earlier 'unverified' exclusion
pol["exclusions"] = [x for x in pol["exclusions"] if not (x["registry"] == "v5" and x["id"] == "temporal-nli")]
known = {(x["registry"], x["id"], x.get("config")) for x in pol["exclusions"]}
for sec, key, c, v in excl:
    if sec == "v6":
        sid, config = key.split("::", 1)
        x = {"registry": "v6", "id": sid, "scope": "source", "reason_class": c, "reason": v, "audit": "2026-10-03"}
        if passed_ids.get(sid):
            x["config"] = config; x["configs_only"] = True
        k = ("v6", sid, x.get("config"))
    else:
        x = {"registry": "v5", "id": key, "scope": "source", "reason_class": c, "reason": v, "audit": "2026-10-03"}
        k = ("v5", key, None)
    if k not in known:
        pol["exclusions"].append(x); known.add(k)
json.dump(pol, open(D + "policy.json", "w"), indent=2, ensure_ascii=False); open(D + "policy.json", "a").write("\n")
json.dump(v6, open(D + "v6-keep.json", "w"), indent=1, ensure_ascii=False); open(D + "v6-keep.json", "a").write("\n")
json.dump(audit, open("tools/finetune/licence_audit.json", "w"), indent=2, ensure_ascii=False); open("tools/finetune/licence_audit.json", "a").write("\n")
print(dict(stats)); print("exclusions now", len(pol["exclusions"]))
