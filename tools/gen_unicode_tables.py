#!/usr/bin/env python3
"""Generates src/unicode_tables.inc by probing HuggingFace `tokenizers` itself.

HF does not use current Unicode data: its NFC comes from the
`unicode-normalization-alignments` crate (old Unicode tables) and the GPT-2
regex classes (\\s, \\p{L}, \\p{N}) from its bundled Oniguruma. To be
bit-exact we derive every table from HF's observable behavior:

  * regex class of every code point (Letter / Number / Space / Other)
  * canonical combining class (only where HF knows it)
  * full canonical decomposition (HF NFD of each code point)
  * primary composition pairs (a, b) -> c that HF NFC actually forms
  * "NFC-relevant" code points (quick-check filter)

The tables are verified against HF NFC with a Python model of the C++
algorithm before being written.

Run: .venv-ref/bin/python tools/gen_unicode_tables.py
"""
import os
import random
import sys
import unicodedata

from tokenizers import normalizers, pre_tokenizers

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "src", "unicode_tables.inc")
MAXCP = 0x110000
SBASE, LBASE, VBASE, TBASE = 0xAC00, 0x1100, 0x1161, 0x11A7
LCOUNT, VCOUNT, TCOUNT = 19, 21, 28
NCOUNT = VCOUNT * TCOUNT
SCOUNT = LCOUNT * NCOUNT

LETTER, NUMBER, SPACE, OTHER = 0, 1, 2, 3

hf_nfd = normalizers.NFD()
hf_nfc = normalizers.NFC()
bl = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)


def cps():
    for cp in range(MAXCP):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        yield cp


def probe_class(c):
    if len(bl.pre_tokenize_str("a" + c)) == 1:
        return LETTER
    if len(bl.pre_tokenize_str("1" + c)) == 1:
        return NUMBER
    if len(bl.pre_tokenize_str("!" + c)) == 1:
        return OTHER
    return SPACE


def hf_ccc_nonzero(c):
    # Reordering probes with U+0334 (ccc 1) and U+0345 (ccc 240).
    t1 = "x" + c + "̴"
    t2 = "xͅ" + c
    return hf_nfd.normalize_str(t1) != t1 or hf_nfd.normalize_str(t2) != t2


def main():
    print("probing regex classes ...", flush=True)
    cls = [OTHER] * MAXCP
    for cp in cps():
        cls[cp] = probe_class(chr(cp))
    for cp in range(128):
        c = chr(cp)
        want = (LETTER if c.isascii() and c.isalpha() else NUMBER if c.isdigit()
                else SPACE if cp in (9, 10, 11, 12, 13, 32) else OTHER)
        assert cls[cp] == want, (hex(cp), cls[cp], want)
    diff_py = sum(1 for cp in cps() if cp >= 128 and cls[cp] != (
        LETTER if unicodedata.category(chr(cp))[0] == "L" else
        NUMBER if unicodedata.category(chr(cp))[0] == "N" else
        SPACE if chr(cp).isspace() and unicodedata.category(chr(cp)) in ("Zs", "Zl", "Zp", "Cc") else OTHER))
    print(f"  classes differing from Python unicodedata {unicodedata.unidata_version}: {diff_py}")

    print("probing decompositions ...", flush=True)
    decomp = {}
    for cp in cps():
        if SBASE <= cp < SBASE + SCOUNT:
            continue
        c = chr(cp)
        d = hf_nfd.normalize_str(c)
        if d != c:
            decomp[cp] = [ord(x) for x in d]

    print("probing combining classes ...", flush=True)
    ccc = {}
    for cp in cps():
        c = chr(cp)
        if cp in decomp:
            continue
        pc = unicodedata.combining(c)
        if pc == 0:
            # Code points newer than Python's Unicode data are unknown to HF too.
            continue
        if hf_ccc_nonzero(c):
            ccc[cp] = pc
    unknown = [cp for cp in cps() if unicodedata.combining(chr(cp)) and cp not in decomp and cp not in ccc]
    print(f"  ccc known to HF: {len(ccc)}, nonzero in Python but 0 in HF: {len(unknown)}")

    print("probing compositions ...", flush=True)
    compose = {}
    for cp in cps():
        if cp not in decomp:
            continue
        d = unicodedata.decomposition(chr(cp))
        if not d or d.startswith("<"):
            continue
        parts = [int(x, 16) for x in d.split()]
        if len(parts) != 2:
            continue
        pair = chr(parts[0]) + chr(parts[1])
        if hf_nfc.normalize_str(pair) == chr(cp):
            compose[(parts[0], parts[1])] = cp
    print(f"  decompositions: {len(decomp)}, composition pairs: {len(compose)}")

    seconds = {b for (_, b) in compose}

    def get_ccc(cp):
        return ccc.get(cp, 0)

    def compose_pair(a, b):
        if LBASE <= a < LBASE + LCOUNT and VBASE <= b < VBASE + VCOUNT:
            return SBASE + ((a - LBASE) * VCOUNT + (b - VBASE)) * TCOUNT
        if SBASE <= a < SBASE + SCOUNT and (a - SBASE) % TCOUNT == 0 and TBASE < b < TBASE + TCOUNT:
            return a + (b - TBASE)
        return compose.get((a, b))

    def model_nfc(s):
        out = []
        for ch in s:
            cp = ord(ch)
            if SBASE <= cp < SBASE + SCOUNT:
                si = cp - SBASE
                out += [LBASE + si // NCOUNT, VBASE + (si % NCOUNT) // TCOUNT]
                if si % TCOUNT:
                    out.append(TBASE + si % TCOUNT)
            else:
                out += decomp.get(cp, [cp])
        # canonical ordering (stable)
        i = 0
        while i < len(out):
            if get_ccc(out[i]) == 0:
                i += 1
                continue
            j = i
            while j < len(out) and get_ccc(out[j]) != 0:
                j += 1
            out[i:j] = sorted(out[i:j], key=get_ccc)
            i = j
        if not out:
            return ""
        res = [out[0]]
        starter = 0 if get_ccc(out[0]) == 0 else -1
        last = 0 if starter == 0 else 256
        for cp in out[1:]:
            cc = get_ccc(cp)
            comp = compose_pair(res[starter], cp) if starter >= 0 else None
            if comp is not None and (last < cc or last == 0):
                res[starter] = comp
                continue
            if cc == 0:
                starter = len(res)
            last = cc
            res.append(cp)
        return "".join(map(chr, res))

    # Quick-check filter: code points that can make NFC(s) != s or interact.
    relevant = set(ccc) | seconds | set(range(VBASE, VBASE + VCOUNT)) | set(range(TBASE + 1, TBASE + TCOUNT))
    for cp in decomp:
        if hf_nfc.normalize_str(chr(cp)) != chr(cp):
            relevant.add(cp)

    print("verifying model against HF NFC ...", flush=True)
    bad = 0
    for cp in cps():
        c = chr(cp)
        if model_nfc(c) != hf_nfc.normalize_str(c):
            bad += 1
            if bad < 10:
                print("  single mismatch", hex(cp))
        if cp not in relevant and hf_nfc.normalize_str(c) != c:
            bad += 1
            print("  quick-check hole", hex(cp))
    rng = random.Random(7)
    marks = sorted(set(ccc) | set(unknown) | seconds)
    bases = sorted({a for (a, _) in compose} | set(decomp) | set(range(0x41, 0x7B)))
    jamo = list(range(LBASE, LBASE + LCOUNT)) + list(range(VBASE, VBASE + VCOUNT)) + \
        list(range(TBASE, TBASE + TCOUNT)) + [SBASE, SBASE + 28, SBASE + 29]
    for (a, b) in compose:
        s = chr(a) + chr(b)
        if model_nfc(s) != hf_nfc.normalize_str(s):
            bad += 1
    for _ in range(300000):
        n = rng.randint(1, 6)
        s = "".join(chr(rng.choice(marks if rng.random() < 0.6 else bases if rng.random() < 0.7 else jamo))
                    for _ in range(n))
        if model_nfc(s) != hf_nfc.normalize_str(s):
            bad += 1
            if bad < 20:
                print("  seq mismatch", [hex(ord(x)) for x in s])
    print(f"  mismatches: {bad}")
    if bad:
        return 1

    # Emit.
    lines = ["// Generated by tools/gen_unicode_tables.py from HuggingFace tokenizers behavior.",
             "// Do not edit by hand.", ""]
    ranges = []
    prev = None
    for cp in range(MAXCP):
        v = cls[cp]
        if v != prev:
            ranges.append((cp, v))
            prev = v
    lines.append("// Regex class runs: {first code point, class}; class holds until the next entry.")
    lines.append(f"static const uint32_t kClassRuns[{len(ranges)}] = {{")
    lines += ["    " + ", ".join(f"0x{cp:X}u << 2 | {v}" for cp, v in ranges[i:i + 6]) + ","
              for i in range(0, len(ranges), 6)]
    lines.append("};")

    cr = sorted(ccc.items())
    lines.append("// Canonical combining class: {code point << 8 | ccc}, sorted.")
    lines.append(f"static const uint32_t kCcc[{len(cr)}] = {{")
    lines += ["    " + ", ".join(f"0x{cp:X}u << 8 | {v}" for cp, v in cr[i:i + 6]) + ","
              for i in range(0, len(cr), 6)]
    lines.append("};")

    dk = sorted(decomp)
    flat, idx = [], []
    for cp in dk:
        idx.append((cp, len(flat), len(decomp[cp])))
        flat += decomp[cp]
    lines.append("// Full canonical decompositions: {code point, offset into kDecompData, length}.")
    lines.append(f"struct DecompEntry {{ uint32_t cp; uint16_t off; uint8_t len; }};")
    lines.append(f"static const DecompEntry kDecomp[{len(idx)}] = {{")
    lines += ["    " + ", ".join(f"{{0x{cp:X}, {o}, {n}}}" for cp, o, n in idx[i:i + 5]) + ","
              for i in range(0, len(idx), 5)]
    lines.append("};")
    lines.append(f"static const uint32_t kDecompData[{len(flat)}] = {{")
    lines += ["    " + ", ".join(f"0x{x:X}" for x in flat[i:i + 10]) + "," for i in range(0, len(flat), 10)]
    lines.append("};")

    ck = sorted(compose.items())
    lines.append("// Primary composites: {(first << 21 | second), composite}, sorted by key.")
    lines.append("struct ComposeEntry { uint64_t key; uint32_t cp; };")
    lines.append(f"static const ComposeEntry kCompose[{len(ck)}] = {{")
    lines += ["    " + ", ".join(f"{{0x{a:X}ull << 21 | 0x{b:X}, 0x{c:X}}}" for (a, b), c in ck[i:i + 4]) + ","
              for i in range(0, len(ck), 4)]
    lines.append("};")

    rel = sorted(relevant)
    rr = []
    for cp in rel:
        if rr and rr[-1][1] + 1 == cp:
            rr[-1][1] = cp
        else:
            rr.append([cp, cp])
    lines.append("// Code points that may change under NFC or interact with neighbors: [first, last].")
    lines.append(f"static const uint32_t kNfcRelevant[{len(rr)}][2] = {{")
    lines += ["    " + ", ".join(f"{{0x{a:X}, 0x{b:X}}}" for a, b in rr[i:i + 6]) + ","
              for i in range(0, len(rr), 6)]
    lines.append("};")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {OUT}: {len(ranges)} class runs, {len(cr)} ccc, {len(idx)} decomps, "
          f"{len(ck)} compositions, {len(rr)} relevant ranges")
    return 0


if __name__ == "__main__":
    sys.exit(main())
