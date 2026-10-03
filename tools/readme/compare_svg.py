#!/usr/bin/env python3
"""Write assets/readme/compare.svg: Statim against ONNX Runtime, the Laya PyTorch reference, Lev and
Jev, as an animated SVG for the README and the site.

    python3 tools/readme/compare_svg.py            # rewrites assets/readme/compare.svg and site/images/compare.svg
    python3 tools/readme/compare_svg.py --check    # fails if the file is out of date

Every number below names the file it comes from; change it there first. The animation (bars grow,
values appear) is CSS inside the SVG, so it also runs where the SVG is shown as an <img>, and it is
off under prefers-reduced-motion. Standard library only.
"""
import argparse
import sys
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[2]
OUTS = [ROOT / "assets/readme/compare.svg", ROOT / "site/images/compare.svg"]  # README and site

STATIM, OTHER = "statim", "other"
# (label, value, display, kind). Lower is better for the first four panels.
PANELS = [
    {"title": "Short text, 8 CPU threads", "unit": "ms per decision batch, lower is better",
     "source": "bench/results/same-session/ (all three measured back to back)",
     "rows": [("Statim", 262, "262 ms", STATIM), ("ONNX Runtime", 266, "266 ms", OTHER),
              ("Laya (PyTorch)", 304, "304 ms", OTHER)]},
    {"title": "770-token text, 8 CPU threads", "unit": "seconds, lower is better",
     "source": "bench/results/same-session/ (all three measured back to back)",
     "rows": [("Statim", 3.044, "3.0 s", STATIM), ("ONNX Runtime", 7.054, "7.1 s", OTHER),
              ("Laya (PyTorch)", 4.850, "4.9 s", OTHER)]},
    {"title": "Process start to first answer", "unit": "seconds, lower is better",
     "source": "docs/ORT.md, docs/reproductions/laya-speed-2026-10-03.md",
     "rows": [("Statim", 0.45, "0.45 s", STATIM), ("ONNX Runtime", 2.06, "2.06 s", OTHER),
              ("Laya (PyTorch)", 5.03, "5.03 s", OTHER)]},
    {"title": "Peak memory at start-up, f32", "unit": "MiB, lower is better",
     "source": "docs/ORT.md, docs/reproductions/laya-speed-2026-10-03.md",
     "rows": [("Statim", 637, "637 MiB", STATIM), ("ONNX Runtime", 729, "729 MiB", OTHER),
              ("Laya (PyTorch)", 2653, "2,653 MiB", OTHER)]},
]
S1BENCH = {"title": "S1Bench: 13 public decision tasks, 3,880 items", "unit": "macro accuracy, higher is better",
           "source": "docs/reproductions/s1bench-2026-10-03.md, s1bench-stage1-2026-10-03.md",
           "rows": [("Jev (hosted API)", 0.761, "0.761", OTHER), ("Lev (4B LLM, GPU)", 0.689, "0.689", OTHER),
                    ("Statim, consensus of two models", 0.651, "0.651", STATIM),
                    ("Statim multilingual 0.7.0", 0.638, "0.638", STATIM),
                    ("Laya (untuned base)", 0.579, "0.579", OTHER)]}

W, PAD = 600, 24
BAR_H, ROW_H = 16, 32
FONT = "ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
MONO = "ui-monospace, 'SFMono-Regular', Menlo, Consolas, monospace"


def panel(x, y, w, p, label_w, delay0, scale_max=None):
    out, rows = [], p["rows"]
    out.append(f'<text x="{x}" y="{y}" class="h">{escape(p["title"])}</text>')
    out.append(f'<text x="{x}" y="{y + 18}" class="u">{escape(p["unit"])}</text>')
    vmax = scale_max or max(v for _, v, _, _ in rows)
    track = w - label_w - 92
    for i, (name, v, shown, kind) in enumerate(rows):
        ry = y + 40 + i * ROW_H
        bw = max(3.0, track * v / vmax)
        d = delay0 + i * 0.08
        out.append(f'<text x="{x}" y="{ry + 11}" class="l {kind}">{escape(name)}</text>')
        out.append(f'<rect x="{x + label_w}" y="{ry}" width="{track}" height="{BAR_H}" rx="4" class="t"/>')
        out.append(f'<rect x="{x + label_w}" y="{ry}" width="{bw:.1f}" height="{BAR_H}" rx="4" '
                   f'class="b {kind}" style="animation-delay:{d:.2f}s"/>')
        out.append(f'<text x="{x + label_w + bw + 8:.1f}" y="{ry + 11}" class="v {kind}" '
                   f'style="animation-delay:{d + 0.55:.2f}s">{escape(shown)}</text>')
    out.append(f'<text x="{x}" y="{y + 40 + len(rows) * ROW_H + 6}" class="s">Source: {escape(p["source"])}</text>')
    return out, 40 + len(rows) * ROW_H + 18


def build():
    parts, y = [], 112
    for i, p in enumerate(PANELS):
        body, h = panel(PAD, y, W - 2 * PAD, p, 128, 0.25 + i * 0.18)
        parts += body
        y += h + 26
    body, h = panel(PAD, y, W - 2 * PAD, S1BENCH, 228, 0.25 + len(PANELS) * 0.18, scale_max=1.0)
    parts += body
    height = y + h + 30
    css = f"""
    .bg{{fill:#0d1014}} .card{{fill:#12171d;stroke:#232b34}}
    text{{font-family:{FONT};fill:#a7b0bb}}
    .title{{font-size:22px;font-weight:600;fill:#e7eaef}} .sub{{font-size:13px;fill:#8a95a1}}
    .h{{font-size:16px;font-weight:600;fill:#e7eaef}} .u{{font-size:12px;fill:#8a95a1}}
    .l{{font-size:13px}} .l.statim{{fill:#e7eaef;font-weight:600}}
    .v{{font-family:{MONO};font-size:13px;fill:#a7b0bb;opacity:0;animation:fade .5s ease-out forwards}}
    .v.statim{{fill:#34d399}}
    .s{{font-size:11px;fill:#6f7a87}}
    .t{{fill:#1a2129}}
    .b{{fill:#6f7a87;transform-box:fill-box;transform-origin:left center;transform:scaleX(0);
        animation:grow 1.1s cubic-bezier(.16,1,.3,1) forwards}}
    .b.statim{{fill:#34d399}}
    .pulse{{fill:none;stroke:#34d399;stroke-width:2;stroke-linecap:round;stroke-dasharray:420;
            stroke-dashoffset:420;animation:draw 1.4s cubic-bezier(.16,1,.3,1) .1s forwards}}
    @keyframes grow{{to{{transform:scaleX(1)}}}}
    @keyframes fade{{from{{opacity:0;transform:translateX(-4px)}}to{{opacity:1;transform:none}}}}
    @keyframes draw{{to{{stroke-dashoffset:0}}}}
    @media (prefers-reduced-motion: reduce){{.b,.v,.pulse{{animation:none;transform:none;opacity:1;stroke-dashoffset:0}}}}
    """
    head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {height}" width="{W}" height="{height}" '
            f'role="img" aria-labelledby="t d">\n<title id="t">Statim compared with ONNX Runtime, Laya, Lev and Jev</title>\n'
            f'<desc id="d">Speed, start-up and memory on one AMD Ryzen 7 5800X CPU against ONNX Runtime and the Laya '
            f'PyTorch reference, and S1Bench accuracy against Lev and Jev. Statim is fastest on long inputs, starts '
            f'fastest and needs the least memory; on short inputs it is level with ONNX Runtime; on S1Bench Statim '
            f'scores 0.651 against 0.689 for Lev and 0.761 for Jev.</desc>\n<style>{" ".join(css.split())}</style>\n'
            f'<rect class="bg" width="{W}" height="{height}" rx="16"/>\n'
            f'<path class="pulse" d="M{PAD} 84 H{PAD + 200} l12 -10 l12 20 l12 -10 H{W - PAD}"/>\n'
            f'<text x="{PAD}" y="40" class="title">Statim, measured against the alternatives</text>\n'
            f'<text x="{PAD}" y="64" class="sub">Same model and tokens on one CPU; '
            f'S1Bench with Lev\'s own harness</text>\n')
    return head + "\n".join(parts) + "\n</svg>\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    svg = build()
    stale = [o for o in OUTS if not o.exists() or o.read_text() != svg]
    if a.check:
        for o in stale:
            print(f"{o.relative_to(ROOT)} is out of date: run tools/readme/compare_svg.py", file=sys.stderr)
        if not stale:
            print("compare.svg up to date")
        return 1 if stale else 0
    for o in OUTS:
        o.write_text(svg)
        print(f"wrote {o.relative_to(ROOT)} ({len(svg)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
