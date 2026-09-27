#!/usr/bin/env python3
"""Embed the site's subset fonts (SIL OFL 1.1) into src/playground.html as data URIs.

The playground is compiled into the binary and must not load anything from other hosts, so its
fonts live inline. Re-run after changing the fonts in site/assets/fonts:

    python3 tools/playground_fonts.py
"""
import base64
import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FONTS = [("Sora", 400, "Sora-400"), ("Sora", 600, "Sora-600"), ("IBM Plex Mono", 400, "IBMPlexMono-400")]
page = os.path.join(ROOT, "src", "playground.html")
faces = "".join(
    '@font-face{font-family:"%s";font-weight:%d;font-display:swap;src:url(data:font/woff2;base64,%s) format("woff2")}\n'
    % (family, weight, base64.b64encode(open(os.path.join(ROOT, "site", "assets", "fonts", f + ".woff2"), "rb").read()).decode())
    for family, weight, f in FONTS)
html = open(page).read()
new, n = re.subn(r"(/\* fonts:start[^\n]*\*/\n).*?(/\* fonts:end \*/)", lambda m: m.group(1) + faces + m.group(2), html, flags=re.S)
assert n == 1, "fonts markers not found"
open(page, "w").write(new)
print("embedded %d fonts, playground %d bytes" % (len(FONTS), len(new.encode())))
