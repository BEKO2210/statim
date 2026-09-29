#!/usr/bin/env python3
"""Every place that states Statim's engine version must agree with CMakeLists.txt.

    python3 tools/release/check_versions.py            # exit 1 and list every mismatch
    python3 tools/release/check_versions.py --set 0.9.0  # bump them all (a release)

The version lives in `project(statim VERSION x.y.z)`. The places below repeat it: the server's
compiled-in string, the client SDKs, the demo Space's Dockerfile, the site's footers and JSON-LD,
the README's quick start (release download), the client READMEs, and the examples in the API
reference and the OpenAPI spec. Model versions
(statim-decide-* 0.5.0, 0.7.0) are a different number and are never touched: every pattern below
is anchored on text that only surrounds the engine version. Standard library only, so CI runs it
before anything is built.
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V = r"(\d+\.\d+\.\d+)"

# (file, regex with one group: the version, how many matches the file must have)
SITES = [
    ("include/statim/server.h", r'#define STATIM_VERSION "%s"' % V, 1),
    ("clients/js/package.json", r'"version": "%s"' % V, 1),
    ("clients/js/package-lock.json", r'"name": "@statim/client",\s*"version": "%s"' % V, 2),
    ("clients/js/src/version.ts", r'export const version = "%s"' % V, 1),
    ("clients/python/pyproject.toml", r'^version = "%s"' % V, 1),
    ("clients/python/src/statim/_version.py", r'__version__ = "%s"' % V, 1),
    ("deploy/hf-space/Dockerfile", r"^ARG STATIM_VERSION=%s" % V, 1),
    ("site/index.html", r'"softwareVersion": "%s"' % V, 1),
    ("site/index.html", r"<span>Statim %s\." % V, 1),
    ("site/404.html", r"<span>Statim %s\." % V, 1),
    ("site/impressum/index.html", r"<span>Statim %s\." % V, 1),
    ("site/datenschutz/index.html", r"<span>Statim %s\." % V, 1),
    ("site/license/index.html", r"<span>Statim %s\." % V, 1),
    ("README.md", r"This downloads the v%s Linux" % V, 1),
    ("README.md", r"releases/download/v%s/" % V, 2),
    ("README.md", r"statim-%s-linux-x86_64-cpu" % V, 3),
    ("README.md", r"The current release is v%s:" % V, 1),
    ("docs/API.md", r"In this tree that version is `%s`" % V, 1),
    ("docs/API.md", r'\{"status":"ok","version":"%s"\}' % V, 1),
    ("docs/API.md", r'weights="[a-z0-9_]+",version="%s"' % V, 3),
    ("docs/openapi.yaml", r"^  version: %s$" % V, 1),  # info.version
    ("docs/openapi.yaml", r"^ {6,}version: %s$" % V, 1),  # the /health example
    ("docs/openapi.yaml", r'weights="[a-z0-9_]+",version="%s"' % V, 1),
    ("clients/python/README.md", r"\(server %s\)" % V, 1),
    ("clients/js/README.md", r"the same API as server %s" % V, 1),
]


def project_version():
    text = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    m = re.search(r"project\(statim VERSION " + V, text)
    if not m:
        sys.exit("check_versions: no project(statim VERSION x.y.z) in CMakeLists.txt")
    return m.group(1)


def check(expected):
    problems = []
    for path, pattern, count in SITES:
        text = (ROOT / path).read_text(encoding="utf-8")
        found = [m.group(1) for m in re.finditer(pattern, text, re.M)]
        if len(found) != count:
            problems.append("%s: %d match(es) of %r, expected %d" % (path, len(found), pattern, count))
        wrong = sorted({v for v in found if v != expected})
        if wrong:
            problems.append("%s: version %s, expected %s (%r)" % (path, ", ".join(wrong), expected, pattern))
    return problems


def bump(new):
    """Rewrite CMakeLists.txt and every site to `new`."""
    old = project_version()
    files = {"CMakeLists.txt"} | {p for p, _, _ in SITES}
    for path in sorted(files):
        p = ROOT / path
        text = p.read_text(encoding="utf-8")
        if path == "CMakeLists.txt":
            text = text.replace("project(statim VERSION %s" % old, "project(statim VERSION %s" % new, 1)
        else:
            for site, pattern, _ in SITES:
                if site == path:
                    text = re.sub(pattern, lambda m: m.group(0).replace(m.group(1), new), text, flags=re.M)
        p.write_text(text, encoding="utf-8")
    return old


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", metavar="X.Y.Z", help="bump every version site to this version")
    a = ap.parse_args()
    if a.set:
        if not re.fullmatch(V, a.set):
            sys.exit("check_versions: --set needs x.y.z, got %r" % a.set)
        print("version %s -> %s" % (bump(a.set), a.set))
    expected = project_version()
    problems = check(expected)
    for p in problems:
        print(p)
    print("%s: %s" % (expected, "%d problem(s)" % len(problems) if problems else "all %d version sites agree" % len(SITES)))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
