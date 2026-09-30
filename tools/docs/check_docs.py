#!/usr/bin/env python3
"""Documentation must not drift from the code. CI runs this on every push; it exits 1 on any finding.

    python3 tools/docs/check_docs.py            # all checks
    python3 tools/docs/check_docs.py --verbose  # also list what was checked

Checks, over every tracked Markdown file and the site's HTML (history is exempt, see HISTORY):
  links     a relative link or image, or a github.com link into this repository's main branch,
            points at a file that exists, and a #fragment at a heading (or HTML id) of that file
  paths     a backticked repository path (`tools/finetune/gate.py`, `docs/API.md`) exists; paths
            with placeholders (<...>, *, ...) and generated paths (build/, models/, data/, dist/)
            are skipped, and so are records, which may name a file that was removed since
  flags     every --flag in a documented command is one the program parses: `statim ...` against
            the arguments src/main.cpp compares, `python ... script.py ...` against that script's
            add_argument() calls, `fuzz/run.sh` and `tools/fetch_models.sh` are skipped (positional)
  releases  a download URL of a Statim release names the current version (CMakeLists.txt), except
            in files that record a past run (PINNED)
  versions  tools/release/check_versions.py: every place that repeats the engine version
And over docs/API.md and docs/openapi.yaml, against the C++ source (check_server):
  server    every route, request and question field, HTTP error message, metric family (with the
            /metrics example's TYPE and HELP lines in the server's order) and serve flag; every flag
            src/main.cpp parses is in its usage text
And the published numbers the docs repeat (check_facts, see FACTS): model scores and baselines in
README.md, docs/ROADMAP.md, docs/BASELINES.md and the site must agree with their source.

Standard library only. What it cannot check (numbers, prose claims) is what the audit agents and
the release checklist cover.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Records of the past: their links are still checked, but not their paths, versions and flags.
HISTORY = {"CHANGELOG.md", "tools/finetune/sources/v6-research.md", "docs/reproductions/clean-room.md"}
# Files whose release download URLs pin the version a recorded run used (with the reason).
PINNED = {"docs/reproductions/clean-room.md": "a dated clean-room reproduction of v0.6.0",
          "docs/ORT.md": "the ONNX Runtime comparison, measured with the v0.9.0 release binary"}
GENERATED = ("build", "build-", "models/", "data/", "dist/", "logs/", "runs/", "out/")


def tracked(patterns):
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files", *patterns], capture_output=True, text=True, check=True)
    skip = ("third_party/", "tests/data/", "fuzz/seeds/", "fuzz/regressions/")
    return [p for p in out.stdout.splitlines() if not p.startswith(skip)]


def slug(heading):
    """GitHub's anchor for a Markdown heading."""
    text = re.sub(r"<[^>]+>", "", heading).strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


def split_fences(text):
    """CommonMark fenced code blocks: (the text outside them, [(info string, code), ...]). A fence is
    ``` or ~~~ (three or more), indented at most three spaces, closed by the same character at least
    as long; an unclosed fence runs to the end."""
    prose, blocks, fence, info, code = [], [], None, "", []
    for line in text.splitlines():
        m = FENCE.match(line)
        if fence is None:
            if m and not (m.group(1)[0] == "`" and "`" in m.group(2)):
                fence, info, code = m.group(1), m.group(2).strip(), []
            else:
                prose.append(line)
        elif m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence) and not m.group(2).strip():
            blocks.append((info, "\n".join(code)))
            fence = None
        else:
            code.append(line)
    if fence is not None:
        blocks.append((info, "\n".join(code)))
    return "\n".join(prose), blocks


def anchors(path, cache={}):
    if path not in cache:
        text = (ROOT / path).read_text(encoding="utf-8", errors="replace")
        found = set()
        if path.endswith(".md"):
            seen = {}
            for line in split_fences(text)[0].splitlines():
                m = re.match(r"#{1,6}\s+(.*?)\s*#*\s*$", line)
                if m:
                    s = slug(m.group(1))
                    n = seen.get(s, 0)
                    seen[s] = n + 1
                    found.add(s if n == 0 else "%s-%d" % (s, n))
        found |= set(re.findall(r'\bid="([^"]+)"', text)) | set(re.findall(r'\bname="([^"]+)"', text))
        cache[path] = found
    return cache[path]


def check_links(path, text, problems):
    if path.endswith(".md"):
        text = re.sub(r"`[^`\n]*`", "", split_fences(text)[0])
        targets = re.findall(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)", text)
        targets += re.findall(r'(?:href|src|srcset)="([^"]+)"', text)
    else:
        targets = re.findall(r'(?:href|src)="([^"]+)"', text)
    base = (ROOT / path).parent
    for t in targets:
        t = t.split()[0]
        repo = re.match(r"^https://github\.com/BEKO2210/statim(?:/(?:blob|tree)/main/([^#?]*))?/?(#.*)?$", t)
        if repo:  # a link into this repository's main branch: check it like a relative one
            file_part, frag = repo.group(1) or "README.md", (repo.group(2) or "#")[1:]
            target = ROOT / file_part
        elif re.match(r"^[a-z][a-z0-9+.-]*:", t) or t.startswith(("//", "mailto:")) or t in ("#", ""):
            continue
        elif path.startswith("site/") and t.startswith("/statim/"):
            t = "site/" + t[len("/statim/"):]
            file_part, _, frag = t.partition("#")
            target = ROOT / file_part
        else:
            file_part, _, frag = t.partition("#")
            target = (base / file_part).resolve() if file_part else ROOT / path
        if file_part and not target.exists():
            problems.append("%s: link to missing %s" % (path, t))
            continue
        if target.is_dir():
            target = target / "index.html" if (target / "index.html").exists() else target / "README.md"
        if frag and target.exists() and target.suffix in (".md", ".html"):
            rel = str(target.relative_to(ROOT))
            if frag.lower() not in {a.lower() for a in anchors(rel)}:
                problems.append("%s: link to missing anchor %s" % (path, t))


REPO_DIRS = ("src/", "include/", "tools/", "bench/", "tests/", "docs/", "examples/", "clients/", "deploy/",
             "fuzz/", "site/", "assets/", ".github/")


def check_paths(path, text, problems):
    for p in set(re.findall(r"`([^`\s]+)`", text)):
        p = p.rstrip(".,:;)")
        if not p.startswith(REPO_DIRS) or re.search(r"[<>*{}$]|\.\.\.|/\.cache", p):
            continue
        p = re.sub(r"(::.*|:\d+(-\d+)?)$", "", p)  # file.py::test or file.cpp:12-34
        if not (ROOT / p).exists():
            problems.append("%s: `%s` does not exist" % (path, p))


def statim_flags():
    src = (ROOT / "src" / "main.cpp").read_text(encoding="utf-8")
    return set(re.findall(r'a == "(-{1,2}[a-z0-9-]+)"', src)) | {"--version", "--help"}


def script_flags(script, cache={}):
    if script not in cache:
        p = ROOT / script
        cache[script] = None
        if p.exists() and p.suffix == ".py":
            src = p.read_text(encoding="utf-8", errors="replace")
            flags = set(re.findall(r"""add_argument\(\s*(?:"-[a-zA-Z]",\s*)?["'](--[a-zA-Z0-9-]+)["']""", src))
            flags |= set(re.findall(r"""add_argument\(\s*["']-[a-zA-Z]["'],\s*["'](--[a-zA-Z0-9-]+)["']""", src))
            flags |= set(re.findall(r"""["'](--[a-zA-Z0-9-]+)["']\s*[,)]""", src)) if "add_argument" in src else set()
            cache[script] = flags | {"--help"}
    return cache[script]


def commands(text):
    """Documented shell commands: fenced blocks and inline code, joined across line continuations."""
    prose, blocks = split_fences(text)
    lines = []
    for info, b in blocks:
        if info.split()[:1] not in ([], ["bash"], ["sh"], ["shell"], ["console"], ["text"]):
            continue
        joined = re.sub(r"\\\n\s*", " ", b)
        lines += [l.strip() for l in joined.splitlines()]
    lines += re.findall(r"`((?:\./|\.venv[\w-]*/bin/)?(?:python3?|statim|\S*/statim)\s[^`]+)`", prose)
    return lines


def check_flags(path, text, problems):
    s_flags = statim_flags()
    for line in commands(text):
        line = line.split("#", 1)[0] if " #" in line else line
        for part in re.split(r"\s*(?:&&|\|\||;|\|)\s*", line):
            words = part.split()
            if not words:
                continue
            flags = [w.split("=")[0] for w in words if re.match(r"^--[a-zA-Z]", w)]
            if not flags:
                continue
            prog = next((i for i, w in enumerate(words) if re.search(r"(^|/)statim$", w)), None)
            subcommands = ("serve", "decide", "bench", "info", "version")
            if prog is not None and len(words) > prog + 1 and words[prog + 1] in subcommands:
                bad = [f for f in flags if f not in s_flags]
                if bad:
                    problems.append("%s: statim does not parse %s (in: %s)" % (path, ", ".join(bad), part[:100]))
                continue
            script = next((w for w in words if w.endswith(".py")), None)
            if script and any(re.search(r"(^|/)python3?$", w) for w in words[:words.index(script)]):
                allowed = script_flags(script.lstrip("./"))
                if allowed is None:
                    continue
                after = words[words.index(script) + 1:]
                bad = [f for f in (w.split("=")[0] for w in after if re.match(r"^--[a-zA-Z]", w)) if f not in allowed]
                if bad:
                    problems.append("%s: %s does not accept %s" % (path, script, ", ".join(sorted(set(bad)))))


def read(path):
    return (ROOT / path).read_text(encoding="utf-8", errors="replace")


def cpp_call_args(src, name):
    """The argument text of every `name(...)` call; parentheses inside string literals do not count."""
    for m in re.finditer(r"\b%s\(" % re.escape(name), src):
        i, depth = m.end(), 1
        while depth and i < len(src):
            if src[i] == '"':
                i += 1
                while src[i] != '"':
                    i += 2 if src[i] == "\\" else 1
            elif src[i] in "()":
                depth += 1 if src[i] == "(" else -1
            i += 1
        yield src[m.end():i - 1]


def token(word, text):
    """`word` (a flag or name) occurs in `text` on its own, not as part of a longer one."""
    return re.search(r"(?<![\w-])%s(?![\w-])" % re.escape(word), text) is not None


# Question checks in src/engine.cpp that src/security.cpp always makes first, so no HTTP client
# receives these messages and the API reference need not list them.
ENGINE_ONLY = {"definition must be an object", "no 'instructions'; add the text the model should answer",
               "unknown type; use one of ['choice', 'noul', 'score']"}


def check_server(problems):
    """docs/API.md and docs/openapi.yaml against the server: routes, request fields, error messages,
    metric families and serve flags, each read from the C++ source."""
    srv, sec, main = read("src/server.cpp"), read("src/security.cpp"), read("src/main.cpp")
    docs = {"docs/API.md": read("docs/API.md"), "docs/openapi.yaml": read("docs/openapi.yaml")}
    api, oa = docs["docs/API.md"], docs["docs/openapi.yaml"]

    for method, route in re.findall(r'srv\.(Get|Post)\("([^"]+)"', srv):
        if "`%s %s`" % (method.upper(), route) not in api:
            problems.append("docs/API.md: the endpoint table lacks `%s %s`" % (method.upper(), route))
        if not re.search(r"^  %s:$" % re.escape(route), oa, re.M):
            problems.append("docs/openapi.yaml: paths lack %s" % route)

    for scope, names in re.findall(r"fields\((body|q), \{([^}]*)\}", sec):
        for field in re.findall(r'"(\w+)"', names):
            if "`%s`" % field not in api:
                problems.append("docs/API.md: request field `%s` is not documented" % field)
            if not re.search(r"^\s+%s:\s*$" % field, oa, re.M):
                problems.append("docs/openapi.yaml: request field %s is not a schema property" % field)

    messages = set()
    for path in sorted((ROOT / "src").glob("*.cpp")):
        src = path.read_text(encoding="utf-8")
        calls = ["HttpError", "QuestionError"] + (["err"] if path.name == "engine.cpp" else [])
        for name in calls:
            for args in cpp_call_args(src, name):
                for lit in re.findall(r'"((?:[^"\\\n]|\\.)*)"', args):
                    lit = lit.encode().decode("unicode_escape")
                    if len(lit.strip(" '\":")) >= 12 and lit not in ENGINE_ONLY:
                        messages.add(lit)
    for msg in sorted(messages):
        if msg not in api:
            problems.append("docs/API.md: error message %r (src/) is not documented" % msg)

    handler = srv[srv.index('srv.Get("/metrics"'):]
    handler = handler[:handler.index("srv.", 5)] if "srv." in handler[5:] else handler
    always, _, optional = handler.partition("if (any_adapters)")
    families = re.findall(r"# TYPE (statim_\w+) (\w+)", always)
    helps = re.findall(r"# HELP (statim_\w+) ([^\\\"]+)", always)
    for path, text in docs.items():
        for name in re.findall(r"# TYPE (statim_\w+)", handler):
            if not token(name, text):
                problems.append("%s: metric family %s is not mentioned" % (path, name))
        examples = [code for _, code in split_fences(text)[1]] if path.endswith(".md") else [text]
        for block in examples:
            shown = re.findall(r"# TYPE (statim_\w+) (\w+)", block)
            if shown and shown != families:
                problems.append("%s: a /metrics example's TYPE lines %s differ from the server's %s" % (
                    path, [n for n, _ in shown], [n for n, _ in families]))
        for name, help_text in helps:
            if "# HELP %s %s" % (name, help_text) not in text:
                problems.append("%s: the /metrics example lacks `# HELP %s %s`" % (path, name, help_text))

    usage = main[main.index("void usage()"):]
    usage = usage[:usage.index("\n}\n")]
    for line in main.splitlines():
        group = re.findall(r'a == "(-{1,2}[a-z0-9-]+)"', line)  # one condition: a flag and its aliases
        if group and not any(token(f, usage) for f in group) and not set(group) <= {"-h", "--help"}:
            problems.append("src/main.cpp: %s is parsed but not in the usage text" % " / ".join(group))
    serve = usage[usage.index("statim serve"):usage.index("statim decide")]
    for flag in sorted(set(re.findall(r"(?<![\w-])(--[a-z0-9-]+|-m)(?![\w-])", serve))):
        for path, text in docs.items():
            if not token(flag, text):
                problems.append("%s: serve flag %s is not documented" % (path, flag))


NUM = r"\**(\d?\.\d+)\**"


def cell(first, col):
    """The number opening column `col` (0 = right after the first cell) of the Markdown table row whose
    first cell matches the regex `first`."""
    return r"(?m)^\|\s*%s\s*\|(?:[^|\n]*\|){%d}\s*%s" % (first, col, NUM)


def after(start, pattern):
    """The number in `pattern`, first found after the text `start` on the same line."""
    return r"%s[^\n]*?%s" % (re.escape(start), pattern)


def site_card(model, label):
    """A model card on the site: the number under `label`."""
    return r'<h3>%s</h3>(?:(?!</li>)[\s\S])*?<dt>%s</dt><dd class="num">(\d?\.\d+)</dd>' % (
        re.escape(model), re.escape(label))


def site_strip(title):
    """A benchmark strip on the site: Statim's score, its dot on the axis and its list entry."""
    block = r"<h3>%s</h3>(?:(?!</figure>)[\s\S])*?" % re.escape(title)
    return [("site/index.html", block + r'<span class="score">(\d?\.\d+)</span>'),
            ("site/index.html", block + r'class="dot us" style="--v:(\d?\.\d+)"'),
            ("site/index.html", block + r'<li class="us">[^<]*<span class="num">(\d?\.\d+)</span>')]


EN = r"\[statim-decide-en-large [^]]*\]\([^)]*\)"
ML = r"\[statim-decide-multilingual-base [^]]*\]\([^)]*\)"
# Every published number that the docs repeat. The first place is the source: after a new model or
# baseline run, change it there, and this check lists every copy still to update. A place whose
# wording changed so that its pattern no longer matches is reported too, never skipped.
class _Facts(list):
    """Published facts whose places are ``(path, pattern)`` or ``(path, pattern, heading)``.

    A heading scopes the search to that Markdown section: from the exact heading line through the
    line before the next heading at the same or a higher level. Use it when natural labels repeat in
    different sections of a document.
    """


FACTS = _Facts([
    ("engine binary size", [
        ("README.md", r"Statim uses one (\d+(?:\.\d+)?) MB binary", "## Why Statim"),
        ("README.md", r"\| Runtime footprint \| PyTorch ≥ 1\.2 GB \| \*\*(\d+(?:\.\d+)?) MB\*\*", "### CPU performance"),
        ("site/index.html", r'content="Choice, score and yes/no answers with calibrated confidence from a (\d+(?:\.\d+)?) MB C\+\+ binary'),
        ("site/index.html", r"<li>(\d+(?:\.\d+)?) MB static binary</li>"),
        ("site/index.html", r'<span class="to">One (\d+(?:\.\d+)?) MB binary and one \.gguf file</span>')]),
    ("en-large typed-decisions", [
        ("README.md", cell("typed-decisions", 0), "## Models"),
        ("README.md", cell("typed-decisions", 0), "### Published model gates"),
        ("docs/ROADMAP.md", after("| typed-decisions test |", NUM + " statim-decide-en-large")),
        ("site/index.html", site_card("Statim Decide EN Large", "typed-decisions"))] + site_strip("typed-decisions")),
    ("en-large Banking77", [
        ("README.md", cell("Banking77", 0), "## Models"),
        ("README.md", cell("Banking77", 0), "### Published model gates"),
        ("docs/ROADMAP.md", after("| Banking77, trained on train split |", NUM + " en-large")),
        ("site/index.html", site_card("Statim Decide EN Large", "Banking77"))] + site_strip("Banking77")),
    ("en-large MASSIVE English", [
        ("README.md", cell("MASSIVE", 0), "## Models"),
        ("README.md", cell("MASSIVE", 0), "### Published model gates"),
        ("docs/ROADMAP.md", after("| MASSIVE, trained |", NUM + " en-large")),
        ("site/index.html", site_card("Statim Decide EN Large", "MASSIVE, English"))]),
    ("en-large AG News zero-shot", [
        ("README.md", cell(r"AG News \(never trained\)", 0), "### Published model gates"),
        ("docs/ROADMAP.md", after("| AG News, zero-shot (never trained) |", NUM + " en-large"))]
        + site_strip("AG News, zero-shot")),
    ("multilingual-base typed-decisions", [
        ("README.md", cell("typed-decisions", 1), "## Models"),
        ("README.md", cell("typed-decisions", 1), "### Published model gates"),
        ("docs/ROADMAP.md", after("| typed-decisions test |", NUM + " statim-decide-multilingual-base")),
        ("site/index.html", site_card("Statim Decide Multilingual Base", "typed-decisions"))]),
    ("multilingual-base Banking77", [
        ("README.md", cell("Banking77", 1), "## Models"),
        ("README.md", cell("Banking77", 1), "### Published model gates"),
        ("docs/ROADMAP.md", after("| Banking77, trained on train split |", NUM + " multilingual-base")),
        ("site/index.html", site_card("Statim Decide Multilingual Base", "Banking77"))]),
    ("multilingual-base MASSIVE", [
        ("README.md", cell("MASSIVE", 1), "## Models"),
        ("README.md", cell("MASSIVE", 1), "### Published model gates"),
        ("docs/ROADMAP.md", after("| MASSIVE, trained |", NUM + " multilingual-base"))]
        + site_strip("MASSIVE, 12 languages")),
    ("multilingual-base AG News zero-shot", [
        ("README.md", cell(r"AG News \(never trained\)", 1), "### Published model gates"),
        ("docs/ROADMAP.md", after("| AG News, zero-shot (never trained) |", NUM + " multilingual-base"))]),
    ("14 decision categories, Statim", [
        ("docs/BASELINES.md", cell(r"\*\*mean of categories\*\*", 0)),
        ("docs/BASELINES.md", after("Over 14 decision categories, Statim scores", NUM)),
        ("README.md", cell("14-category macro accuracy", 0), "## At a glance"),
        ("README.md", cell("14-category macro accuracy", 0), "### Trained Statim versus zero-shot general models"),
        ("docs/ROADMAP.md", after("| 14 decision categories, held out |", NUM + " multilingual-base")),
        ("site/index.html", site_card("Statim Decide Multilingual Base", "Decision categories")),
        ("site/index.html", after("decision categories, Statim scores", NUM + " macro accuracy"))]),
    ("14 decision categories, Qwen3-8B", [
        ("docs/BASELINES.md", cell(r"\*\*mean of categories\*\*", 1)),
        ("docs/BASELINES.md", r"Qwen3-8B, an LLM 26 times its\s+size, scores (\d?\.\d+)"),
        ("README.md", cell("14-category macro accuracy", 1), "## At a glance"),
        ("README.md", cell("14-category macro accuracy", 1), "### Trained Statim versus zero-shot general models"),
        ("docs/ROADMAP.md", after("| 14 decision categories, held out |", NUM + " Qwen3-8B")),
        ("site/index.html", after("macro accuracy to their", NUM))]),
    ("14 decision categories, mDeBERTa-XNLI", [
        ("docs/BASELINES.md", cell(r"\*\*mean of categories\*\*", 2)),
        ("docs/BASELINES.md", r"mDeBERTa-XNLI scores (\d?\.\d+)"),
        ("README.md", r"mDeBERTa[^\n]*?" + NUM, "## At a glance"),
        ("README.md", cell("14-category macro accuracy", 2), "### Trained Statim versus zero-shot general models"),
        ("docs/ROADMAP.md", after("| 14 decision categories, held out |", NUM + " mDeBERTa")),
        ("site/index.html", after("macro accuracy to their", r"\d?\.\d+ and " + NUM))]),
])
for col, system, lead in ((0, "Statim", ""), (1, "Qwen3-8B", "Qwen3-8B "), (2, "mDeBERTa-XNLI", "XNLI ")):
    glance = (("README.md", cell("Banking77", col), "## At a glance") if col < 2 else
              ("README.md", r"mDeBERTa-v3 XNLI scored \d?\.\d+[\s\S]*?and " + NUM +
               r" on\s+Banking77", "## At a glance"))
    FACTS.append(("Banking77 baseline, " + system, [
        ("docs/BASELINES.md", cell("test/banking77", col)),
        ("README.md", cell("Banking77", col), "### Trained Statim versus zero-shot general models"),
        glance]))
    FACTS.append(("AG News baseline, " + system, [
        ("docs/BASELINES.md", cell("test/ag_news", col)),
        ("README.md", cell("AG News", col), "### Trained Statim versus zero-shot general models")]))

# The five categories where Qwen3-8B leads: ROADMAP repeats BASELINES' category rows.
for key, name in (("emotion", "emotion"), ("fact_check", "fact-check"), ("sentiment", "sentiment"),
                  ("safety", "safety"), ("pii", "PII")):
    FACTS.append(("%s, Statim 0.7.0" % key, [
        ("docs/BASELINES.md", cell(key, 0)), ("docs/ROADMAP.md", r"\b%s \((\d?\.\d+) vs" % name)]))
    FACTS.append(("%s, Qwen3-8B" % key, [
        ("docs/BASELINES.md", cell(key, 1)), ("docs/ROADMAP.md", r"\b%s \(\d?\.\d+ vs\s+(\d?\.\d+)\)" % name)]))
# The published adapters (docs/ADAPTERS.md is the source).
FACTS += [
    ("PII adapter, base", [
        ("docs/ADAPTERS.md", r"PII: the mean over the 11 cells rises from (\d?\.\d+)"),
        ("docs/BASELINES.md", cell("pii", 0)), ("docs/ROADMAP.md", after("pass the gate for PII (", NUM + " to"))]),
    ("PII adapter", [
        ("docs/ADAPTERS.md", r"PII: the mean over the 11 cells rises from \d?\.\d+ to (\d?\.\d+)"),
        ("docs/ROADMAP.md", after("pass the gate for PII (", r"\d?\.\d+ to " + NUM))]),
    ("emotion adapter, base", [
        ("docs/ADAPTERS.md", r"Emotion: the mean over its six gate cells rises from (\d?\.\d+)"),
        ("docs/BASELINES.md", cell("emotion", 0)), ("docs/ROADMAP.md", after("and emotion (", NUM + " to"))]),
    ("safety adapter, replication base", [
        ("docs/ADAPTERS.md", r"On the 1,350 fresh items, accuracy rises from (\d?\.\d+)"),
        ("docs/ROADMAP.md", after("safety passes too (", NUM + " to"))]),
    ("safety adapter, replication", [
        ("docs/ADAPTERS.md", r"On the 1,350 fresh items, accuracy rises from \d?\.\d+ to (\d?\.\d+)"),
        ("docs/ROADMAP.md", after("safety passes too (", r"\d?\.\d+ to " + NUM))]),
    ("emotion adapter", [
        ("docs/ADAPTERS.md", r"Emotion: the mean over its six gate cells rises from \d?\.\d+ to (\d?\.\d+)"),
        ("docs/ROADMAP.md", after("and emotion (", r"\d?\.\d+ to " + NUM))]),
]


def markdown_section(text, heading):
    """Return the Markdown section beginning at an exact heading line, or an empty string."""
    start = re.search(r"(?m)^%s[ \t]*$" % re.escape(heading), text)
    if not start:
        return ""
    level = len(heading) - len(heading.lstrip("#"))
    following = re.search(r"(?m)^#{1,%d}[ \t]+" % level, text[start.end():])
    end = start.end() + following.start() if following else len(text)
    return text[start.start():end]


def check_facts(problems):
    for name, places in FACTS:
        values = []
        for place in places:
            path, pattern = place[:2]
            text = read(path)
            if len(place) == 3:
                text = markdown_section(text, place[2])
            found = re.findall(pattern, text)
            if not found:
                problems.append("facts: %s: no match in %s (reworded? update FACTS in tools/docs/check_docs.py)"
                                % (name, path))
            values += [(path, v) for v in found]
        for path, v in values[1:]:
            if float(v) != float(values[0][1]):
                problems.append("facts: %s: %s says %s, %s says %s" % (name, path, v, values[0][0], values[0][1]))


def check_releases(path, text, version, problems):
    if path in PINNED:
        return
    for v in set(re.findall(r"releases/download/v(\d+\.\d+\.\d+)/", text)) | \
            set(re.findall(r"statim-(\d+\.\d+\.\d+)-linux-x86_64", text)):
        if v != version:
            problems.append("%s: release %s in a download instruction, current is %s" % (path, v, version))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    version = re.search(r"project\(statim VERSION (\d+\.\d+\.\d+)", (ROOT / "CMakeLists.txt").read_text()).group(1)
    problems = []
    files = tracked(["*.md", "site/*.html", "site/**/*.html"])
    for path in files:
        text = (ROOT / path).read_text(encoding="utf-8", errors="replace")
        check_links(path, text, problems)
        if path not in HISTORY:
            check_paths(path, text, problems)
            if path.endswith(".md"):
                check_flags(path, text, problems)
            check_releases(path, text, version, problems)
    check_server(problems)
    check_facts(problems)
    v = subprocess.run([sys.executable, str(ROOT / "tools" / "release" / "check_versions.py")],
                       capture_output=True, text=True)
    if v.returncode:
        found = ["versions: " + l for l in v.stdout.splitlines() if not l.startswith(version + ":")]
        problems += found or ["versions: check_versions.py exited %d: %s" % (v.returncode, v.stderr.strip()[-500:])]
    for p in problems:
        print(p)
    if a.verbose:
        print("checked %d files" % len(files))
    print("%d problem(s)" % len(problems) if problems else "docs: %d files, no drift found" % len(files))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
