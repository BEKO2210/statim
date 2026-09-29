#!/usr/bin/env python3
"""Documentation must not drift from the code. CI runs this on every push; it exits 1 on any finding.

    python3 tools/docs/check_docs.py            # all checks
    python3 tools/docs/check_docs.py --verbose  # also list what was checked

Checks, over every tracked Markdown file and the site's HTML (history is exempt, see HISTORY):
  links     a relative link or image, or a github.com link into this repository's main branch,
            points at a file that exists, and a #fragment at a heading (or HTML id) of that file
  paths     a backticked repository path (`tools/finetune/gate.py`, `docs/API.md`) exists; paths
            with placeholders (<...>, *, ...) and generated paths (build/, models/, data/, dist/)
            are skipped
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

# Records of the past: their links and paths are still checked, but not their versions and flags.
HISTORY = {"CHANGELOG.md", "tools/finetune/sources/v6-research.md", "docs/reproductions/clean-room.md",
           "docs/WEITERMACHEN.md"}
# Files whose release download URLs pin the version a recorded run used (with the reason).
PINNED = {"docs/reproductions/clean-room.md": "a dated clean-room reproduction of v0.6.0"}
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


def anchors(path, cache={}):
    if path not in cache:
        text = (ROOT / path).read_text(encoding="utf-8", errors="replace")
        found = set()
        if path.endswith(".md"):
            seen = {}
            in_code = False
            for line in text.splitlines():
                if line.startswith("```"):
                    in_code = not in_code
                if in_code:
                    continue
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
        text = re.sub(r"```.*?```", "", text, flags=re.S)
        text = re.sub(r"`[^`\n]*`", "", text)
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
    blocks = re.findall(r"```(?:bash|sh|shell|console|text)?\n(.*?)```", text, re.S)
    lines = []
    for b in blocks:
        joined = re.sub(r"\\\n\s*", " ", b)
        lines += [l.strip() for l in joined.splitlines()]
    lines += re.findall(r"`((?:\./|\.venv[\w-]*/bin/)?(?:python3?|statim|\S*/statim)\s[^`]+)`", text)
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
        examples = re.findall(r"```.*?```", text, re.S) if path.endswith(".md") else [text]
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
FACTS = [
    ("en-large typed-decisions", [
        ("README.md", cell(EN, 2)), ("README.md", after("| English 0.5.0 |", "typed-decisions " + NUM)),
        ("docs/ROADMAP.md", after("| typed-decisions test |", NUM + " statim-decide-en-large")),
        ("site/index.html", site_card("Statim Decide EN Large", "typed-decisions"))] + site_strip("typed-decisions")),
    ("en-large Banking77", [
        ("README.md", cell(EN, 3)), ("README.md", after("| English 0.5.0 |", "Banking77 " + NUM)),
        ("docs/ROADMAP.md", after("| Banking77, trained on train split |", NUM + " en-large")),
        ("site/index.html", site_card("Statim Decide EN Large", "Banking77"))] + site_strip("Banking77")),
    ("en-large MASSIVE English", [
        ("README.md", cell(EN, 4)), ("README.md", after("| English 0.5.0 |", "MASSIVE English " + NUM)),
        ("docs/ROADMAP.md", after("| MASSIVE, trained |", NUM + " en-large")),
        ("site/index.html", site_card("Statim Decide EN Large", "MASSIVE, English"))]),
    ("en-large AG News zero-shot", [
        ("README.md", after("| English 0.5.0 |", "AG News " + NUM)),
        ("docs/ROADMAP.md", after("| AG News, zero-shot (never trained) |", NUM + " en-large"))]
        + site_strip("AG News, zero-shot")),
    ("multilingual-base typed-decisions", [
        ("README.md", cell(ML, 2)), ("README.md", after("| Multilingual 0.7.0 |", "typed-decisions " + NUM)),
        ("docs/ROADMAP.md", after("| typed-decisions test |", NUM + " statim-decide-multilingual-base")),
        ("site/index.html", site_card("Statim Decide Multilingual Base", "typed-decisions"))]),
    ("multilingual-base Banking77", [
        ("README.md", cell(ML, 3)), ("README.md", after("| Multilingual 0.7.0 |", "Banking77 " + NUM)),
        ("docs/ROADMAP.md", after("| Banking77, trained on train split |", NUM + " multilingual-base")),
        ("site/index.html", site_card("Statim Decide Multilingual Base", "Banking77"))]),
    ("multilingual-base MASSIVE", [
        ("README.md", cell(ML, 4)), ("README.md", after("| Multilingual 0.7.0 |", "MASSIVE " + NUM)),
        ("docs/ROADMAP.md", after("| MASSIVE, trained |", NUM + " multilingual-base"))]
        + site_strip("MASSIVE, 12 languages")),
    ("multilingual-base AG News zero-shot", [
        ("README.md", after("| Multilingual 0.7.0 |", "AG News " + NUM)),
        ("docs/ROADMAP.md", after("| AG News, zero-shot (never trained) |", NUM + " multilingual-base"))]),
    ("14 decision categories, Statim", [
        ("docs/BASELINES.md", cell(r"\*\*mean of categories\*\*", 0)),
        ("docs/BASELINES.md", after("Over 14 decision categories, Statim scores", NUM)),
        ("README.md", cell("14 decision categories, macro accuracy", 0)),
        ("README.md", cell("14-category macro accuracy", 0)),
        ("docs/ROADMAP.md", after("| 14 decision categories, held out |", NUM + " multilingual-base")),
        ("site/index.html", site_card("Statim Decide Multilingual Base", "Decision categories")),
        ("site/index.html", after("decision categories, Statim scores", NUM + " macro accuracy"))]),
    ("14 decision categories, Qwen3-8B", [
        ("docs/BASELINES.md", cell(r"\*\*mean of categories\*\*", 1)),
        ("docs/BASELINES.md", r"Qwen3-8B, an LLM 26 times its\s+size, scores (\d?\.\d+)"),
        ("README.md", after("| 14 decision categories, macro accuracy |", "Qwen3-8B " + NUM)),
        ("README.md", cell("14-category macro accuracy", 1)),
        ("docs/ROADMAP.md", after("| 14 decision categories, held out |", NUM + " Qwen3-8B")),
        ("site/index.html", after("macro accuracy to their", NUM))]),
    ("14 decision categories, mDeBERTa-XNLI", [
        ("docs/BASELINES.md", cell(r"\*\*mean of categories\*\*", 2)),
        ("docs/BASELINES.md", r"mDeBERTa-XNLI scores (\d?\.\d+)"),
        ("README.md", after("| 14 decision categories, macro accuracy |", "XNLI " + NUM)),
        ("README.md", cell("14-category macro accuracy", 2)),
        ("docs/ROADMAP.md", after("| 14 decision categories, held out |", NUM + " mDeBERTa")),
        ("site/index.html", after("macro accuracy to their", r"\d?\.\d+ and " + NUM))]),
]
for col, system, lead in ((0, "Statim", ""), (1, "Qwen3-8B", "Qwen3-8B "), (2, "mDeBERTa-XNLI", "XNLI ")):
    FACTS.append(("Banking77 baseline, " + system, [
        ("docs/BASELINES.md", cell("test/banking77", col)), ("README.md", cell("Banking77", col)),
        ("README.md", after("| Banking77, 77 intents |", lead + NUM))]))
    FACTS.append(("AG News baseline, " + system, [
        ("docs/BASELINES.md", cell("test/ag_news", col)), ("README.md", cell("AG News", col))]))


def check_facts(problems):
    for name, places in FACTS:
        values = []
        for path, pattern in places:
            found = re.findall(pattern, read(path))
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
        check_paths(path, text, problems)
        if path not in HISTORY:
            if path.endswith(".md"):
                check_flags(path, text, problems)
            check_releases(path, text, version, problems)
    check_server(problems)
    check_facts(problems)
    v = subprocess.run([sys.executable, str(ROOT / "tools" / "release" / "check_versions.py")],
                       capture_output=True, text=True)
    if v.returncode:
        problems += ["versions: " + l for l in v.stdout.splitlines() if not l.startswith(version + ":")]
    for p in problems:
        print(p)
    if a.verbose:
        print("checked %d files" % len(files))
    print("%d problem(s)" % len(problems) if problems else "docs: %d files, no drift found" % len(files))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
