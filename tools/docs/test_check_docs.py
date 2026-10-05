#!/usr/bin/env python3
"""check_docs.py must catch each kind of drift it claims to: every test copies the files a check
reads into a temporary tree, plants one inconsistency, and expects exactly that finding.

    python3 tools/docs/test_check_docs.py
"""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_docs  # noqa: E402

REAL = check_docs.ROOT
SERVER_FILES = ["src/server.cpp", "src/security.cpp", "src/main.cpp", "src/engine.cpp", "src/model.cpp",
                "docs/API.md", "docs/openapi.yaml"]


class Tree(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()).resolve()  # links resolve symlinks (macOS /var)
        check_docs.ROOT = self.tmp
        check_docs.anchors.__defaults__[0].clear()
        for rel in SERVER_FILES + ["README.md"]:
            (self.tmp / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(REAL / rel, self.tmp / rel)
        for extra in (REAL / "src").glob("*.cpp"):
            if not (self.tmp / "src" / extra.name).exists():
                shutil.copy(extra, self.tmp / "src" / extra.name)

    def tearDown(self):
        check_docs.ROOT = REAL
        check_docs.anchors.__defaults__[0].clear()
        shutil.rmtree(self.tmp)

    def edit(self, rel, old, new):
        p = self.tmp / rel
        text = p.read_text(encoding="utf-8")
        self.assertIn(old, text, "fixture text moved in %s" % rel)
        p.write_text(text.replace(old, new, 1), encoding="utf-8")

    def server(self):
        problems = []
        check_docs.check_server(problems)
        return problems


class ServerChecks(Tree):
    def test_current_tree_is_clean(self):
        self.assertEqual(self.server(), [])

    def test_new_route(self):
        self.edit("src/server.cpp", 'srv.Get("/health"', 'srv.Get("/v1/status", nullptr);\n    srv.Get("/health"')
        found = self.server()
        self.assertIn("docs/API.md: the endpoint table lacks `GET /v1/status`", found)
        self.assertIn("docs/openapi.yaml: paths lack /v1/status", found)

    def test_new_request_field(self):
        self.edit("src/security.cpp", '"min_confidence", "adapter"}', '"min_confidence", "adapter", "priority"}')
        found = self.server()
        self.assertIn("docs/API.md: request field `priority` is not documented", found)
        self.assertIn("docs/openapi.yaml: request field priority is not a schema property", found)

    def test_new_error_message(self):
        self.edit("src/security.cpp", 'throw HttpError(413, "too many questions");',
                  'throw HttpError(413, "too many questions per request");')
        self.assertIn("docs/API.md: error message 'too many questions per request' (src/) is not documented",
                      self.server())

    def test_metrics_example_missing_a_family(self):
        self.edit("docs/API.md", "# TYPE statim_in_flight gauge\nstatim_in_flight N\n", "")
        self.assertTrue(any(p.startswith("docs/API.md: a /metrics example's TYPE lines") for p in self.server()))

    def test_metrics_help_text_changed(self):
        self.edit("src/server.cpp", "End-to-end request latency.", "End-to-end latency of a request.")
        found = self.server()
        self.assertIn("docs/API.md: the /metrics example lacks `# HELP statim_request_duration_ms End-to-end "
                      "latency of a request.`", found)
        self.assertIn("docs/openapi.yaml: the /metrics example lacks `# HELP statim_request_duration_ms End-to-end "
                      "latency of a request.`", found)

    def test_new_serve_flag(self):
        self.edit("src/main.cpp", "[--gpu-fast]", "[--gpu-fast] [--warmup]")
        self.edit("src/main.cpp", 'else if (a == "--max-batch")',
                  'else if (a == "--warmup") {}\n            else if (a == "--max-batch")')
        found = self.server()
        self.assertIn("docs/API.md: serve flag --warmup is not documented", found)
        self.assertIn("docs/openapi.yaml: serve flag --warmup is not documented", found)

    def test_flag_missing_from_usage(self):
        self.edit("src/main.cpp", 'else if (a == "--max-batch")',
                  'else if (a == "--secret") {}\n            else if (a == "--max-batch")')
        self.assertIn("src/main.cpp: --secret is parsed but not in the usage text", self.server())


class FactChecks(Tree):
    def setUp(self):
        super().setUp()
        for rel in sorted({place[0] for _, places in check_docs.FACTS for place in places}):
            (self.tmp / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(REAL / rel, self.tmp / rel)

    def facts(self):
        problems = []
        check_docs.check_facts(problems)
        return problems

    def test_current_tree_is_clean(self):
        self.assertEqual(self.facts(), [])

    def test_stale_copy_on_the_site(self):  # the 0.8.1 site still showed the 0.4.0 model's MASSIVE score
        self.edit("site/index.html", '<li class="us">Statim Decide Multilingual<span class="num">0.816</span>',
                  '<li class="us">Statim Decide Multilingual<span class="num">0.772</span>')
        self.assertEqual(self.facts(), ["facts: multilingual-base MASSIVE: site/index.html says 0.772, "
                                        "README.md says 0.816"])

    def test_new_source_value_lists_every_copy(self):
        self.edit("docs/BASELINES.md", "| **mean of categories** | **0.826** |",
                  "| **mean of categories** | **0.751** |")
        found = self.facts()
        self.assertEqual(len(found), 6)  # BASELINES' summary, README twice, ROADMAP, the site's card and note
        self.assertTrue(all(p.startswith("facts: 14 decision categories, Statim: ") for p in found))

    def test_reworded_place_is_reported(self):
        self.edit("docs/ROADMAP.md", "| MASSIVE, trained |", "| MASSIVE (trained) |")
        found = self.facts()
        self.assertIn("facts: en-large MASSIVE English: no match in docs/ROADMAP.md (reworded? update FACTS in "
                      "tools/docs/check_docs.py)", found)

    def test_same_label_is_scoped_to_its_section(self):
        readme = self.tmp / "README.md"
        source = self.tmp / "source.md"
        readme.write_text("## First\n\n| Suite | Score |\n|---|---:|\n| Shared | 0.1 |\n\n"
                          "## Second\n\n| Suite | Score |\n|---|---:|\n| Shared | 0.2 |\n",
                          encoding="utf-8")
        source.write_text("first 0.1\nsecond 0.2\n", encoding="utf-8")
        old_facts = check_docs.FACTS
        check_docs.FACTS = [
            ("first", [("source.md", r"first (\d?\.\d+)"),
                       ("README.md", check_docs.cell("Shared", 0), "## First")]),
            ("second", [("source.md", r"second (\d?\.\d+)"),
                        ("README.md", check_docs.cell("Shared", 0), "## Second")]),
        ]
        try:
            self.assertEqual(self.facts(), [])
            self.edit("README.md", "| Shared | 0.2 |", "| Shared | 0.3 |")
            self.assertEqual(self.facts(),
                             ["facts: second: README.md says 0.3, source.md says 0.2"])
        finally:
            check_docs.FACTS = old_facts


class DocChecks(Tree):
    def run_file(self, rel, text):
        (self.tmp / rel).parent.mkdir(parents=True, exist_ok=True)
        (self.tmp / rel).write_text(text, encoding="utf-8")
        problems = []
        check_docs.check_links(rel, text, problems)
        check_docs.check_paths(rel, text, problems)
        check_docs.check_flags(rel, text, problems)
        check_docs.check_releases(rel, text, "9.9.9", problems)
        return problems

    def test_links_and_anchors(self):
        found = self.run_file("docs/X.md", "[a](../README.md#no-such-heading) [b](missing.md) [c](../README.md)")
        self.assertIn("docs/X.md: link to missing anchor ../README.md#no-such-heading", found)
        self.assertIn("docs/X.md: link to missing missing.md", found)
        self.assertEqual(len(found), 2)

    def test_github_links_into_the_repository(self):
        found = self.run_file("site/x.html", '<a href="https://github.com/BEKO2210/statim#no-such-heading">r</a>'
                              '<a href="https://github.com/BEKO2210/statim/blob/main/docs/NOPE.md">d</a>'
                              '<a href="https://github.com/BEKO2210/statim/blob/main/docs/API.md">ok</a>'
                              '<a href="https://github.com/BEKO2210/statim/releases">ok</a>')
        repo = "https://github.com/BEKO2210/statim"
        self.assertEqual(found, ["site/x.html: link to missing anchor %s#no-such-heading" % repo,
                                 "site/x.html: link to missing %s/blob/main/docs/NOPE.md" % repo])

    def test_fences(self):
        text = ("# Kept\n\n~~~sh\n# not a heading\nstatim serve --turbo\n~~~\n\n    ```\nnot a fence (four spaces)\n"
                "````\n```\n# still code: a shorter fence does not close\n````\n[a](#kept) [b](#not-a-heading)\n")
        found = self.run_file("docs/F.md", text)
        self.assertEqual(len(found), 2)
        self.assertIn("docs/F.md: link to missing anchor #not-a-heading", found)
        self.assertTrue(any(p.startswith("docs/F.md: statim does not parse --turbo") for p in found))

    def test_paths(self):
        found = self.run_file("docs/X.md", "`docs/API.md` `tools/nope.py` `models/<name>.gguf` `src/server.cpp:12-30`")
        self.assertEqual(found, ["docs/X.md: `tools/nope.py` does not exist"])

    def test_statim_flags(self):
        found = self.run_file("docs/X.md", "```sh\nstatim serve -m a.gguf --port 8080 --turbo\n```\n")
        self.assertEqual(len(found), 1)
        self.assertIn("statim does not parse --turbo", found[0])

    def test_release_downloads(self):
        found = self.run_file("docs/X.md", "curl -fLO https://github.com/BEKO2210/statim/releases/download/v0.1.0/x")
        self.assertEqual(found, ["docs/X.md: release 0.1.0 in a download instruction, current is 9.9.9"])


if __name__ == "__main__":
    unittest.main()
