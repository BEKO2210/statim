#!/usr/bin/env python3
"""Offline tests for hf_publish_adapter.py."""
import hashlib
import json
from pathlib import Path
import subprocess
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest

from hf_publish_adapter import read_gguf_metadata, resolved_name, source_url


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "release" / "hf_publish_adapter.py"


def write_gguf(path, metadata):
    def pack(fmt, value):
        return struct.pack("<" + fmt, value)

    def string(value):
        encoded = value.encode("utf-8")
        return pack("Q", len(encoded)) + encoded

    formats = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
               7: "?", 10: "Q", 11: "q", 12: "d"}

    def encoded_value(kind, value):
        if kind == 8:
            return string(value)
        if kind == 9:
            element_kind, values = value
            return pack("I", element_kind) + pack("Q", len(values)) + b"".join(
                encoded_value(element_kind, item) for item in values
            )
        return pack(formats[kind], value)

    body = b""
    for key, (kind, value) in metadata.items():
        body += string(key) + pack("I", kind) + encoded_value(kind, value)
    path.write_bytes(b"GGUF" + pack("I", 3) + pack("Q", 0) + pack("Q", len(metadata)) + body)


class PublishAdapterTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.work = self.root / "fake-work"
        self.out = self.root / "owner" / "fake-adapter"
        peft = self.work / "pii"
        peft.mkdir(parents=True)
        self.summary = {
            "created": "2099-01-02T03:04:05",
            "base_checkpoint": "/home/person/repo/models/base",
            "n": 37,
            "seed": 7654321,
            "z": 2.25,
            "alpha": 0.031,
            "argv": ["/home/person/repo/tool.py", "--work", "/home/person/run"],
            "categories": {"pii": {
                "status": "ok", "promote": True, "reason": "", "commands": {
                    "train": "'/home/person/repo/python' '/home/person/repo/train.py' --out '/home/person/run'"
                },
                "cells": [
                    {"lang": "en", "base_acc": 0.1234, "adapter_acc": 0.8765,
                     "delta": 0.7531, "se": 0.0123, "verdict": "gain (2 SE)"},
                    {"lang": "de", "base_acc": 0.2, "adapter_acc": 0.4,
                     "delta": 0.2, "se": 0.02, "verdict": "gain (2 SE)"},
                ],
                "family": {"rows": {"delta": 0.2468, "se": 0.0135, "groups": 2, "flag": "gain"},
                           "suites": {"delta": 0.2468, "se": 0.0135, "groups": 1, "flag": "gain"}},
            }},
        }
        train = {
            "category": "pii", "base": "/home/person/repo/models/base",
            "base_sha256": "a" * 64, "mixture": "/home/person/data/mix.jsonl",
            "mixture_sha256": "b" * 64,
            "registry": "/home/person/repo/tools/finetune/sources/v6-keep.json",
            "registry_sha256": "c" * 64,
            "sources": {"v6/nvidia/Nemotron-PII/default": {"rows": 2971, "train": 2968, "dev": 3}},
            "items": {"train": 40, "dev": 3},
            "lora": {"r": 7, "lora_alpha": 19.5, "lora_dropout": 0.17,
                     "target_modules": ["Wqkv"], "wrapped_modules": 11,
                     "trainable_parameters": 12345},
            "updates": 29, "dev_before": {"dev_acc": 0.2345},
            "best": {"dev_acc": 0.7891}, "seconds": 45.6, "peak_memory_mb": 678,
            "args": {"out": "/home/person/run"},
        }
        (self.work / "summary.json").write_text(json.dumps(self.summary), encoding="utf-8")
        self.fingerprint = "e" * 64
        write_gguf(self.work / "pii.lora.gguf", {
            "statim.lora.base_fingerprint": (8, self.fingerprint),
            "statim.lora.base_name": (8, "fake-base"),
        })
        (peft / "adapter_config.json").write_text('{"path":"/home/person/model"}\n', encoding="utf-8")
        (peft / "adapter_model.safetensors").write_bytes(b"tiny safetensors")
        (peft / "train_lora.json").write_text(json.dumps(train), encoding="utf-8")
        self.write_eval(self.work / "pii.base.jsonl", (0.25, 0.5))
        self.write_eval(self.work / "pii.adapter.jsonl", (0.75, 1.0))
        self.base = self.root / "base-q8_0.gguf"
        self.base.write_bytes(b"base")

    def tearDown(self):
        self.temp.cleanup()

    def write_eval(self, path, accuracies, hashes=("hash-en", "hash-de")):
        rows = [
            {"lang": lang, "accuracy": accuracy, "pool_items_sha256": pool_hash}
            for lang, accuracy, pool_hash in zip(("en", "de"), accuracies, hashes)
        ]
        rows.append({"lang": "macro", "accuracy": sum(accuracies) / len(accuracies)})
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def run_tool(self, *extra, include_base=True, skip_load=True):
        command = [
            sys.executable, str(SCRIPT), str(self.work), "--category", "pii",
            "--base-repo", "Beko2210/statim-decide-multilingual-base",
            "--base-version", "9.8.7", "--statim", str(self.root / "statim"),
            "--out", str(self.out),
        ]
        if include_base:
            command.extend(("--base-gguf", str(self.base)))
        if skip_load:
            command.append("--skip-load-check")
        command.extend(extra)
        return subprocess.run(command, cwd=ROOT, text=True, capture_output=True)

    def test_rejects_unpromoted_category(self):
        self.summary["categories"]["pii"]["promote"] = False
        self.summary["categories"]["pii"]["reason"] = "within noise"
        (self.work / "summary.json").write_text(json.dumps(self.summary), encoding="utf-8")
        run = self.run_tool(include_base=False)
        self.assertEqual(run.returncode, 1)
        self.assertIn("refusing to package pii", run.stderr)
        self.assertIn("within noise", run.stderr)

    def test_name_resolution_and_upload_mismatch(self):
        self.assertEqual(
            resolved_name(SimpleNamespace(
                name=None, upload=None,
                base_repo="Beko2210/statim-decide-multilingual-base", category="pii",
            )),
            "statim-decide-multilingual-base-pii",
        )
        self.assertEqual(
            resolved_name(SimpleNamespace(
                name=None, upload="Beko2210/custom-pii",
                base_repo="Beko2210/statim-decide-multilingual-base", category="pii",
            )),
            "custom-pii",
        )
        run = self.run_tool(
            "--upload", "Beko2210/custom-pii", "--name", "different-pii",
            include_base=False,
        )
        self.assertEqual(run.returncode, 1)
        self.assertIn("does not match", run.stderr)

    def test_base_gguf_required_without_skip(self):
        run = self.run_tool(include_base=False, skip_load=False)
        self.assertEqual(run.returncode, 1)
        self.assertIn("--base-gguf is required unless --skip-load-check is given", run.stderr)

    def test_gguf_metadata_reader(self):
        path = self.root / "metadata.gguf"
        write_gguf(path, {
            "text": (8, "hello"), "count": (5, -7), "enabled": (7, True),
            "values": (9, (4, [2, 3, 5])), "ratio": (12, 1.25),
        })
        self.assertEqual(read_gguf_metadata(path), {
            "text": "hello", "count": -7, "enabled": True,
            "values": [2, 3, 5], "ratio": 1.25,
        })

    def test_fingerprint_mismatch_fails_load_check(self):
        statim = self.root / "statim"
        statim.write_text(
            "#!/usr/bin/env python3\nimport json,sys\n"
            "print('statim 9.9' if sys.argv[1] == 'version' else json.dumps({'fingerprint':'" +
            "f" * 64 + "'}))\n",
            encoding="utf-8",
        )
        statim.chmod(0o755)
        run = self.run_tool(skip_load=False)
        self.assertEqual(run.returncode, 1)
        self.assertIn("base fingerprint mismatch", run.stderr)

    def test_package_checksums_card_and_paths(self):
        run = self.run_tool()
        self.assertEqual(run.returncode, 0, run.stderr)
        expected = {
            "LICENSE-MODEL.md", "NOTICE", "README.md", "SHA256SUMS",
            "statim-decide-multilingual-base-pii.lora.gguf", "peft/adapter_config.json",
            "peft/adapter_model.safetensors", "training/train_lora.json",
            "evaluation/summary.json", "evaluation/pii.base.jsonl",
            "evaluation/pii.adapter.jsonl",
        }
        actual = {path.relative_to(self.out).as_posix() for path in self.out.rglob("*") if path.is_file()}
        self.assertEqual(actual, expected)
        lines = (self.out / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
        self.assertEqual([line[66:] for line in lines], sorted(expected - {"SHA256SUMS"}))
        for line in lines:
            digest, relative = line.split("  ", 1)
            self.assertEqual(digest, hashlib.sha256((self.out / relative).read_bytes()).hexdigest())
        card = (self.out / "README.md").read_text(encoding="utf-8")
        for number in ("0.1234", "0.8765", "+75.31", "+24.68", "3.04", "0.031",
                       "7654321", "19.5", "0.17", "12,345", "45.6", "678"):
            self.assertIn(number, card)
        self.assertIn("| **Mean** | 0.1617 | **0.6382** | +47.66 |", card)
        self.assertIn(
            "[`nvidia/Nemotron-PII`](https://huggingface.co/datasets/nvidia/Nemotron-PII)",
            card,
        )
        self.assertIn("| 2,971 | CC-BY-4.0 |", card)
        self.assertIn("# Statim Decide Multilingual Base: PII adapter", card)
        self.assertIn("statim-decide-multilingual-base-pii.lora.gguf", card)
        self.assertEqual(card.count("The pooled family change is"), 1)
        self.assertNotIn("Pooled by rows", card)
        for path in self.out.rglob("*"):
            if path.is_file():
                self.assertNotIn(b"/home/", path.read_bytes(), path)
        notice = (self.out / "NOTICE").read_text(encoding="utf-8")
        required = [line for line in (ROOT / "LICENSE-MODEL.md").read_text().splitlines()
                    if line.startswith("Required Notice:")]
        for line in required:
            self.assertIn(line, notice)
        self.assertIn("licensed under CC BY 4.0; 2,971 rows used", notice)
        self.assertNotIn("DATA_LICENSES.md", notice)

    def test_source_link_rules(self):
        self.assertEqual(source_url({"id": "github:a/b", "url": "https://example.test/url",
                                     "homepage": "https://example.test/home"}),
                         "https://example.test/url")
        self.assertEqual(source_url({"id": "plain", "homepage": "https://example.test/home",
                                     "pinned_url": "https://example.test/pinned"}),
                         "https://example.test/home")
        self.assertEqual(source_url({"id": "plain", "pinned_url": "https://example.test/pinned"}),
                         "https://example.test/pinned")
        self.assertEqual(source_url({"id": "github:a/b"}), "https://github.com/a/b")
        self.assertEqual(source_url({"id": "zenodo:12345-v2"}),
                         "https://zenodo.org/records/12345")
        self.assertEqual(source_url({"id": "owner/name"}),
                         "https://huggingface.co/datasets/owner/name")
        self.assertIsNone(source_url({"id": "plain"}))

    def test_published_checks_table_yes_and_no(self):
        f32_base, f32_adapter = self.root / "f32-base.jsonl", self.root / "f32-adapter.jsonl"
        q8_base, q8_adapter = self.root / "q8-base.jsonl", self.root / "q8-adapter.jsonl"
        self.write_eval(f32_base, (0.25, 0.5))
        self.write_eval(f32_adapter, (0.75, 1.0))
        self.write_eval(q8_base, (0.25, 0.5))
        self.write_eval(q8_adapter, (0.5, 1.0))
        run = self.run_tool(
            "--check", f"f32={f32_base},{f32_adapter}",
            "--check", f"q8_0={q8_base},{q8_adapter}",
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        card = (self.out / "README.md").read_text(encoding="utf-8")
        self.assertIn("## Checked on the published files", card)
        self.assertIn("**2.25 SE = ", card)  # the fixture's z, never a hard-coded 2
        self.assertIn("This run used 2.25 standard errors.", card)
        self.assertIn("| f32 | — | 0.3750 | 0.8750 | +50.00 | yes |", card)
        self.assertIn("| q8_0 | — | 0.3750 | 0.7500 | +37.50 | no |", card)
        for filename in (
            "published-f32-base.jsonl", "published-f32-adapter.jsonl",
            "published-q8_0-base.jsonl", "published-q8_0-adapter.jsonl",
        ):
            self.assertTrue((self.out / "evaluation" / filename).is_file())

    def test_published_checks_compare_only_the_experiment_weights(self):
        summary_path = self.work / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["categories"]["pii"]["commands"]["serve"] = "statim serve -m multilingual=base-f32.gguf"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        q8_base, q8_adapter = self.root / "q8-base.jsonl", self.root / "q8-adapter.jsonl"
        self.write_eval(q8_base, (0.25, 0.5))
        self.write_eval(q8_adapter, (0.5, 1.0))
        run = self.run_tool("--check", f"q8_0={q8_base},{q8_adapter}")
        self.assertEqual(run.returncode, 0, run.stderr)
        card = (self.out / "README.md").read_text(encoding="utf-8")
        self.assertIn("| q8_0 | — | 0.3750 | 0.7500 | +37.50 | — (experiment: f32) |", card)

    def test_front_matter_is_yaml(self):
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML is not installed")
        run = self.run_tool()
        self.assertEqual(run.returncode, 0, run.stderr)
        card = (self.out / "README.md").read_text(encoding="utf-8")
        metadata = yaml.safe_load(card.split("---", 2)[1])
        self.assertEqual(metadata["base_model_relation"], "adapter")
        self.assertEqual(metadata["model-index"][0]["results"][0]["metrics"][0]["value"], 0.8765)


if __name__ == "__main__":
    unittest.main()
