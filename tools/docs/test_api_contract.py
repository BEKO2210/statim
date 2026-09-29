#!/usr/bin/env python3
"""Mutation tests for the frozen API v1 compatibility checker."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

try:
    import yaml
except ImportError:
    print("SKIP: PyYAML is required for API contract tests", file=sys.stderr)
    raise SystemExit(77)


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "tools/docs/api_contract.py"
OPENAPI = ROOT / "docs/openapi.yaml"


class ContractCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = yaml.safe_load(OPENAPI.read_text(encoding="utf-8"))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="statim-api-contract-")
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.openapi = self.directory / "openapi.yaml"
        self.contract = self.directory / "contract.json"
        self._write(self.original)
        result = self._run("--write")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def _write(self, document):
        self.openapi.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    def _run(self, action):
        return subprocess.run(
            [sys.executable, str(CHECKER), action, "--openapi", str(self.openapi),
             "--contract", str(self.contract)], capture_output=True, text=True,
        )

    def check_mutation(self, mutate, breaking, needle):
        document = copy.deepcopy(self.original)
        mutate(document)
        self._write(document)
        result = self._run("--check")
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 1 if breaking else 0, output)
        self.assertIn("BREAKING " if breaking else "NOTE ", output)
        self.assertIn(needle, output)

    def check_transition(self, prepare, mutate, breaking, needle):
        document = copy.deepcopy(self.original)
        prepare(document)
        self._write(document)
        result = self._run("--write")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        mutate(document)
        self._write(document)
        result = self._run("--check")
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 1 if breaking else 0, output)
        self.assertIn("BREAKING " if breaking else "NOTE ", output)
        self.assertIn(needle, output)

    def test_operation_removed(self):
        self.check_mutation(lambda d: d["paths"].pop("/ready"), True, "GET /ready")

    def test_documented_status_removed(self):
        self.check_mutation(lambda d: d["paths"]["/health"]["get"]["responses"].pop("200"),
                            True, "documented status removed")

    def test_request_field_removed(self):
        self.check_mutation(lambda d: d["components"]["schemas"]["SystemOneRequest"]
                            ["properties"].pop("lang"), True, "lang")

    def test_request_enum_added_to_free_field(self):
        # "model" accepts any string today; an enum would reject every other value
        self.check_mutation(lambda d: d["components"]["schemas"]["SystemOneRequest"]["properties"]["model"]
                            .__setitem__("enum", ["english"]), True, "request enum added")

    def test_response_enum_added_is_a_note(self):
        self.check_mutation(lambda d: d["components"]["schemas"]["Routing"]["properties"]["engine"]
                            .__setitem__("enum", ["statim"]), False, "response enum value added")

    def test_request_field_made_required(self):
        self.check_mutation(lambda d: d["components"]["schemas"]["SystemOneRequest"]
                            ["required"].append("lang"), True, "request field made required")

    def test_new_required_request_field(self):
        def mutate(document):
            schema = document["components"]["schemas"]["SystemOneRequest"]
            schema["properties"]["trace"] = {"type": "string"}
            schema["required"].append("trace")
        self.check_mutation(mutate, True, "new required request field")

    def test_request_type_narrowed(self):
        self.check_mutation(lambda d: d["components"]["schemas"]["SystemOneRequest"]
                            ["properties"]["adapter"].update(type="string"),
                            True, "request type changed")

    def test_request_type_changed(self):
        self.check_mutation(lambda d: d["components"]["schemas"]["SystemOneRequest"]
                            ["properties"]["lang"].update(type="integer"),
                            True, "request type changed")

    def test_request_enum_value_removed(self):
        self.check_mutation(lambda d: d["components"]["schemas"]["Question"]
                            ["properties"]["type"]["enum"].remove("score"),
                            True, "request enum value removed")

    def test_request_maximum_tightened(self):
        self.check_mutation(lambda d: d["components"]["schemas"]["SystemOneRequest"]
                            ["properties"]["ensemble"].update(maximum=1),
                            True, "request bound tightened")

    def test_request_maximum_loosened_is_additive(self):
        self.check_mutation(lambda d: d["components"]["schemas"]["BatchRequest"]
                            ["properties"]["states"].update(maxItems=512),
                            False, "request bound loosened")

    def test_new_request_bound_is_breaking(self):
        self.check_mutation(lambda d: d["components"]["schemas"]["SystemOneRequest"]
                            ["properties"]["model"].update(maxLength=8),
                            True, "request bound added")

    def test_removed_request_bound_is_additive(self):
        self.check_mutation(lambda d: d["components"]["schemas"]["BatchRequest"]
                            ["properties"]["states"].pop("maxItems"),
                            False, "request bound removed")

    def test_request_minimum_tightened(self):
        self.check_mutation(lambda d: d["components"]["schemas"]["SystemOneRequest"]
                            ["properties"]["ensemble"].update(minimum=2),
                            True, "request bound tightened")

    def test_request_minimum_loosened_is_additive(self):
        self.check_mutation(lambda d: d["components"]["schemas"]["SystemOneRequest"]
                            ["properties"]["ensemble"].update(minimum=0),
                            False, "request bound loosened")

    def test_new_and_changed_request_pattern_are_breaking(self):
        field = lambda d: d["components"]["schemas"]["SystemOneRequest"]["properties"]["model"]
        self.check_mutation(lambda d: field(d).update(pattern="^[a-z]+$"),
                            True, "request validation changed")
        self.check_transition(lambda d: field(d).update(pattern="^[a-z]+$"),
                              lambda d: field(d).update(pattern="^[A-Z]+$"),
                              True, "request validation changed")

    def test_removed_request_pattern_is_additive(self):
        field = lambda d: d["components"]["schemas"]["SystemOneRequest"]["properties"]["model"]
        self.check_transition(lambda d: field(d).update(pattern="^[a-z]+$"),
                              lambda d: field(d).pop("pattern"),
                              False, "request validation removed")

    def test_new_and_changed_request_format_are_breaking(self):
        field = lambda d: d["components"]["schemas"]["SystemOneRequest"]["properties"]["model"]
        self.check_mutation(lambda d: field(d).update(format="hostname"),
                            True, "request validation changed")
        self.check_transition(lambda d: field(d).update(format="hostname"),
                              lambda d: field(d).update(format="uuid"),
                              True, "request validation changed")

    def test_removed_request_format_is_additive(self):
        field = lambda d: d["components"]["schemas"]["SystemOneRequest"]["properties"]["model"]
        self.check_transition(lambda d: field(d).update(format="hostname"),
                              lambda d: field(d).pop("format"),
                              False, "request validation removed")

    def test_request_multiple_of_tightened_and_loosened(self):
        field = lambda d: d["components"]["schemas"]["SystemOneRequest"]["properties"]["ensemble"]
        self.check_transition(lambda d: field(d).update(multipleOf=2),
                              lambda d: field(d).update(multipleOf=4),
                              True, "request bound tightened")
        self.check_transition(lambda d: field(d).update(multipleOf=2),
                              lambda d: field(d).update(multipleOf=1),
                              False, "request bound loosened")

    def test_request_default_added_removed_and_changed_are_breaking(self):
        field = lambda d: d["components"]["schemas"]["SystemOneRequest"]["properties"]["ensemble"]
        self.check_mutation(lambda d: field(d).update(default=1), True, "request default changed")
        self.check_transition(lambda d: field(d).update(default=1),
                              lambda d: field(d).pop("default"),
                              True, "request default changed")
        self.check_transition(lambda d: field(d).update(default=1),
                              lambda d: field(d).update(default=2),
                              True, "request default changed")

    def test_response_field_removed(self):
        self.check_mutation(lambda d: d["components"]["schemas"]["Decision"]
                            ["properties"].pop("model"), True, "response field removed")

    def test_response_field_made_optional(self):
        self.check_mutation(lambda d: d["components"]["schemas"]["Decision"]
                            ["required"].remove("routing"), True, "response field made optional")

    def test_response_type_changed(self):
        self.check_mutation(lambda d: d["components"]["schemas"]["Usage"]
                            ["properties"]["input_tokens"].update(type="string"),
                            True, "response type changed")

    def test_response_bound_change_is_a_note(self):
        self.check_mutation(lambda d: d["components"]["schemas"]["Usage"]
                            ["properties"]["input_tokens"].update(maximum=1000),
                            False, "response bound changed")

    def test_response_default_change_is_a_note(self):
        self.check_mutation(lambda d: d["components"]["schemas"]["Usage"]
                            ["properties"]["input_tokens"].update(default=0),
                            False, "response default changed")

    def test_error_shape_changed(self):
        self.check_mutation(lambda d: d["components"]["schemas"]["Error"]
                            ["properties"]["detail"].update(type="integer"),
                            True, "$.error")

    def test_new_operation_is_additive(self):
        def mutate(document):
            document["paths"]["/version"] = {"get": {"responses": {"200": {
                "description": "Version", "content": {"application/json": {
                    "schema": {"type": "object"}}}}}}}
        self.check_mutation(mutate, False, "operation added")

    def test_new_optional_request_field_is_additive(self):
        self.check_mutation(lambda d: d["components"]["schemas"]["SystemOneRequest"]
                            ["properties"].update(trace={"type": "string"}),
                            False, "new request field")

    def test_new_response_field_is_additive(self):
        self.check_mutation(lambda d: d["components"]["schemas"]["Health"]
                            ["properties"].update(commit={"type": "string"}),
                            False, "new response field")

    def test_new_documented_status_is_additive(self):
        def mutate(document):
            document["paths"]["/health"]["get"]["responses"]["429"] = {
                "description": "Later", "content": {"application/json": {
                    "schema": {"$ref": "#/components/schemas/Error"}}}}
        self.check_mutation(mutate, False, "documented status added")

    def test_new_request_enum_value_is_additive(self):
        self.check_mutation(lambda d: d["components"]["schemas"]["Question"]
                            ["properties"]["type"]["enum"].append("rank"),
                            False, "request enum value added")

    def test_metric_family_removed_or_renamed_is_breaking(self):
        def remove(document):
            content = document["paths"]["/metrics"]["get"]["responses"]["200"]["content"]["text/plain"]
            content["example"] = content["example"].replace(
                "# TYPE statim_requests_total counter\n", "")
        self.check_mutation(remove, True, "metric family removed")

        def rename(document):
            content = document["paths"]["/metrics"]["get"]["responses"]["200"]["content"]["text/plain"]
            content["example"] = content["example"].replace(
                "statim_requests_total", "statim_requests_renamed")
        self.check_mutation(rename, True, "metric family removed")

    def test_metric_family_kind_changed_is_breaking(self):
        def mutate(document):
            content = document["paths"]["/metrics"]["get"]["responses"]["200"]["content"]["text/plain"]
            content["example"] = content["example"].replace(
                "# TYPE statim_requests_total counter",
                "# TYPE statim_requests_total gauge")
        self.check_mutation(mutate, True, "metric kind changed")

    def test_new_metric_family_is_additive(self):
        def mutate(document):
            content = document["paths"]["/metrics"]["get"]["responses"]["200"]["content"]["text/plain"]
            content["example"] += "# TYPE statim_new_total counter\nstatim_new_total 0\n"
        self.check_mutation(mutate, False, "metric family added")

    def test_descriptions_and_examples_are_allowed_notes(self):
        def mutate(document):
            document["info"]["description"] = "Entirely new prose"
            operation = document["paths"]["/v1/systemone"]["post"]
            operation["description"] = "Changed"
            operation["requestBody"]["content"]["application/json"]["examples"]["invoice"] = {
                "value": {"anything": True}}
        document = copy.deepcopy(self.original)
        mutate(document)
        self._write(document)
        result = self._run("--check")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("description or example changed", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
