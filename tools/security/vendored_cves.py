#!/usr/bin/env python3
"""vendored_cves.py - Scan vendored components for known vulnerabilities.

Standard library only. Checks:
1. cpp-httplib: OSV.dev and GitHub Security Advisories
2. nlohmann/json: GitHub Security Advisories
3. ggml: GitHub Security Advisories for ggml-org/ggml and ggml-org/llama.cpp
   filtered by ggml|gguf|tensor|quant and triaged in tools/security/cve-triage.json.

Exit codes:
  0: Clean (no known high/critical CVEs, all ggml/GGUF advisories triaged, no expired entries).
  1: Known high or critical vulnerability, untriaged advisory, or expired triage entry.
  2: Database could not be reached or answered unexpectedly.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import urllib.error
import urllib.request


class DatabaseError(Exception):
    """Raised when an external vulnerability database cannot be reached or responds unexpectedly."""


class NetworkError(DatabaseError):
    """Raised on network connection failures."""


def default_http_fetch(
    url: str,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 20,
) -> bytes:
    """Fetch URL contents using urllib.request with GitHub token support."""
    req_headers = {
        "User-Agent": "statim-vendored-cve-checker/1.0",
        "Accept": "application/vnd.github+json, application/json",
    }
    if headers:
        req_headers.update(headers)

    if "api.github.com" in url:
        token = os.environ.get("GITHUB_TOKEN")
        if token and "Authorization" not in req_headers:
            req_headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url, data=data, headers=req_headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise NetworkError(f"HTTP {e.code} for {url}: {e.reason} - {body}") from e
    except urllib.error.URLError as e:
        raise NetworkError(f"Failed to reach {url}: {e.reason}") from e
    except (TimeoutError, OSError) as e:
        raise NetworkError(f"Network error accessing {url}: {e}") from e


def parse_version(v_str: str) -> tuple[int, ...]:
    """Parse a version string into an integer tuple."""
    v = v_str.strip()
    v = re.sub(r"^[vV]\.?", "", v)
    parts = re.findall(r"\d+", v)
    if not parts:
        raise ValueError(f"Invalid version string: '{v_str}'")
    return tuple(int(p) for p in parts)


def compare_versions(v1: str | tuple[int, ...], v2: str | tuple[int, ...]) -> int:
    """Compare two version strings or tuples. Returns -1, 0, or 1."""
    t1 = parse_version(v1) if isinstance(v1, str) else v1
    t2 = parse_version(v2) if isinstance(v2, str) else v2
    max_len = max(len(t1), len(t2))
    p1 = t1 + (0,) * (max_len - len(t1))
    p2 = t2 + (0,) * (max_len - len(t2))
    if p1 < p2:
        return -1
    elif p1 > p2:
        return 1
    return 0


def check_version_clause(ver_tuple: tuple[int, ...], clause: str) -> bool:
    """Evaluate a single version clause like '<= 0.43.4', 'v.0.50.0', 'v0.49.0 (commit 0fa4912)'."""
    cleaned = re.sub(r"\(.*?\)", "", clause).strip()
    m = re.match(r"^(<=|>=|==|!=|<|>|=)?\s*(?:[vV]\.?)?([0-9][0-9.]*)$", cleaned)
    if not m:
        raise ValueError(f"Malformed clause: '{clause}'")
    op, target_str = m.group(1), m.group(2)
    target_ver = parse_version(target_str)
    cmp = compare_versions(ver_tuple, target_ver)

    if op in (None, "", "=", "=="):
        return cmp == 0
    elif op == "<=":
        return cmp <= 0
    elif op == "<":
        return cmp < 0
    elif op == ">=":
        return cmp >= 0
    elif op == ">":
        return cmp > 0
    elif op == "!=":
        return cmp != 0
    raise ValueError(f"Unknown operator: '{op}'")


def is_version_in_range(ver_str: str, range_expr: str) -> bool:
    """Check if version is within vulnerable_version_range.

    Fails closed: malformed ranges are treated as affected with a warning.
    """
    if not range_expr or not range_expr.strip():
        return False
    try:
        ver_tuple = parse_version(ver_str)
        # Handle OR branches if separated by ||
        branches = [b.strip() for b in range_expr.split("||") if b.strip()]
        for branch in branches:
            clauses = [c.strip() for c in branch.split(",") if c.strip()]
            if not clauses:
                continue
            if all(check_version_clause(ver_tuple, c) for c in clauses):
                return True
        return False
    except Exception as e:
        print(
            f"WARNING: malformed range '{range_expr}' ({e}), treating as affected (fails closed).",
            file=sys.stderr,
        )
        return True


def get_pinned_versions(repo_root: Path) -> dict[str, str]:
    """Extract pinned versions from repository files and git metadata."""
    # 1. cpp-httplib
    httplib_file = repo_root / "third_party" / "httplib.version"
    if not httplib_file.is_file():
        raise DatabaseError(f"Missing {httplib_file}")
    m = re.search(r"cpp-httplib\s+v?([0-9.]+)", httplib_file.read_text(encoding="utf-8"))
    if not m:
        raise DatabaseError(f"Could not parse cpp-httplib version from {httplib_file}")
    httplib_ver = m.group(1)

    # 2. nlohmann/json
    json_file = repo_root / "third_party" / "json.hpp"
    if not json_file.is_file():
        raise DatabaseError(f"Missing {json_file}")
    content = json_file.read_text(encoding="utf-8")
    maj = re.search(r"#define\s+NLOHMANN_JSON_VERSION_MAJOR\s+(\d+)", content)
    min_ = re.search(r"#define\s+NLOHMANN_JSON_VERSION_MINOR\s+(\d+)", content)
    pat = re.search(r"#define\s+NLOHMANN_JSON_VERSION_PATCH\s+(\d+)", content)
    if not (maj and min_ and pat):
        raise DatabaseError(f"Could not parse NLOHMANN_JSON_VERSION macros from {json_file}")
    json_ver = f"{maj.group(1)}.{min_.group(1)}.{pat.group(1)}"

    # 3. ggml submodule commit
    try:
        out = subprocess.check_output(
            ["git", "ls-tree", "HEAD", "third_party/ggml"],
            cwd=str(repo_root),
            stderr=subprocess.PIPE,
        ).decode("utf-8")
    except Exception as e:
        raise DatabaseError(f"Failed to query git ls-tree for third_party/ggml: {e}") from e

    parts = out.split()
    if len(parts) < 3 or parts[1] != "commit":
        raise DatabaseError(f"Unexpected git ls-tree output for third_party/ggml: '{out.strip()}'")
    ggml_commit = parts[2]

    return {
        "httplib": httplib_ver,
        "json": json_ver,
        "ggml_commit": ggml_commit,
    }


def load_triage_file(triage_path: Path) -> list[dict]:
    """Load and validate the triage JSON file."""
    if not triage_path.is_file():
        raise DatabaseError(f"Triage file not found: {triage_path}")
    try:
        data = json.loads(triage_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise DatabaseError(f"Failed to parse triage JSON {triage_path}: {e}") from e

    if isinstance(data, dict):
        if "entries" in data and isinstance(data["entries"], list):
            data = data["entries"]
        else:
            entries = []
            for k, v in data.items():
                if isinstance(v, dict):
                    v.setdefault("id", k)
                    entries.append(v)
            data = entries

    if not isinstance(data, list):
        raise DatabaseError(f"Expected list of triage entries in {triage_path}")

    valid_decisions = {"fixed-in-pinned", "not-affected", "accepted-risk"}
    for entry in data:
        if not isinstance(entry, dict):
            raise DatabaseError(f"Invalid entry in {triage_path}: {entry}")
        if "id" not in entry or not entry["id"]:
            raise DatabaseError(f"Entry missing 'id' in {triage_path}: {entry}")
        decision = entry.get("decision")
        if decision not in valid_decisions:
            raise DatabaseError(
                f"Entry {entry['id']} has invalid decision '{decision}', expected one of {valid_decisions}"
            )
        if "review_by" not in entry:
            raise DatabaseError(f"Entry {entry['id']} missing 'review_by' date")
        try:
            datetime.date.fromisoformat(entry["review_by"])
        except ValueError as e:
            raise DatabaseError(
                f"Entry {entry['id']} has invalid review_by date format '{entry['review_by']}': {e}"
            )

    return data


def find_triage_entry(adv: dict, triage_entries: list[dict]) -> dict | None:
    """Find a triage entry matching the advisory by GHSA ID, CVE ID, or alias."""
    ids = set()
    if adv.get("ghsa_id"):
        ids.add(adv["ghsa_id"])
    if adv.get("cve_id"):
        ids.add(adv["cve_id"])
    if adv.get("id"):
        ids.add(adv["id"])
    for alias in adv.get("aliases", []):
        if isinstance(alias, str):
            ids.add(alias)
    for ident in adv.get("identifiers", []):
        if isinstance(ident, dict) and "value" in ident:
            ids.add(ident["value"])

    for entry in triage_entries:
        entry_id = entry.get("id")
        entry_cve = entry.get("cve_id")
        if entry_id in ids or (entry_cve and entry_cve in ids):
            return entry
    return None


def check_triage_expiration(
    triage_entries: list[dict],
    today: datetime.date | None = None,
) -> list[str]:
    """Check all triage entries for expiration."""
    if today is None:
        today = datetime.date.today()
    expired = []
    for entry in triage_entries:
        review_by = datetime.date.fromisoformat(entry["review_by"])
        if today > review_by:
            expired.append(
                f"Triage entry {entry['id']} expired on {entry['review_by']} (decision: {entry.get('decision')})"
            )
    return expired


def check_osv_control(fetch_fn) -> int:
    """Live control step: query OSV for cpp-httplib v0.43.0; must return >= 1 vulnerability."""
    url = "https://api.osv.dev/v1/query"
    payload = json.dumps({
        "package": {
            "name": "https://github.com/yhirose/cpp-httplib",
            "ecosystem": "GIT",
        },
        "version": "v0.43.0",
    }).encode("utf-8")

    try:
        raw = fetch_fn(url, data=payload, headers={"Content-Type": "application/json"})
        data = json.loads(raw.decode("utf-8"))
    except Exception as e:
        print(f"ERROR: OSV control query failed: {e}", file=sys.stderr)
        return 2

    vulns = data.get("vulns", [])
    if not isinstance(vulns, list) or len(vulns) == 0:
        print(
            f"ERROR: OSV control query returned {len(vulns) if isinstance(vulns, list) else 0} "
            "vulnerabilities for cpp-httplib v0.43.0. Database coverage is lost!",
            file=sys.stderr,
        )
        return 2

    print(f"[OK] OSV control check passed ({len(vulns)} vulnerabilities found for cpp-httplib v0.43.0)")
    return 0


def calculate_cvss31_score(vector: str) -> float | None:
    """Calculate CVSS 3.1 base score from a vector string."""
    try:
        parts = dict(p.split(":") for p in vector.split("/") if ":" in p)
        if "AV" not in parts:
            return None
        av = {"N": 0.85, "A": 0.62, "L": 0.55, "P": 0.2}[parts["AV"]]
        ac = {"L": 0.77, "H": 0.44}[parts["AC"]]
        s = parts.get("S", "U")
        if s == "U":
            pr = {"N": 0.85, "L": 0.62, "H": 0.27}[parts["PR"]]
        else:
            pr = {"N": 0.85, "L": 0.68, "H": 0.5}[parts["PR"]]
        ui = {"N": 0.85, "R": 0.62}[parts["UI"]]
        c = {"N": 0.0, "L": 0.22, "H": 0.56}[parts["C"]]
        i = {"N": 0.0, "L": 0.22, "H": 0.56}[parts["I"]]
        a = {"N": 0.0, "L": 0.22, "H": 0.56}[parts["A"]]

        iss = 1.0 - ((1.0 - c) * (1.0 - i) * (1.0 - a))
        if s == "U":
            impact = 6.42 * iss
        else:
            impact = 7.52 * (iss - 0.029) - 3.25 * ((iss - 0.02) ** 15)
        exploitability = 8.22 * av * ac * pr * ui
        if impact <= 0:
            return 0.0
        if s == "U":
            score = min(impact + exploitability, 10.0)
        else:
            score = min(1.08 * (impact + exploitability), 10.0)
        return math.ceil(round(score, 4) * 10) / 10.0
    except Exception:
        return None


def get_osv_severity(vuln: dict) -> str:
    """Extract or estimate severity ('critical', 'high', 'medium', 'low') from OSV vuln."""
    db_sev = vuln.get("database_specific", {}).get("severity")
    if db_sev and isinstance(db_sev, str):
        return db_sev.lower()

    sevs = vuln.get("severity", [])
    if isinstance(sevs, list):
        for s in sevs:
            if isinstance(s, dict):
                score_str = s.get("score")
                if score_str:
                    try:
                        score_val = float(score_str)
                        if score_val >= 9.0:
                            return "critical"
                        elif score_val >= 7.0:
                            return "high"
                        elif score_val >= 4.0:
                            return "medium"
                        return "low"
                    except ValueError:
                        calc = calculate_cvss31_score(score_str)
                        if calc is not None:
                            if calc >= 9.0:
                                return "critical"
                            elif calc >= 7.0:
                                return "high"
                            elif calc >= 4.0:
                                return "medium"
                            return "low"
    return "high"  # fail closed


PAGE_SIZE = 100
MAX_PAGES = 20


def fetch_repo_advisories(repo: str, fetch_fn) -> list[dict]:
    """Every published security advisory of a GitHub repository, following pages until a short
    one. A silent cap would hide advisories, so running past MAX_PAGES is a database error."""
    out: list[dict] = []
    for page in range(1, MAX_PAGES + 1):
        url = (f"https://api.github.com/repos/{repo}/security-advisories"
               f"?per_page={PAGE_SIZE}&page={page}")
        try:
            advs = json.loads(fetch_fn(url).decode("utf-8"))
        except DatabaseError:
            raise
        except Exception as e:
            raise DatabaseError(f"GitHub advisories error for {repo}: {e}") from e
        if not isinstance(advs, list):
            raise DatabaseError(f"GitHub advisories returned non-list for {repo}: {type(advs)}")
        out.extend(advs)
        if len(advs) < PAGE_SIZE:
            return out
    raise DatabaseError(f"GitHub advisories for {repo}: more than {MAX_PAGES} pages")


def check_cpp_httplib(
    version: str,
    fetch_fn,
    triage_entries: list[dict],
) -> tuple[list[str], list[str]]:
    """Check cpp-httplib against OSV and GitHub security advisories."""
    errors: list[str] = []
    warnings: list[str] = []

    # 1. OSV Query
    osv_url = "https://api.osv.dev/v1/query"
    tag = version if version.startswith("v") else f"v{version}"
    payload = json.dumps({
        "package": {
            "name": "https://github.com/yhirose/cpp-httplib",
            "ecosystem": "GIT",
        },
        "version": tag,
    }).encode("utf-8")

    try:
        raw = fetch_fn(osv_url, data=payload, headers={"Content-Type": "application/json"})
        osv_data = json.loads(raw.decode("utf-8"))
    except DatabaseError:
        raise
    except Exception as e:
        raise DatabaseError(f"OSV query error for cpp-httplib {tag}: {e}") from e

    vulns = osv_data.get("vulns", [])
    for v in vulns:
        vuln_id = v.get("id", "UNKNOWN")
        sev = get_osv_severity(v)
        summary = v.get("summary", "")
        entry = find_triage_entry(v, triage_entries)
        if entry:
            warnings.append(
                f"cpp-httplib {tag}: OSV vuln {vuln_id} ({sev}) triaged as {entry.get('decision')}: {summary}"
            )
        elif sev in ("critical", "high"):
            errors.append(f"cpp-httplib {tag}: OSV reports known {sev} vulnerability: {vuln_id} - {summary}")
        else:
            warnings.append(f"cpp-httplib {tag}: OSV reports {sev} vulnerability: {vuln_id} - {summary}")

    # 2. GitHub Security Advisories
    gh_advs = fetch_repo_advisories("yhirose/cpp-httplib", fetch_fn)

    for adv in gh_advs:
        ghsa_id = adv.get("ghsa_id", adv.get("id", "UNKNOWN"))
        sev = (adv.get("severity") or "high").lower()
        summary = adv.get("summary", "")
        vuln_list = adv.get("vulnerabilities", [])

        ranges = []
        if isinstance(vuln_list, list) and vuln_list:
            for item in vuln_list:
                r = item.get("vulnerable_version_range")
                if r:
                    ranges.append(r)
        if adv.get("vulnerable_version_range"):
            ranges.append(adv["vulnerable_version_range"])

        affects = False
        for r in ranges:
            if is_version_in_range(version, r):
                affects = True
                break

        if affects:
            entry = find_triage_entry(adv, triage_entries)
            if entry:
                warnings.append(
                    f"cpp-httplib {version}: advisory {ghsa_id} ({sev}) triaged as {entry.get('decision')}: {summary}"
                )
            elif sev in ("critical", "high"):
                errors.append(
                    f"cpp-httplib {version}: affected by {sev} advisory {ghsa_id} (ranges: {ranges}): {summary}"
                )
            else:
                warnings.append(
                    f"cpp-httplib {version}: affected by {sev} advisory {ghsa_id} (ranges: {ranges}): {summary}"
                )

    return errors, warnings


def check_nlohmann_json(
    version: str,
    fetch_fn,
    triage_entries: list[dict],
) -> tuple[list[str], list[str]]:
    """Check nlohmann/json against GitHub security advisories."""
    errors: list[str] = []
    warnings: list[str] = []

    gh_advs = fetch_repo_advisories("nlohmann/json", fetch_fn)

    for adv in gh_advs:
        ghsa_id = adv.get("ghsa_id", adv.get("id", "UNKNOWN"))
        sev = (adv.get("severity") or "high").lower()
        summary = adv.get("summary", "")
        vuln_list = adv.get("vulnerabilities", [])

        ranges = []
        if isinstance(vuln_list, list) and vuln_list:
            for item in vuln_list:
                r = item.get("vulnerable_version_range")
                if r:
                    ranges.append(r)
        if adv.get("vulnerable_version_range"):
            ranges.append(adv["vulnerable_version_range"])

        affects = False
        for r in ranges:
            if is_version_in_range(version, r):
                affects = True
                break

        if affects:
            entry = find_triage_entry(adv, triage_entries)
            if entry:
                warnings.append(
                    f"nlohmann/json {version}: advisory {ghsa_id} ({sev}) triaged as {entry.get('decision')}: {summary}"
                )
            elif sev in ("critical", "high"):
                errors.append(
                    f"nlohmann/json {version}: affected by {sev} advisory {ghsa_id} (ranges: {ranges}): {summary}"
                )
            else:
                warnings.append(
                    f"nlohmann/json {version}: affected by {sev} advisory {ghsa_id} (ranges: {ranges}): {summary}"
                )

    return errors, warnings


def check_ggml(
    commit: str,
    fetch_fn,
    triage_entries: list[dict],
    today: datetime.date | None = None,
) -> tuple[list[str], list[str]]:
    """Check ggml advisories in ggml-org/ggml and ggml-org/llama.cpp.

    ggml versions cannot be matched to llama.cpp build tags, so every high or critical advisory of
    either repository must be triaged in cve-triage.json; a keyword filter could miss one.
    """
    if today is None:
        today = datetime.date.today()

    errors: list[str] = []
    warnings: list[str] = []

    repos = ["ggml-org/ggml", "ggml-org/llama.cpp"]
    all_advs: list[dict] = []

    for repo in repos:
        advs = fetch_repo_advisories(repo, fetch_fn)

        for adv in advs:
            adv["_source_repo"] = repo
            all_advs.append(adv)

    for adv in all_advs:
        ghsa_id = adv.get("ghsa_id", adv.get("id", "UNKNOWN"))
        cve_id = adv.get("cve_id")
        sev = (adv.get("severity") or "high").lower()
        summary = adv.get("summary") or ""


        entry = find_triage_entry(adv, triage_entries)
        if entry:
            review_by = datetime.date.fromisoformat(entry["review_by"])
            decision = entry.get("decision")
            if today > review_by:
                errors.append(
                    f"ggml advisory {ghsa_id} ({cve_id}): triage entry expired on {entry['review_by']} "
                    f"(decision: {decision})"
                )
            elif decision not in ("fixed-in-pinned", "not-affected", "accepted-risk"):
                errors.append(
                    f"ggml advisory {ghsa_id} ({cve_id}): invalid triage decision '{decision}'"
                )
            else:
                # Valid triage entry
                if sev in ("critical", "high"):
                    # Triaged clean
                    pass
                else:
                    warnings.append(
                        f"ggml advisory {ghsa_id} ({cve_id}) [{sev}]: {decision} - {summary}"
                    )
        else:
            # Untriaged
            if sev in ("critical", "high"):
                errors.append(
                    f"Untriaged {sev} ggml/GGUF advisory {ghsa_id} ({cve_id}): '{summary}'. "
                    "Must be added to tools/security/cve-triage.json."
                )
            else:
                warnings.append(
                    f"Untriaged {sev} ggml/GGUF advisory {ghsa_id} ({cve_id}): '{summary}'"
                )

    return errors, warnings


def run_checks(
    repo_root: Path | None = None,
    fetch_fn=None,
    triage_path: Path | None = None,
    today: datetime.date | None = None,
    run_control: bool = True,
) -> int:
    """Execute all vulnerability checks and return exit code 0, 1, or 2."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent.parent

    if triage_path is None:
        triage_path = repo_root / "tools" / "security" / "cve-triage.json"

    if fetch_fn is None:
        fetch_fn = default_http_fetch

    if today is None:
        today = datetime.date.today()

    try:
        versions = get_pinned_versions(repo_root)
    except DatabaseError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    try:
        triage_entries = load_triage_file(triage_path)
    except DatabaseError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    # Check for expired entries in the triage file
    expired_errors = check_triage_expiration(triage_entries, today=today)

    # Optional control step check
    if run_control:
        control_rc = check_osv_control(fetch_fn)
        if control_rc != 0:
            return control_rc

    all_errors: list[str] = list(expired_errors)
    all_warnings: list[str] = []

    print(
        f"Scanning vendored dependencies: cpp-httplib {versions['httplib']}, "
        f"nlohmann/json {versions['json']}, ggml {versions['ggml_commit'][:8]}..."
    )

    try:
        # 1. cpp-httplib
        h_err, h_warn = check_cpp_httplib(versions["httplib"], fetch_fn, triage_entries)
        all_errors.extend(h_err)
        all_warnings.extend(h_warn)

        # 2. nlohmann/json
        j_err, j_warn = check_nlohmann_json(versions["json"], fetch_fn, triage_entries)
        all_errors.extend(j_err)
        all_warnings.extend(j_warn)

        # 3. ggml
        g_err, g_warn = check_ggml(versions["ggml_commit"], fetch_fn, triage_entries, today=today)
        all_errors.extend(g_err)
        all_warnings.extend(g_warn)

    except DatabaseError as e:
        print(f"ERROR: Database check failed: {e}", file=sys.stderr)
        return 2

    # Print results
    if all_warnings:
        print(f"\n--- Warnings ({len(all_warnings)}) ---")
        for w in all_warnings:
            print(f"WARN: {w}")

    if all_errors:
        print(f"\n--- Errors ({len(all_errors)}) ---", file=sys.stderr)
        for err in all_errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1

    print("\n[OK] Clean: No unhandled or active high/critical vulnerabilities found in vendored dependencies.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check vendored dependencies for known CVEs")
    parser.add_argument(
        "--control",
        action="store_true",
        help="Run only the live OSV database coverage control check",
    )
    parser.add_argument(
        "--no-control",
        action="store_true",
        help="Skip the live OSV control check during standard scan",
    )
    parser.add_argument(
        "--triage-file",
        type=Path,
        default=None,
        help="Path to cve-triage.json (default: tools/security/cve-triage.json)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Path to repository root (default: inferred from script location)",
    )

    args = parser.parse_args(argv)

    if args.control:
        return check_osv_control(default_http_fetch)

    rc = run_checks(
        repo_root=args.repo_root,
        triage_path=args.triage_file,
        run_control=not args.no_control,
    )
    return rc


if __name__ == "__main__":
    sys.exit(main())
