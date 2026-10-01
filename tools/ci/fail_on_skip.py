#!/usr/bin/env python3
"""Fail a CI step when ctest skipped a test.

ctest counts a test that exits with its SKIP_RETURN_CODE (77) as passed, so a server suite that
cannot bind a socket, or a model that is missing, would turn CI green without having run. CI pipes
every ctest run through `tee ctest.log` and then runs this on the log:

    ctest --test-dir build --output-on-failure | tee ctest.log
    python3 tools/ci/fail_on_skip.py ctest.log

Exit 1 with the skipped tests' names, 0 when nothing was skipped. Standard library only.
"""
import re
import sys

SKIPPED = re.compile(r"Test\s+#\d+:\s+(\S+)\s+\.*\s*\*\*\*Skipped")


def skipped_tests(text):
    return SKIPPED.findall(text)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: fail_on_skip.py CTEST_LOG", file=sys.stderr)
        return 2
    names = skipped_tests(open(argv[0], encoding="utf-8", errors="replace").read())
    if names:
        print(f"::error::ctest skipped {len(names)} test(s), which CI must run: {', '.join(names)}")
        return 1
    print("ctest: no skipped tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
