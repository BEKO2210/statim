#!/usr/bin/env python3
"""Create the SPDX 2.3 SBOM shipped beside a Statim release archive.

This tool intentionally uses only the Python standard library.  It inventories the
archive itself and asks readelf about the exact executables in that archive.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "security"))
from vendored_cves import DatabaseError, get_pinned_versions  # noqa: E402


class SbomError(Exception):
    """An input cannot be represented faithfully in the release SBOM."""


def digest(data: bytes, algorithm: str) -> str:
    return hashlib.new(algorithm, data).hexdigest()


def checksum(algorithm: str, value: str) -> dict[str, str]:
    return {"algorithm": algorithm.upper(), "checksumValue": value}


def spdx_id(kind: str, value: str) -> str:
    suffix = re.sub(r"[^A-Za-z0-9.-]+", "-", value).strip("-") or "item"
    return f"SPDXRef-{kind}-{suffix}-{hashlib.sha1(value.encode('utf-8')).hexdigest()[:10]}"


def release_details(archive: Path) -> tuple[str, str]:
    match = re.fullmatch(r"statim-(.+)-linux-x86_64-(cpu|vulkan)\.tar\.gz", archive.name)
    if not match:
        raise SbomError(
            "archive name must be statim-<version>-linux-x86_64-<cpu|vulkan>.tar.gz"
        )
    return match.group(1), match.group(2)


def created_time() -> str:
    value = os.environ.get("SOURCE_DATE_EPOCH")
    try:
        instant = dt.datetime.fromtimestamp(int(value), dt.timezone.utc) if value else dt.datetime.now(dt.timezone.utc)
    except (ValueError, OSError, OverflowError) as exc:
        raise SbomError(f"invalid SOURCE_DATE_EPOCH {value!r}: {exc}") from exc
    return instant.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def unpack_regular_files(archive: Path, destination: Path) -> tuple[str, list[tuple[str, bytes]]]:
    files: list[tuple[str, bytes]] = []
    roots: set[str] = set()
    seen: set[str] = set()
    try:
        with tarfile.open(archive, "r:gz") as bundle:
            for member in bundle.getmembers():
                path = PurePosixPath(member.name)
                if path.is_absolute() or not path.parts or ".." in path.parts:
                    raise SbomError(f"unsafe archive member path: {member.name!r}")
                roots.add(path.parts[0])
                if not member.isfile():
                    continue
                if member.name in seen:
                    raise SbomError(f"duplicate regular file in archive: {member.name}")
                seen.add(member.name)
                stream = bundle.extractfile(member)
                if stream is None:
                    raise SbomError(f"could not read archive member: {member.name}")
                data = stream.read()
                relative = PurePosixPath(*path.parts[1:]).as_posix()
                if not relative or relative == ".":
                    raise SbomError(f"regular file is not below the archive root: {member.name}")
                target = destination.joinpath(*PurePosixPath(relative).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                files.append((relative, data))
    except (tarfile.TarError, OSError) as exc:
        raise SbomError(f"cannot read archive {archive}: {exc}") from exc
    if len(roots) != 1:
        raise SbomError(f"archive must have exactly one top-level directory, found: {sorted(roots)}")
    return next(iter(roots)), sorted(files)


def run_readelf(readelf: str, arguments: list[str], binary: Path) -> str:
    try:
        result = subprocess.run(
            [readelf, *arguments, str(binary)], capture_output=True, text=True, check=False
        )
    except OSError as exc:
        raise SbomError(f"failed to run readelf on {binary.name}: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise SbomError(f"readelf {' '.join(arguments)} failed for {binary.name}: {detail}")
    return result.stdout


def version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def inspect_binaries(readelf: str, binaries: list[Path]) -> tuple[list[str], str | None, str | None]:
    needed: set[str] = set()
    glibc_versions: set[str] = set()
    gcc_versions: set[str] = set()
    for binary in binaries:
        dynamic = run_readelf(readelf, ["-d"], binary)
        needed.update(re.findall(r"\(NEEDED\).*?\[([^]]+)\]", dynamic))

        version_info = run_readelf(readelf, ["--version-info"], binary)
        glibc_versions.update(re.findall(r"(?<![A-Z0-9_])GLIBC_([0-9]+(?:\.[0-9]+)+)", version_info))

        comments = run_readelf(readelf, ["-p", ".comment"], binary)
        for line in comments.splitlines():
            if "GCC:" not in line:
                continue
            candidates = re.findall(r"(?<![A-Za-z0-9])([0-9]+(?:\.[0-9]+){1,3})(?![A-Za-z0-9])", line)
            if candidates:
                gcc_versions.add(candidates[-1])
    glibc_minimum = max(glibc_versions, key=version_key) if glibc_versions else None
    gcc_version = max(gcc_versions, key=version_key) if gcc_versions else None
    return sorted(needed), glibc_minimum, gcc_version


def purl(locator: str) -> list[dict[str, str]]:
    return [{
        "referenceCategory": "PACKAGE-MANAGER",
        "referenceType": "purl",
        "referenceLocator": locator,
    }]


def package(
    name: str,
    identifier: str,
    purpose: str,
    download: str,
    declared: str = "NOASSERTION",
    version: str | None = None,
    **extra: object,
) -> dict[str, object]:
    result: dict[str, object] = {
        "name": name,
        "SPDXID": identifier,
        "primaryPackagePurpose": purpose,
        "downloadLocation": download,
        "filesAnalyzed": False,
        "licenseConcluded": "NOASSERTION",
        "licenseDeclared": declared,
        "copyrightText": "NOASSERTION",
    }
    if version is not None:
        result["versionInfo"] = version
    result.update(extra)
    return result


def build_sbom(archive: Path, source: Path, readelf: str) -> dict[str, object]:
    version, _flavor = release_details(archive)
    archive_bytes = archive.read_bytes()
    archive_sha256 = digest(archive_bytes, "sha256")
    asset_url = f"https://github.com/BEKO2210/statim/releases/download/v{version}/{archive.name}"
    sbom_name = archive.name.removesuffix(".tar.gz") + ".spdx.json"

    try:
        pinned = get_pinned_versions(source)
    except DatabaseError as exc:
        raise SbomError(f"cannot determine pinned component versions: {exc}") from exc

    with tempfile.TemporaryDirectory(prefix="statim-sbom-") as temporary:
        extracted = Path(temporary)
        root_name, archive_files = unpack_regular_files(archive, extracted)
        expected_root = archive.name.removesuffix(".tar.gz")
        if root_name != expected_root:
            raise SbomError(
                f"archive root {root_name!r} does not match archive name {expected_root!r}"
            )
        statim_binary = extracted / "statim"
        if not statim_binary.is_file():
            raise SbomError("archive has no regular 'statim' binary at its root")
        binaries = [statim_binary]
        quantize = extracted / "statim-quantize"
        if quantize.is_file():
            binaries.append(quantize)
        needed, glibc_minimum, gcc_version = inspect_binaries(readelf, binaries)

    archive_id = "SPDXRef-Package-Archive"
    statim_id = "SPDXRef-Package-Statim"
    ggml_id = "SPDXRef-Package-ggml"
    httplib_id = "SPDXRef-Package-cpp-httplib"
    json_id = "SPDXRef-Package-nlohmann-json"
    stdcpp_id = "SPDXRef-Package-libstdcxx"

    file_entries: list[dict[str, object]] = []
    file_ids: dict[str, str] = {}
    file_sha1s: list[str] = []
    for relative, data in archive_files:
        identifier = spdx_id("File", relative)
        file_ids[relative] = identifier
        sha1 = digest(data, "sha1")
        file_sha1s.append(sha1)
        file_entries.append({
            "fileName": relative,
            "SPDXID": identifier,
            "checksums": [checksum("SHA1", sha1), checksum("SHA256", digest(data, "sha256"))],
            "licenseConcluded": "NOASSERTION",
            "licenseInfoInFiles": ["NOASSERTION"],
            "copyrightText": "NOASSERTION",
        })

    packages: list[dict[str, object]] = [
        package(
            archive.name,
            archive_id,
            "ARCHIVE",
            asset_url,
            version=version,
            checksums=[checksum("SHA256", archive_sha256)],
            filesAnalyzed=True,
            packageVerificationCode={
                "packageVerificationCodeValue": hashlib.sha1("".join(sorted(file_sha1s)).encode("ascii")).hexdigest()
            },
        ),
        package(
            "Statim", statim_id, "APPLICATION",
            f"git+https://github.com/BEKO2210/statim@v{version}",
            declared="Apache-2.0", version=version,
            supplier="Organization: BEKO2210",
            externalRefs=purl(f"pkg:github/BEKO2210/statim@v{version}"),
            comment=(f"Requires glibc symbol version GLIBC_{glibc_minimum} or newer."
                     if glibc_minimum else "No GLIBC symbol-version requirement was reported by readelf."),
        ),
        package(
            "ggml", ggml_id, "LIBRARY",
            f"git+https://github.com/ggml-org/ggml@{pinned['ggml_commit']}",
            declared="MIT", version=pinned["ggml_commit"],
            externalRefs=purl(f"pkg:github/ggml-org/ggml@{pinned['ggml_commit']}"),
        ),
        package(
            "cpp-httplib", httplib_id, "LIBRARY", "NOASSERTION",
            declared="MIT", version=pinned["httplib"],
            externalRefs=purl(f"pkg:github/yhirose/cpp-httplib@v{pinned['httplib']}"),
        ),
        package(
            "nlohmann/json", json_id, "LIBRARY", "NOASSERTION",
            declared="MIT", version=pinned["json"],
            externalRefs=purl(f"pkg:github/nlohmann/json@v{pinned['json']}"),
        ),
        package(
            "libstdc++", stdcpp_id, "LIBRARY", "NOASSERTION",
            declared="GPL-3.0-only WITH GCC-exception-3.1",
            version=gcc_version or "NOASSERTION",
            comment="Statically linked into the shipped executables."
                    + (" Compiler version was unavailable in the ELF .comment sections." if not gcc_version else ""),
        ),
    ]

    system_ids: dict[str, str] = {}
    for soname in needed:
        identifier = spdx_id("Package-SystemLibrary", soname)
        system_ids[soname] = identifier
        note = "provided by the operating system, not shipped"
        if soname == "libc.so.6" and glibc_minimum:
            note += f"; minimum required glibc symbol version: GLIBC_{glibc_minimum}"
        packages.append(
            package(
                soname, identifier, "LIBRARY", "NOASSERTION",
                version="NOASSERTION", comment=note,
            )
        )

    relationships: list[dict[str, str]] = [{
        "spdxElementId": "SPDXRef-DOCUMENT",
        "relationshipType": "DESCRIBES",
        "relatedSpdxElement": archive_id,
    }]
    for relative, _data in archive_files:
        relationships.append({
            "spdxElementId": archive_id,
            "relationshipType": "CONTAINS",
            "relatedSpdxElement": file_ids[relative],
        })
        if relative in {"statim", "statim-quantize"}:
            relationships.append({
                "spdxElementId": file_ids[relative],
                "relationshipType": "GENERATED_FROM",
                "relatedSpdxElement": statim_id,
            })
    for dependency in (ggml_id, httplib_id, json_id, stdcpp_id):
        relationships.append({
            "spdxElementId": statim_id,
            "relationshipType": "STATIC_LINK",
            "relatedSpdxElement": dependency,
        })
    for soname in needed:
        relationships.append({
            "spdxElementId": statim_id,
            "relationshipType": "DYNAMIC_LINK",
            "relatedSpdxElement": system_ids[soname],
        })

    return {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": f"{archive.name} SBOM",
        "documentNamespace": (
            f"https://github.com/BEKO2210/statim/releases/v{version}/{sbom_name}/sha256-{archive_sha256}"
        ),
        "creationInfo": {
            "created": created_time(),
            "creators": ["Tool: statim-sbom.py", "Organization: BEKO2210"],
        },
        "documentDescribes": [archive_id],
        "packages": sorted(packages, key=lambda item: str(item["SPDXID"])),
        "files": sorted(file_entries, key=lambda item: str(item["fileName"])),
        "relationships": sorted(
            relationships,
            key=lambda item: (item["spdxElementId"], item["relationshipType"], item["relatedSpdxElement"]),
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path, help="release .tar.gz to inventory")
    parser.add_argument("--source", required=True, type=Path, help="Statim source checkout")
    parser.add_argument("--out", required=True, type=Path, help="output SPDX JSON path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    readelf = shutil.which("readelf")
    if not readelf:
        print("sbom.py: error: readelf is required (install binutils)", file=sys.stderr)
        return 2
    try:
        if not args.archive.is_file():
            raise SbomError(f"archive not found: {args.archive}")
        if not args.source.is_dir():
            raise SbomError(f"source checkout not found: {args.source}")
        result = build_sbom(args.archive.resolve(), args.source.resolve(), readelf)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (SbomError, OSError) as exc:
        print(f"sbom.py: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
