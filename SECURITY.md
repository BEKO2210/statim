# Security policy

## Reporting a vulnerability

Report vulnerabilities privately through GitHub:
[Report a vulnerability](https://github.com/BEKO2210/statim/security/advisories/new). Private
vulnerability reporting is enabled for this repository. Only the maintainers can see the report.

Do not open a public issue, pull request or discussion for a suspected vulnerability.

Please include:

- the affected version (`statim version`) and how it was built or obtained: a release archive, a
  container image, or a source build with its CMake options;
- the component: the server, the CLI, `statim-quantize`, GGUF loading, the tokenizer or an SDK;
- steps or an input that reproduces it, and what you observed.

## What happens next

| Step | Target |
|---|---|
| Acknowledgement of your report | within 3 working days |
| First assessment: confirmed or not, and its severity (CVSS 4.0) | within 10 working days |
| A fix released for high and critical issues | within 30 days of confirmation |
| A fix released for medium and low issues | with the next regular release, at most 90 days |

These are targets, not a contract. If one slips, the advisory thread says why and gives the new
date.

Fixes go through the normal pull request and CI process. They are published as a GitHub Security
Advisory, with a CVE when one applies, and as a CHANGELOG entry. Coordinated disclosure:

- the advisory is published when the fix is released;
- the reporter is credited, unless they prefer not to be;
- if no fix is released within 90 days, the reporter may disclose.

## Supported versions

Until 1.0, only the latest released minor version receives security fixes.

| Version | Security fixes |
|---|---|
| 0.9.x (latest patch release) | yes |
| older than 0.9 | no; upgrade to the latest 0.9 |

From 1.0 on, the latest minor version of the current major version is supported.

## Scope

**In scope:**

- the `statim` binary and its HTTP server;
- `statim-quantize`;
- GGUF and tokenizer loading;
- the official Python and TypeScript SDKs in `clients/`;
- the release archives and container files in this repository;
- the CI and release workflows.

**Out of scope:**

- **Deliberately unauthenticated servers.** A server started with `--allow-unauthenticated` on a
  network that untrusted clients can reach is a configuration that `statim serve` refuses unless
  the operator opts in explicitly.
- **Denial of service beyond the documented limits.** A deployment is expected to set limits
  (`docs/DEPLOY.md`). Load that stays within them, or that only a missing proxy allows, is a
  deployment concern. A request that bypasses a documented limit is in scope.
- **Model quality.** Wrong or biased answers are not vulnerabilities. Report them as normal issues.
- **Third-party vulnerabilities.** Issues in vendored ggml, cpp-httplib or nlohmann/json that do
  not affect Statim. Report those upstream. If one does affect Statim, it is in scope here too.

## Hardening record

[docs/SECURITY.md](docs/SECURITY.md) records the security findings and their fixes, the fuzzing
campaigns, the design limits and the supply-chain controls.
