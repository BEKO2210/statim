# Compatibility, deprecation and upgrades

This page states what a Statim version promises and what an upgrade or a rollback involves. It
applies from 1.0.0. Before 1.0, the HTTP API v1 promise already holds (frozen since 0.9.0); the rest
holds from 1.0.0.

## What the version number covers

Statim follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html) for the engine. The
**engine version** (`statim version`, `/health`) covers:

| Interface | Promise within a major version | Enforced by |
|---|---|---|
| HTTP API v1: routes, request and response fields, status codes, error shape, `/metrics` families | additive changes only ([API.md](API.md#stability-api-v1)) | `tools/docs/api_contract.py --check` and the contract tests in CI |
| CLI: subcommands, flags, exit codes | a flag or subcommand is never removed or changed in meaning; new ones may be added | `check_docs.py` compares the documented flags with the parser |
| Model files: `statim.format` = `statim-decision-v1` | every 1.x engine loads every `statim-decision-v1` file that a 1.x converter wrote | the parity tests on the published files |
| Adapter files: `statim.format` = `statim-lora-v1` | every 1.x engine loads every `statim-lora-v1` file, on the base it was converted for | `lora_*` tests |

The **model versions** (statim-decide-en-large 0.5.0, statim-decide-multilingual-base 0.7.0) are
separate numbers. A new model version changes answers and goes through the promotion gate
([RESULTS.md](RESULTS.md#published-model-gates)). It never changes the file format.

Not covered:

- the wording of error details and log lines, beyond the documented event names;
- the exact probabilities of a model (they are pinned by the parity tests per model file, not by
  the engine version);
- undocumented flags, environment variables marked internal, and `bench/` and `tools/` scripts.

A change to a GGUF metadata key that the engine **reads** is part of the format. New optional keys
are additive. A file format change that an older 1.x engine could not read requires a new
`statim.format` value and a new major version.

## Deprecation

1. **Announce.** A deprecated flag, field, route or behaviour is listed under **Deprecated** in the
   CHANGELOG of the minor release that deprecates it, together with its replacement. The docs mark
   it as deprecated, and using it logs a `warn` event once at start-up or per first use.
2. **Keep.** It keeps working for at least one further minor release and at least three months,
   whichever is longer.
3. **Remove** it only in the next major version, listed under **Removed** with the migration.

**Exception: security.** If keeping a behaviour would leave users exposed, it can change in a patch
release. An example is 0.9.x's fail-closed start without an API key on a network address. The
change ships with a security advisory, a CHANGELOG entry marked **Breaking**, and a one-step
migration, here `--allow-unauthenticated`.

## Upgrading

The release archive holds `statim` and `statim-quantize`. Models and adapters are separate files and
stay as they are.

1. **Read the CHANGELOG** from your version to the new one, especially **Breaking**, **Deprecated**
   and **Security**.
2. **Verify the download.**

   ```sh
   sha256sum --check --ignore-missing SHA256SUMS
   gh attestation verify statim-X.Y.Z-linux-x86_64-cpu.tar.gz --owner BEKO2210
   ```

3. **Test the new binary next to the old one,** on another port, with the production model:
   `statim serve -m multilingual=/var/lib/statim/multilingual.gguf --port 8081`. Then check:
   - `/health` reports the new version and `/ready` answers 200;
   - a few known requests give the answers you expect.

   An unknown flag stops the server with exit status 2, so a flag mismatch shows here and not in
   production.
4. **Swap and restart.** Replace `/usr/local/bin/statim` and `systemctl restart statim`. Requests
   in flight drain before the old process exits, within the unit's `TimeoutStopSec`
   ([DEPLOY.md](DEPLOY.md)). For a container, change the image tag and recreate the container.
5. **Watch** `/ready`, the 503 and 422 rates and the latency in `/metrics` for the first hour.

## Rolling back

Within one major version, a rollback is the upgrade in reverse:

1. Keep the previous binary (or image tag) until the new one has run in production.
2. Put it back and restart.
3. Models and adapters need no change: every 1.x engine reads every 1.x file.

What can stop a rollback, and what to do:

- **A flag that only the newer version knows.** The older binary refuses it with exit status 2.
  Remove the flag from the unit or the compose file first.
- **A model or adapter converted by a newer converter.** Older 1.x engines load it, because the
  format promise above runs in both directions within a major version. A new optional metadata key
  is ignored by an engine that does not know it.
- **Clients that started using a newer additive API field.** Older engines do not send it. Roll back
  the clients' use of it too.

Rolling back across a major version is not supported. Keep the old major's binary and its
configuration together.
