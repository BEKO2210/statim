# S1Bench, stage 1: deployable configurations, fixed before running

Written after the first S1Bench run ([s1bench-2026-10-03.md](s1bench-2026-10-03.md)) and before any
of the runs below. Same harness, tasks, engine binary, scoring and reporting rules as
[s1bench-protocol.md](s1bench-protocol.md). Every configuration listed here is reported, whatever it
scores; none is dropped after the fact.

## Configurations (each one a serve flag or a model file a user can deploy)
| | Model(s) | Flags |
|---|---|---|
| A | statim-decide-en-large 0.5.0, f32 (Hub file SHA-256 from its `SHA256SUMS`) | none |
| B | en-large 0.5.0 f32 + multilingual 0.7.0 f32, loaded as `english` and `multilingual` | `--consensus`, and requests send `"model": "consensus"` (levbench sends `"model": "local"`, which turns the server default off; the wrapper's `--model consensus` sets it). Both checkpoints answer, option log-probabilities averaged |
| C | multilingual 0.7.0 f32 | `--calibrate` (contextual calibration) |

## Exploratory, labelled as such
D. A post-hoc "best of base and fine-tuned" number, computed from the stored per-item records of
the first run by giving each subset the better of the two models. It uses the test labels to choose,
so it is an upper bound for routing, not a result, and it is never quoted as Statim's score.

## Headline rule
The headline S1Bench number stays the first run's 0.638 unless one of A to C is better on the macro
over all 13 subsets **and** on the macro over the 9 subsets whose sources Statim never trained on;
then that configuration is named as the best deployable one, with all others shown next to it.
