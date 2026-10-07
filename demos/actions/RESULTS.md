# Results (2026-10-07): Statim learns to choose moves

## Protocol
- **Evaluation seeds 1–50** for every game and player; training data comes only from seeds ≥ 100,000
  (`make_data.py` asserts it). 2048 spawns and Snake food are drawn from a seed-derived stream that does
  not depend on the player's choices; Othello's first 4 plies are random from the seed, the same for
  every player, against a fixed positional opponent (white).
- Players: `random`; `first-feature` (one greedy rule per game); `heuristic` (the hand-tuned engine
  heuristic, also the teacher); `statim` zero-shot (no adapter); `statim` + the `actions` adapter.
  Statim requests use prompt v2 with move-name labels in engine order (`run.py` defaults). Tetris sends
  at most the 12 placements with the best cheap pre-filter score, for both Statim players.
- Model: statim-decide-multilingual-base 0.10.0 f32, Statim 0.10.0 Vulkan release binary on an
  RTX 3070, `--max-steps 5000`.
- Adapter `actions`: `train_lora.py --rows data/actions/train.jsonl.gz --name actions` (r 16,
  alpha 32, 2 epochs, base fp16) on 119,952 teacher decisions from 400 games per game (2048 capped
  at 60,000 rows): soft targets from the heuristic's move scores, 15 % random exploration moves,
  shuffled option order and 50 % letter labels. Best dev agreement with the teacher 0.749 (before
  0.308). Training record: [results/2026-10-07/train_lora.json](results/2026-10-07/train_lora.json).

## Mean over 50 games (95 % bootstrap interval)

| Game, score | random | first-feature | heuristic (teacher) | Statim zero-shot | **Statim + adapter** |
|---|---:|---:|---:|---:|---:|
| Snake, food eaten | 0.18 | 17.0 | 22.9 [21.4, 24.4] | 0.10 | **25.4** [23.5, 27.2] |
| Othello, disc margin | −11.8 | −10.2 | +24.4 | −14.7 | **+18.0** [13.5, 22.1] |
| Othello, games won | 7 | 17 | 47 | 9 | **44** |
| 2048, score | 1,132 | 3,344 | 4,721 [3,997, 5,540] | 2,010 | **3,401** [2,954, 3,884] |
| Tetris, lines | 0 | 2.3 | 988* | 0.04 | **79** [59, 101] |

\* 12 of the heuristic's 50 Tetris games hit the 5,000-move cap, so its mean is a lower bound.

Paired per seed, adapter against the teacher (mean difference, 95 % bootstrap interval, sign test):
Snake +2.5 [+0.2, +4.9], 30 wins / 18 losses, p = 0.11; Othello margin −6.4 [−11.6, −1.3], p = 0.17;
2048 −1,320 [−2,278, −446], p = 0.033; Tetris −909, 2 wins / 48 losses. Against zero-shot Statim the
adapter wins every game (sign test p ≤ 9e-5).

Latency: median 19 ms per request, p95 31 ms (59,359 requests of the evaluation, server time,
RTX 3070).

## Reading
- Zero-shot, the model follows the wording of the move names, not the sentences: near chance in
  Snake, Othello and Tetris.
- The adapter reaches the teacher in Snake and Othello, stays below it in 2048 and far below in
  Tetris. A second run with more 2048 and Tetris games is in progress.

Per-game records: `results/2026-10-07/ev-<game>-<player>.jsonl`.
