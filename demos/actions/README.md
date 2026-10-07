# Statim action-choice games

Four small engines (2048, Snake, Othello, Tetris) list the legal moves and describe each one in a
plain sentence computed by simulating it one step. Statim reads the sentences with a one-line
strategy as one `choice` question and returns a probability per move; the highest is played. One
`POST /v1/systemone` request per decision. Zero-shot this does not work (the model follows the move
names, not the sentences); a LoRA adapter trained on the decisions of the engine heuristic does.
Measured results: [RESULTS.md](RESULTS.md). The replays on the site come from `site_traces.py`.

## Run it

Start a Statim server, then run, for example:

```sh
python3 demos/actions/run.py --game 2048 --player statim \
  --base-url http://127.0.0.1:8080 --games 20 --seed 1 \
  --out results/2048-statim.jsonl --trace --summary
```

Players are `statim`, `random`, `first-feature`, and `heuristic`. Games are `2048`, `snake`,
`othello`, and `tetris`. `--model` selects an optional Statim model, `--timeout` controls each HTTP
request, and `--max-steps` caps a game. With `--trace`, decision records precede the final game
record in JSONL; without it, the file contains exactly one line per game. A decision record contains
the rendered state, option descriptions, chosen label, and every returned probability.
`--prompt v2` is the default; use `--prompt v1` for the original numeric-feature prompt. A Statim
run can select a loaded LoRA with `--adapter NAME`. The normal evaluation uses stable move labels
and ordering; `--labels letters` and `--shuffle` enable the corresponding training ablations.

Generate engine-teacher rows for LoRA training and development with:

```sh
python3 demos/actions/make_data.py --games-per-game 400 --seed-base 100000 \
  --out data/actions/train.jsonl.gz
python3 demos/actions/make_data.py --games-per-game 40 --seed-base 200000 \
  --out data/actions/dev.jsonl.gz
```

The generator records the heuristic's soft target while occasionally playing a random move to
visit off-policy states. It shuffles every option list, alternates move and neutral letter labels,
and prints fitted temperatures plus row/drop summaries. Its seeds must be at least 100000.

Run the offline tests with:

```sh
python3 -m unittest discover -s demos/actions -p 'test_*.py' -v
```

## Exact evaluation protocol

For a comparison, choose one initial `--seed S`, game count `N`, and step cap, and keep all three
fixed across all four players. Game `i` uses engine seed `S + i`, for `i` from zero through `N - 1`.
The held-out evaluation protocol uses `--seed 1 --games N`, hence seeds 1 through N, with N below
100000; the generator rejects seeds 0 through 99999, so no evaluation game is used for training
(individual positions can still coincide across seeds, as in any game).
The random baseline also starts its independent action RNG at `S + i`. In 2048 and Snake, every
spawn event consumes the next seed-derived pair of uniform draws. The first maps to the rank in the
current row-major empty-cell list; the second selects 2 versus 4 in 2048 and is deliberately consumed
but unused in Snake. Thus event number, not a player's board shape or legal choices, determines its
luck. Report the requested score and final statistic from every game, plus the
mean, median, minimum, maximum, and the deterministic 10,000-resample bootstrap 95% interval for
the mean printed by `--summary`. Capped games remain in the results and are marked `"capped":true`.

The baselines are fixed:

- Random samples uniformly from all legal actions.
- First-feature maximizes merges in 2048, minimizes food distance among non-death moves in Snake,
  maximizes immediate flips in Othello, and minimizes holes in Tetris. Lexicographic action labels
  break ties.
- Heuristic uses the fixed, hand-tuned weighted sums in `players.py` over every supplied feature and
  is an upper reference, not a learned agent.

With the default v2 prompt, Statim receives one string whose first sentence names the game and whose
remaining sentences describe each simulated move in plain English. Every sentence is at most 140
characters, the state is at most 1,800 characters, criteria contain labels mapped to JSON null, and
the strategy instructions end with `Which move is best?`. The retained v1 prompt sends the compact
board object and numeric feature descriptions. In Tetris only, when more than 12 placements are
legal, a fixed cheap weighted feature score selects the top 12 (ties use action labels); they are
sent under their placement action names (or neutral letters with `--labels letters`). Final Tetris
records say whether that pre-filter was used, and trace records include legal and sent counts. This
pre-filter is part of the Statim system under evaluation; local baselines consider every legal
placement.

Rules are standard: 2048 uses a 4x4 board and 90% two spawns; Snake uses 8x8, length three, and one
food. In Othello, a seed-driven random opening supplies the first four plies (two for each color),
identically for every player; Statim or the baseline takes black from ply five against the fixed
positional-table white opponent. Tetris uses a
10x20 board, seven-bag pieces, hard-drop final placements, and standard line clears.

Scores are accumulated merged-tile value for 2048, foods eaten for Snake, black-minus-white disc
count for Othello, and lines cleared for Tetris. Final records additionally contain the largest tile,
snake length and whether the board was filled (`won`), both disc counts and winner, or line count,
respectively. `--games` must be greater than zero.
