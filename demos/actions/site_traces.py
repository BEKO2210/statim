#!/usr/bin/env python3
"""Record one game per kind for the site replay (site/actions/traces/<game>.jsonl).

Plays the game of the given seed with Statim (and an adapter) against a running server and writes
one line per decision in the format site/actions/actions.js reads: the board before the move, every
option with its sentence and probability, the chosen move, the running stats and the server time.
The first line also carries the provenance (seed, adapter, how the seed was picked).

    python3 demos/actions/site_traces.py --base-url http://127.0.0.1:8098 --adapter actions \
        --game snake --seed 17 --picked "median score of seeds 1-50" --out site/actions/traces/snake.jsonl
"""
import argparse
import json
import time
import urllib.request

from games import GAME_CLASSES
from prompting import (STRATEGIES, build_request, _sentence_2048, _sentence_othello, _sentence_snake,
                       _sentence_tetris, decision_options)


def board_of(game):
    if game.name == "2048":
        return [list(row) for row in game.board]
    if game.name == "snake":
        return {"size": 8, "snake": [list(cell) for cell in game.snake],
                "food": list(game.food) if game.food else None}
    if game.name == "othello":
        symbols = {0: ".", 1: "b", -1: "w"}
        return [[symbols[game.board[r * 8 + c]] for c in range(8)] for r in range(8)]
    # the engine stores row 0 at the bottom; the site draws row 0 at the top
    return {"grid": [list(row) for row in reversed(game.board)], "piece": game.current}


def sentence(game, label, facts, action):
    if game.name == "2048":
        return _sentence_2048(label, facts, game)
    if game.name == "snake":
        return _sentence_snake(label, facts, game)
    if game.name == "othello":
        return _sentence_othello(label, facts, game)
    return _sentence_tetris(label, facts, game, action)


def stats_of(game):
    if game.name == "2048":
        return {"score": game.score, "largest": max(max(row) for row in game.board)}
    if game.name == "snake":
        return {"score": game.score, "length": len(game.snake)}
    if game.name == "othello":
        return {"black": game.board.count(1), "white": game.board.count(-1)}
    return {"lines": game.score}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--game", required=True, choices=sorted(GAME_CLASSES))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--picked", required=True, help="how the seed was chosen, shown on the page")
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    game = GAME_CLASSES[a.game](seed=a.seed)
    lines = []
    while not game.done and len(lines) < a.max_steps:
        _, facts, _ = decision_options(game)
        # the exact request StatimPlayer sent in the evaluation (labels="moves")
        payload, choices, _ = build_request(game, labels="moves")
        texts = {label: sentence(game, label, facts[action], action) for label, action in choices.items()}
        assert all(t in payload["state"] for t in texts.values())
        if a.adapter:
            payload["adapter"] = a.adapter
        t0 = time.perf_counter()
        req = urllib.request.Request(a.base_url + "/v1/systemone", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=120) as resp:
            answer = json.load(resp)["answers"]["move"]
        ms = (time.perf_counter() - t0) * 1000
        probs = answer["probabilities"]
        options = sorted(({"label": str(choices[l]), "text": texts[l],
                           "p": round(float(probs[l]), 4)} for l in choices), key=lambda o: -o["p"])
        record = {"game": a.game, "step": len(lines), "board": board_of(game), "options": options,
                  "chosen": str(choices[answer["choice"]]),
                  "stats": stats_of(game), "ms": round(ms, 1),
                  "strategy": STRATEGIES[a.game]}
        if a.game == "tetris":
            record["piece"] = game.current
        if not lines:
            record.update({"seed": a.seed, "adapter": a.adapter, "picked": a.picked})
        lines.append(record)
        game.apply(choices[answer["choice"]])
    lines[-1]["final"] = stats_of(game)
    with open(a.out, "w") as f:
        for record in lines:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
    print(f"{a.game} seed {a.seed}: {len(lines)} decisions, final {stats_of(game)}, done {game.done}")


if __name__ == "__main__":
    main()
