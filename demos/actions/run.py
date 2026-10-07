#!/usr/bin/env python3
"""Run reproducible action-choice game evaluations."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from games import GAME_CLASSES
from players import FirstFeaturePlayer, HeuristicPlayer, RandomPlayer, StatimPlayer
from prompting import describe


def make_player(args, seed):
    if args.player == "statim":
        return StatimPlayer(args.base_url, args.model, args.timeout, args.prompt,
                            args.adapter, args.labels, args.shuffle, seed)
    if args.player == "random":
        return RandomPlayer(seed)
    if args.player == "first-feature":
        return FirstFeaturePlayer()
    return HeuristicPlayer()


def final_stats(game):
    if game.name == "2048":
        return {"largest_tile": max(max(row) for row in game.board)}
    if game.name == "snake":
        return {"length": len(game.snake), "won": game.won}
    if game.name == "othello":
        black, white = game.board.count(1), game.board.count(-1)
        return {"black_discs": black, "white_discs": white,
                "win": "black" if black > white else "white" if white > black else "draw"}
    return {"lines": game.lines}


def bootstrap_interval(values, seed, samples=10000):
    if not values or samples <= 0:
        return [None, None]
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choice(values) for _ in values) for _ in range(samples))
    return [means[round((samples - 1) * .025)], means[round((samples - 1) * .975)]]


def print_summary(scores, seed):
    interval = bootstrap_interval(scores, seed)
    print(json.dumps({
        "games": len(scores), "mean": statistics.fmean(scores), "median": statistics.median(scores),
        "min": min(scores), "max": max(scores), "bootstrap_95_mean": interval,
    }, sort_keys=True))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", choices=GAME_CLASSES, required=True)
    parser.add_argument("--player", choices=("statim", "random", "first-feature", "heuristic"), required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model")
    parser.add_argument("--adapter")
    parser.add_argument("--prompt", choices=("v1", "v2"), default="v2")
    parser.add_argument("--labels", choices=("moves", "letters"), default="moves")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--out", required=True)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args(argv)
    if args.games <= 0:
        parser.error("--games must be greater than zero")

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    scores = []
    with output.open("w", encoding="utf-8") as stream:
        for game_number in range(args.games):
            game_seed = args.seed + game_number
            game = GAME_CLASSES[args.game](game_seed)
            player = make_player(args, game_seed)
            prefilter_decisions = 0
            max_legal_options = 0
            while not game.done and game.steps < args.max_steps:
                state = game.render_text()
                action, probabilities = player.choose(game)
                if args.trace:
                    if player.last_options is not None:
                        options = player.last_options
                    else:
                        options = {move: describe(game.name, game.features(move)) for move in game.legal_actions()}
                    record = {"type": "decision", "game_index": game_number, "seed": game_seed,
                              "step": game.steps, "state": state, "options": options,
                              "chosen": action, "probabilities": probabilities}
                    if player.last_prefilter is not None:
                        record["tetris_prefilter"] = player.last_prefilter
                    stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                if player.last_prefilter and player.last_prefilter["applied"]:
                    prefilter_decisions += 1
                    max_legal_options = max(max_legal_options, player.last_prefilter["legal_count"])
                game.apply(action)
            result = {"type": "game", "game_index": game_number, "game": args.game,
                      "player": args.player, "seed": game_seed, "score": game.score,
                      "steps": game.steps, "capped": not game.done, **final_stats(game)}
            if args.game == "tetris" and args.player == "statim":
                result["prefilter"] = {"name": "fixed weighted feature top-12", "limit": 12,
                                       "decisions_applied": prefilter_decisions,
                                       "max_legal_options": max_legal_options}
            stream.write(json.dumps(result, separators=(",", ":")) + "\n")
            scores.append(game.score)
    if args.summary:
        print_summary(scores, args.seed)


if __name__ == "__main__":
    main()
