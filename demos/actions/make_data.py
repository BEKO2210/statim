#!/usr/bin/env python3
"""Generate action-choice training rows from the deterministic heuristic teacher."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import statistics
from pathlib import Path

from games import GAME_CLASSES
from players import HeuristicPlayer
from prompting import build_request


TARGET_TOP = 0.8
FIT_SAMPLE_ROWS = 4096


def _probabilities(scores, temperature):
    peak = max(scores)
    weights = [math.exp((score - peak) / temperature) for score in scores]
    total = sum(weights)
    return [weight / total for weight in weights]


def _mean_teacher_probability(examples, temperature):
    return statistics.fmean(
        sum(probability for score, probability in zip(scores, _probabilities(scores, temperature))
            if score == max(scores))
        for scores, _teacher_index in examples)


def fit_temperature(examples, target=TARGET_TOP):
    """Fit top-score probability mass by log-space bisection; ties split that mass."""
    if not examples:
        raise ValueError("cannot fit a temperature without examples")
    low, high = 1e-6, 1e6
    low_probability = _mean_teacher_probability(examples, low)
    high_probability = _mean_teacher_probability(examples, high)
    if target >= low_probability:
        return low
    if target <= high_probability:
        return high
    for _ in range(80):
        middle = math.sqrt(low * high)
        if _mean_teacher_probability(examples, middle) > target:
            low = middle
        else:
            high = middle
    return math.sqrt(low * high)


def _pending_row(game, teacher, rng):
    scores = teacher.scores(game)
    teacher_action = min(scores, key=lambda action: (-scores[action], action))
    labels = "letters" if rng.random() < 0.5 else "moves"
    payload, choices, prefilter = build_request(
        game, prompt="v2", labels=labels, shuffle=True, rng=rng)
    if teacher_action not in choices.values():
        if game.name != "tetris" or not prefilter["applied"]:
            raise AssertionError("a teacher action disappeared without the Tetris prefilter")
        return None, teacher_action
    ordered_scores = [scores[action] for action in choices.values()]
    teacher_index = list(choices.values()).index(teacher_action)
    row = {
        "state": payload["state"],
        "q": payload["questions"]["move"],
        "src": f"actions/{game.name}",
        "lang": "en",
    }
    return (row, ordered_scores, teacher_index), teacher_action


def _write_row(stream, pending, temperature):
    row, scores, _teacher_index = pending
    target = _probabilities(scores, temperature)
    row["target"] = target
    stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    peak = max(scores)
    return sum(probability for score, probability in zip(scores, target) if score == peak)


def generate_kind(name, args, rng, stream):
    teacher = HeuristicPlayer()
    pending = []
    fit_examples = []
    temperature = None
    top_probability_sum = 0.0
    option_count = rows = dropped = capped = games_played = 0

    def flush_pending():
        nonlocal pending, temperature, top_probability_sum
        if temperature is None:
            temperature = fit_temperature(fit_examples)
        for item in pending:
            top_probability_sum += _write_row(stream, item, temperature)
        pending = []

    for game_number in range(args.games_per_game):
        games_played += 1
        seed = args.seed_base + game_number
        assert seed >= 100000, "game seeds 0--99999 are reserved for evaluation"
        game = GAME_CLASSES[name](seed)
        while (not game.done and game.steps < args.max_steps
               and rows < args.max_rows_per_game):
            item, teacher_action = _pending_row(game, teacher, rng)
            if item is None:
                dropped += 1
            else:
                _row, scores, teacher_index = item
                rows += 1
                option_count += len(scores)
                if len(fit_examples) < FIT_SAMPLE_ROWS:
                    fit_examples.append((scores, teacher_index))
                    pending.append(item)
                    if len(fit_examples) == FIT_SAMPLE_ROWS:
                        flush_pending()
                else:
                    top_probability_sum += _write_row(stream, item, temperature)
            played = rng.choice(game.legal_actions()) if rng.random() < args.explore else teacher_action
            game.apply(played)
        if not game.done and game.steps >= args.max_steps:
            capped += 1
        if rows >= args.max_rows_per_game:
            break
    if pending:
        flush_pending()
    if temperature is None:
        raise SystemExit(f"no usable rows generated for {name}")
    return {
        "rows": rows,
        "mean_options": option_count / rows,
        "teacher_top_probability": top_probability_sum / rows,
        "dropped_rows": dropped,
        "capped_games": capped,
        "games_played": games_played,
        "row_cap_reached": rows >= args.max_rows_per_game,
        "temperature": temperature,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games-per-game", type=int, default=400)
    parser.add_argument("--seed-base", type=int, default=100000)
    parser.add_argument("--out", required=True)
    parser.add_argument("--explore", type=float, default=0.15)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--max-rows-per-game", type=int, default=60000)
    args = parser.parse_args(argv)
    if args.games_per_game < 1 or args.max_steps < 1 or args.max_rows_per_game < 1:
        parser.error("game and row counts must be positive")
    if not 0 <= args.explore <= 1:
        parser.error("--explore must be between zero and one")
    if args.seed_base < 100000:
        parser.error("--seed-base must be at least 100000; seeds 0--99999 are reserved for evaluation")
    return args


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed_base)
    summary = {}
    with gzip.open(output, "wt", encoding="utf-8") as stream:
        for name in GAME_CLASSES:
            summary[name] = generate_kind(name, args, rng, stream)
            print(f"fitted temperature {name}: {summary[name]['temperature']:.9g}", flush=True)
    print(json.dumps({"out": str(output), "games_per_game": args.games_per_game,
                      "seed_base": args.seed_base, "games": summary}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
