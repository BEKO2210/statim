"""Bit-exact parity checks for the browser action-game engines."""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import unittest
import copy
from pathlib import Path

from games import GAME_CLASSES
from players import HeuristicPlayer
from prompting import build_request
from run import final_stats


NODE = shutil.which("node")
RUNNER = Path(__file__).with_name("js_runner.mjs")


def snapshot(game):
    payload, choices, _prefilter = build_request(game, labels="moves")
    board = ({"snake": copy.deepcopy(game.snake), "food": game.food, "direction": game.direction}
             if game.name == "snake" else copy.deepcopy(game.board))
    result = {
        "board": board,
        "score": game.score,
        "done": game.done,
        "legal_actions": game.legal_actions(),
        "request": json.dumps(payload, separators=(",", ":")),
        "choices": choices,
        "heuristic_choice": None if game.done else HeuristicPlayer().choose(game)[0],
        "final_stats": final_stats(game),
    }
    # Compare the JSON data model: Python game coordinates are tuples, while
    # JavaScript (and their serialized representation) necessarily uses arrays.
    return json.loads(json.dumps(result, separators=(",", ":")))


@unittest.skipUnless(NODE, "node is required for browser-engine parity tests")
class JavaScriptParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cases = []
        cls.expected_cases = []
        for game_name, game_class in GAME_CLASSES.items():
            for seed in range(20):
                game = game_class(seed)
                move_rng = random.Random(0x5A17_0000 + seed * 17 + list(GAME_CLASSES).index(game_name))
                moves = []
                expected = []
                for _ in range(60):
                    if game.done:
                        break
                    expected.append(snapshot(game))
                    action = move_rng.choice(game.legal_actions())
                    moves.append(action)
                    game.apply(action)
                expected.append(snapshot(game))
                cases.append({"game": game_name, "seed": seed, "moves": moves})
                cls.expected_cases.append((game_name, seed, expected))

        cls.rng_seeds = [0, 1, -1, 2**65 + 12345]
        rng_specs = [{"seed": str(seed), "count": 1000} for seed in cls.rng_seeds]
        command = json.dumps({"cases": cases, "rng": rng_specs}, separators=(",", ":"))
        completed = subprocess.run(
            [NODE, str(RUNNER)], input=command, text=True, capture_output=True, check=True,
            timeout=120,
        )
        cls.actual = json.loads(completed.stdout)

    def test_games(self):
        self.assertEqual(len(self.expected_cases), len(self.actual["cases"]))
        for (game_name, seed, expected), actual in zip(self.expected_cases, self.actual["cases"]):
            with self.subTest(game=game_name, seed=seed):
                self.assertEqual(expected, actual)

    def test_python_random(self):
        for seed, actual in zip(self.rng_seeds, self.actual["rng"]):
            with self.subTest(seed=seed):
                random_rng = random.Random(seed)
                expected_random = [random_rng.random() for _ in range(1000)]
                result_rng = random.Random(seed)
                expected_results = []
                choices = ["a", "b", "c", "d", "e", "f", "g"]
                for index in range(1000):
                    if index % 3 == 0:
                        expected_results.append(result_rng.randrange(1, 1000003, 7))
                    elif index % 3 == 1:
                        expected_results.append(result_rng.randrange(-5000, 8000))
                    else:
                        expected_results.append(result_rng.choice(choices))
                self.assertEqual(expected_random, actual["random"])
                self.assertEqual(expected_results, actual["results"])


if __name__ == "__main__":
    unittest.main()
