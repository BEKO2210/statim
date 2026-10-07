"""Statim and local comparison players."""

from __future__ import annotations

import json
import random
import urllib.request

from prompting import build_request


class StatimPlayer:
    def __init__(self, base_url, model=None, timeout=120, prompt="v2", adapter=None,
                 labels="moves", shuffle=False, seed=0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.prompt = prompt
        self.adapter = adapter
        self.labels = labels
        self.shuffle = shuffle
        self.rng = random.Random(seed)
        self.last_options = None
        self.last_prefilter = None

    def choose(self, game):
        payload, choices, prefilter = build_request(
            game, self.model, self.prompt, self.labels, self.shuffle, self.rng)
        if self.adapter is not None:
            payload["adapter"] = self.adapter
        self.last_options = payload["questions"]["move"]["criteria"]
        self.last_prefilter = prefilter
        request = urllib.request.Request(
            self.base_url + "/v1/systemone",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.load(response)
        answer = body["answers"]["move"]
        choice = answer["choice"]
        probabilities = {str(key): float(value) for key, value in answer["probabilities"].items()}
        if choice not in choices:
            raise ValueError(f"Statim returned unknown action: {choice}")
        return choices[choice], {choices[label]: value for label, value in probabilities.items()}


class RandomPlayer:
    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.last_options = self.last_prefilter = None

    def choose(self, game):
        actions = game.legal_actions()
        choice = self.rng.choice(actions)
        return choice, {action: 1 / len(actions) for action in actions}


class FirstFeaturePlayer:
    def __init__(self):
        self.last_options = self.last_prefilter = None

    def choose(self, game):
        actions = game.legal_actions()
        facts = {action: game.features(action) for action in actions}
        if game.name == "2048":
            key = lambda action: (-facts[action]["merges"], action)
        elif game.name == "snake":
            safe = [action for action in actions if not facts[action]["death"]]
            pool = safe or actions
            choice = min(pool, key=lambda action: (facts[action]["food_distance"], action))
            return choice, {action: float(action == choice) for action in actions}
        elif game.name == "othello":
            key = lambda action: (-facts[action]["flips"], action)
        else:
            key = lambda action: (facts[action]["holes"], action)
        choice = min(actions, key=key)
        return choice, {action: float(action == choice) for action in actions}


class HeuristicPlayer:
    def __init__(self):
        self.last_options = self.last_prefilter = None

    @staticmethod
    def _score(name, features):
        if name == "2048":
            return (12 * features["merges"] + .25 * features["merged_value"]
                    + 3 * features["empty"] + 15 * features["corner"] + 2 * features["monotonic"])
        if name == "snake":
            return (-10000 * features["death"] + 50 * features["eats"]
                    + features["reachable"] - 2 * features["food_distance"])
        if name == "othello":
            return (100 * features["corner"] - 100 * features["gives_corner"]
                    + features.get("positional", 0) + 2 * features["flips"]
                    - 4 * features["opp_mobility"])
        return (10 * features["lines"] - 8 * features["holes"]
                - .5 * features["aggregate_height"] - .8 * features["bumpiness"]
                - features["max_height"])

    def choose(self, game):
        scores = self.scores(game)
        actions = list(scores)
        choice = min(actions, key=lambda action: (-scores[action], action))
        return choice, {action: float(action == choice) for action in actions}

    def scores(self, game):
        return {action: self._score(game.name, game.features(action))
                for action in game.legal_actions()}
