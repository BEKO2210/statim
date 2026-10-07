"""Translate game facts into one Statim choice question."""

STRATEGIES = {
    "2048": "Keep the largest tile in a corner; prefer merges and many empty cells.",
    "snake": "Reach food safely while preserving enough open space to avoid trapping the snake.",
    "othello": "Take corners, avoid giving corners, limit opponent mobility, and gain discs.",
    "tetris": "Clear lines while keeping the stack low, even, and free of holes.",
}


def _yes(value):
    return "yes" if value else "no"


def describe(game_name, features):
    """The numeric v1 option description, retained for comparison runs."""
    if game_name == "2048":
        text = (f"merges {features['merges']}; merged value {features['merged_value']}; "
                f"empty after {features['empty']}; corner kept {_yes(features['corner'])}; "
                f"monotonic lines {features['monotonic']}")
    elif game_name == "snake":
        text = (f"food distance {features['food_distance']}; reachable cells {features['reachable']}; "
                f"immediate death {_yes(features['death'])}; eats food {_yes(features['eats'])}")
    elif game_name == "othello":
        text = (f"discs flipped {features['flips']}; corner {_yes(features['corner'])}; "
                f"gives opponent corner {_yes(features['gives_corner'])}; "
                f"opponent moves {features['opp_mobility']}")
    elif game_name == "tetris":
        text = (f"lines cleared {features['lines']}; holes {features['holes']}; "
                f"aggregate height {features['aggregate_height']}; bumpiness {features['bumpiness']}; "
                f"max height {features['max_height']}")
    else:
        raise ValueError(f"unknown game: {game_name}")
    if len(text) > 160:
        raise AssertionError("option text exceeded 160 characters")
    return text


def _plural(number, singular):
    return singular if number == 1 else singular + "s"


def _sentence_2048(label, facts, _game):
    merge = facts["largest_merge"]
    start = (f"Move {label} merges two {merge // 2} tiles into a {merge}" if merge
             else f"Move {label} merges nothing")
    corner = {
        "kept": "keeps the largest tile in the corner",
        "lost": "moves the largest tile out of the corner",
        "gained": "moves the largest tile into a corner",
        "absent": "leaves the largest tile out of the corners",
    }[facts["corner_status"]]
    return f"{start} and {corner}; {facts['empty']} cells stay empty."


def _sentence_snake(label, facts, game):
    if facts["death"]:
        obstacle = "wall" if facts["death_reason"] == "wall" else "snake"
        return f"Move {label} hits the {obstacle} and the snake dies."
    if facts["eats"]:
        return f"Move {label} eats the food."
    if facts["reachable"] < len(game.snake) + 2:
        return f"Move {label} is safe but traps the snake in {facts['reachable']} free cells."
    current = abs(game.snake[0][0] - game.food[0]) + abs(game.snake[0][1] - game.food[1])
    if facts["food_distance"] < current:
        relation = "brings the head closer to"
    elif facts["food_distance"] > current:
        relation = "moves the head farther from"
    else:
        relation = "keeps the head the same distance from"
    return f"Move {label} is safe and {relation} the food ({facts['food_distance']} steps away)."


def _sentence_othello(label, facts, _game):
    if label == "pass":
        return f"Move pass flips no discs and leaves the opponent {facts['opp_mobility']} moves."
    if facts["corner"]:
        return f"Move {label} takes a corner and flips {facts['flips']} {_plural(facts['flips'], 'disc')}."
    if facts["gives_corner"]:
        return (f"Move {label} flips {facts['flips']} {_plural(facts['flips'], 'disc')} "
                "but gives the opponent a corner.")
    return (f"Move {label} flips {facts['flips']} {_plural(facts['flips'], 'disc')} and leaves the opponent "
            f"{facts['opp_mobility']} {_plural(facts['opp_mobility'], 'move')}.")


def _sentence_tetris(label, facts, game, action):
    _piece, rotation, column = game._parse(action)
    lines = ("clears no lines" if not facts["lines"] else
             f"clears {facts['lines']} {_plural(facts['lines'], 'line')}")
    holes = ("leaves no holes" if not facts["holes"] else
             f"leaves {facts['new_holes']} new {_plural(facts['new_holes'], 'hole')}")
    display = label[1:] if label.startswith("p") and label[1:].isdigit() else label
    return (f"Placement {display} (rotation {rotation}, column {column}) {lines} and {holes}; "
            f"the stack is {facts['max_height']} {_plural(facts['max_height'], 'row')} high.")


def _tetris_prefilter_score(features):
    return (8 * features["lines"] - 7 * features["holes"]
            - features["aggregate_height"] - features["bumpiness"] - 2 * features["max_height"])


def decision_options(game, shuffle=False, rng=None):
    actions = game.legal_actions()
    facts = {action: game.features(action) for action in actions}
    prefiltered = False
    total = len(actions)
    if game.name == "tetris" and len(actions) > 12:
        actions = sorted(actions, key=lambda action: (-_tetris_prefilter_score(facts[action]), action))[:12]
        prefiltered = True
    if shuffle:
        if rng is None:
            raise ValueError("shuffle requires a seeded random generator")
        actions = list(actions)
        rng.shuffle(actions)
    labels = ([f"p{index}" for index in range(1, len(actions) + 1)]
              if game.name == "tetris" else actions)
    choices = dict(zip(labels, actions))
    return choices, facts, {"applied": prefiltered, "legal_count": total, "sent_count": len(actions)}


def build_request(game, model=None, prompt="v2", labels=None, shuffle=False, rng=None, sentences_out=None):
    """Build one request; omitted labels retains the original numbered Tetris prompt."""
    if prompt not in ("v1", "v2"):
        raise ValueError(f"unknown prompt version: {prompt}")
    if labels not in (None, "moves", "letters"):
        raise ValueError(f"unknown label style: {labels}")
    choices, facts, prefilter = decision_options(game, shuffle=shuffle, rng=rng)
    label_style = labels or "moves"
    if labels == "moves" and game.name == "tetris":
        choices = {action: action for action in choices.values()}
    elif labels == "letters":
        if len(choices) > 26:
            raise ValueError("letter labels support at most 26 options")
        choices = {chr(ord("A") + index): action
                   for index, action in enumerate(choices.values())}
    if prompt == "v1":
        choices = {action: action for action in choices.values()}
        state = {"game": game.name, "board": game.render_text()}
        criteria = {label: describe(game.name, facts[action]) for label, action in choices.items()}
        instructions = STRATEGIES[game.name]
    else:
        sentences = []
        for label, action in choices.items():
            if game.name == "2048":
                sentence = _sentence_2048(label, facts[action], game)
            elif game.name == "snake":
                sentence = _sentence_snake(label, facts[action], game)
            elif game.name == "othello":
                sentence = _sentence_othello(label, facts[action], game)
            else:
                sentence = _sentence_tetris(label, facts[action], game, action)
            if label_style == "letters":
                if game.name == "tetris":
                    subject = sentence.split(" ", 2)[:2]
                    sentence = sentence.replace(" ".join(subject),
                                                f"Option {label} (placement {action})", 1)
                else:
                    sentence = sentence.replace(f"Move {label}",
                                                f"Option {label} (move {action})", 1)
            if len(sentence) > 140:
                raise AssertionError("move sentence exceeded 140 characters")
            sentences.append(sentence)
            if sentences_out is not None:  # the v2 criteria are null; traces need the text per option
                sentences_out[label] = sentence
        state = f"{game.name.title()} game. " + " ".join(sentences)
        if len(state) > 1800:
            raise AssertionError("state exceeded 1,800 characters")
        criteria = {label: None for label in choices}
        instructions = STRATEGIES[game.name] + " Which move is best?"
    payload = {"state": state, "questions": {"move": {
        "type": "choice", "instructions": instructions, "criteria": criteria}}}
    if model is not None:
        payload["model"] = model
    return payload, choices, prefilter
