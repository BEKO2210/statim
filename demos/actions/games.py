"""Small deterministic game engines used by the action-choice demo."""

from __future__ import annotations

from collections import deque
import copy
import random


class Game2048:
    name = "2048"
    size = 4
    directions = ("left", "right", "up", "down")

    def __init__(self, seed=0, board=None):
        self.rng = random.Random(seed)
        self.board = copy.deepcopy(board) if board is not None else [[0] * 4 for _ in range(4)]
        self.steps = 0
        self.points = 0
        if board is None:
            self._spawn()
            self._spawn()
        self.done = not bool(self.legal_actions())

    @staticmethod
    def _merge_line(line):
        values = [value for value in line if value]
        out, merges, merged_value = [], 0, 0
        i = 0
        while i < len(values):
            if i + 1 < len(values) and values[i] == values[i + 1]:
                value = values[i] * 2
                out.append(value)
                merges += 1
                merged_value += value
                i += 2
            else:
                out.append(values[i])
                i += 1
        return out + [0] * (4 - len(out)), merges, merged_value

    def _moved(self, action):
        if action not in self.directions:
            raise ValueError(f"unknown 2048 action: {action}")
        result = [[0] * 4 for _ in range(4)]
        merges = merged_value = 0
        for index in range(4):
            if action in ("left", "right"):
                line = list(self.board[index])
            else:
                line = [self.board[row][index] for row in range(4)]
            if action in ("right", "down"):
                line.reverse()
            line, made, value = self._merge_line(line)
            if action in ("right", "down"):
                line.reverse()
            merges += made
            merged_value += value
            for offset, cell in enumerate(line):
                if action in ("left", "right"):
                    result[index][offset] = cell
                else:
                    result[offset][index] = cell
        return result, merges, merged_value

    def _spawn(self):
        empty = [(r, c) for r in range(4) for c in range(4) if self.board[r][c] == 0]
        rank_draw, value_draw = self.rng.random(), self.rng.random()
        if empty:
            r, c = empty[min(int(rank_draw * len(empty)), len(empty) - 1)]
            self.board[r][c] = 2 if value_draw < 0.9 else 4

    def legal_actions(self):
        return [action for action in self.directions if self._moved(action)[0] != self.board]

    @staticmethod
    def _monotonic_count(board):
        count = 0
        lines = list(board) + [[board[r][c] for r in range(4)] for c in range(4)]
        for line in lines:
            nonzero = [value for value in line if value]
            if nonzero == sorted(nonzero) or nonzero == sorted(nonzero, reverse=True):
                count += 1
        return count

    def features(self, action):
        board, merges, merged_value = self._moved(action)
        current_largest = max(max(row) for row in self.board)
        current_corners = (self.board[0][0], self.board[0][3], self.board[3][0], self.board[3][3])
        largest = max(max(row) for row in board)
        corners = (board[0][0], board[0][3], board[3][0], board[3][3])
        was_corner = current_largest in current_corners
        is_corner = largest in corners
        merge_values = []
        for index in range(4):
            line = (list(self.board[index]) if action in ("left", "right") else
                    [self.board[row][index] for row in range(4)])
            if action in ("right", "down"):
                line.reverse()
            values = [value for value in line if value]
            offset = 0
            while offset + 1 < len(values):
                if values[offset] == values[offset + 1]:
                    merge_values.append(values[offset] * 2)
                    offset += 2
                else:
                    offset += 1
        return {
            "merges": merges,
            "merged_value": merged_value,
            "largest_merge": max(merge_values, default=0),
            "empty": sum(cell == 0 for row in board for cell in row),
            "corner": is_corner,
            "corner_status": ("kept" if was_corner and is_corner else
                              "lost" if was_corner else "gained" if is_corner else "absent"),
            "monotonic": self._monotonic_count(board),
        }

    def apply(self, action):
        if action not in self.legal_actions():
            raise ValueError(f"illegal 2048 action: {action}")
        self.board, _, merged_value = self._moved(action)
        self.points += merged_value
        self._spawn()
        self.steps += 1
        self.done = not bool(self.legal_actions())

    @property
    def score(self):
        return self.points

    def render_text(self):
        return "/".join(" ".join("." if cell == 0 else str(cell) for cell in row) for row in self.board)


class Snake:
    name = "snake"
    width = height = 8
    vectors = {"up": (-1, 0), "right": (0, 1), "down": (1, 0), "left": (0, -1)}
    opposites = {"up": "down", "down": "up", "left": "right", "right": "left"}

    def __init__(self, seed=0, snake=None, food=None, direction="right"):
        self.rng = random.Random(seed)
        self.snake = list(snake) if snake is not None else [(4, 4), (4, 3), (4, 2)]
        self.direction = direction
        self.food = food
        self.steps = 0
        self.foods = 0
        self.done = False
        self.won = False
        if self.food is None:
            self._place_food()

    def _place_food(self):
        free = [(r, c) for r in range(8) for c in range(8) if (r, c) not in self.snake]
        rank_draw, _unused = self.rng.random(), self.rng.random()
        self.food = free[min(int(rank_draw * len(free)), len(free) - 1)] if free else None
        if not free:
            self.done = True
            self.won = True

    def legal_actions(self):
        if self.done:
            return []
        return [action for action in self.vectors if action != self.opposites[self.direction]]

    def _next(self, action):
        dr, dc = self.vectors[action]
        head = self.snake[0]
        new_head = (head[0] + dr, head[1] + dc)
        grows = new_head == self.food
        occupied = set(self.snake if grows else self.snake[:-1])
        death = not (0 <= new_head[0] < 8 and 0 <= new_head[1] < 8) or new_head in occupied
        body = [new_head] + self.snake if grows else [new_head] + self.snake[:-1]
        return new_head, body, grows, death

    @staticmethod
    def _reachable(head, blocked):
        if not (0 <= head[0] < 8 and 0 <= head[1] < 8) or head in blocked:
            return 0
        seen = {head}
        queue = deque([head])
        while queue:
            r, c = queue.popleft()
            for dr, dc in Snake.vectors.values():
                nxt = (r + dr, c + dc)
                if 0 <= nxt[0] < 8 and 0 <= nxt[1] < 8 and nxt not in blocked and nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        return len(seen)

    def features(self, action):
        if action not in self.legal_actions():
            raise ValueError(f"illegal snake action: {action}")
        head, body, grows, death = self._next(action)
        distance = 0 if self.food is None else abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])
        blocked = set(body[1:]) if not death else set()
        return {
            "food_distance": distance,
            "reachable": 0 if death else self._reachable(head, blocked),
            "death": death,
            "death_reason": ("wall" if death and not (0 <= head[0] < 8 and 0 <= head[1] < 8)
                             else "snake" if death else None),
            "eats": grows and not death,
        }

    def apply(self, action):
        if action not in self.legal_actions():
            raise ValueError(f"illegal snake action: {action}")
        head, body, grows, death = self._next(action)
        self.direction = action
        self.steps += 1
        if death:
            self.done = True
            return
        self.snake = body
        if grows:
            self.foods += 1
            self._place_food()

    @property
    def score(self):
        return self.foods

    def render_text(self):
        cells = [["."] * 8 for _ in range(8)]
        if self.food is not None:
            cells[self.food[0]][self.food[1]] = "F"
        for r, c in self.snake[1:]:
            cells[r][c] = "o"
        r, c = self.snake[0]
        cells[r][c] = "H"
        return "/".join("".join(row) for row in cells)


class Othello:
    name = "othello"
    directions = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0))
    # A common positional table: corners dominate and X/C squares are dangerous.
    weights = (
        120, -20, 20, 5, 5, 20, -20, 120,
        -20, -40, -5, -5, -5, -5, -40, -20,
        20, -5, 15, 3, 3, 15, -5, 20,
        5, -5, 3, 3, 3, 3, -5, 5,
        5, -5, 3, 3, 3, 3, -5, 5,
        20, -5, 15, 3, 3, 15, -5, 20,
        -20, -40, -5, -5, -5, -5, -40, -20,
        120, -20, 20, 5, 5, 20, -20, 120,
    )

    def __init__(self, seed=0, board=None):
        self.board = list(board) if board is not None else [0] * 64
        if board is None:
            self.board[3 * 8 + 3] = self.board[4 * 8 + 4] = -1
            self.board[3 * 8 + 4] = self.board[4 * 8 + 3] = 1
        self.steps = 0
        self.opening_moves = []
        if board is None:
            rng = random.Random(seed)
            player = 1
            for _ in range(4):
                moves = self._moves(player)
                if moves:
                    move = rng.choice(moves)
                    self._put(move, player)
                    self.opening_moves.append(self._label(move))
                else:
                    self.opening_moves.append("pass")
                player = -player
        self.done = not self._moves(1) and not self._moves(-1)

    @staticmethod
    def _label(index):
        return chr(ord("a") + index % 8) + str(index // 8 + 1)

    @staticmethod
    def _index(label):
        if len(label) != 2 or label[0] not in "abcdefgh" or label[1] not in "12345678":
            raise ValueError(f"bad Othello square: {label}")
        return (int(label[1]) - 1) * 8 + ord(label[0]) - ord("a")

    def _flips(self, index, player):
        if self.board[index] != 0:
            return []
        row, col = divmod(index, 8)
        result = []
        for dr, dc in self.directions:
            line = []
            r, c = row + dr, col + dc
            while 0 <= r < 8 and 0 <= c < 8 and self.board[r * 8 + c] == -player:
                line.append(r * 8 + c)
                r, c = r + dr, c + dc
            if line and 0 <= r < 8 and 0 <= c < 8 and self.board[r * 8 + c] == player:
                result.extend(line)
        return result

    def _moves(self, player):
        return [index for index in range(64) if self._flips(index, player)]

    def legal_actions(self):
        if self.done:
            return []
        moves = self._moves(1)
        return [self._label(index) for index in moves] if moves else ["pass"]

    def _put(self, index, player):
        flips = self._flips(index, player)
        if not flips:
            raise ValueError("illegal Othello move")
        self.board[index] = player
        for flipped in flips:
            self.board[flipped] = player
        return len(flips)

    def features(self, action):
        if action == "pass":
            if self._moves(1):
                raise ValueError("illegal Othello pass")
            return {"flips": 0, "corner": False, "gives_corner": False, "opp_mobility": len(self._moves(-1))}
        index = self._index(action)
        board = self.board[:]
        flips = self._flips(index, 1)
        if not flips:
            raise ValueError(f"illegal Othello action: {action}")
        self._put(index, 1)
        opponent_moves = self._moves(-1)
        result = {
            "flips": len(flips),
            "corner": index in (0, 7, 56, 63),
            "gives_corner": any(move in (0, 7, 56, 63) for move in opponent_moves),
            "opp_mobility": len(opponent_moves),
            "positional": self.weights[index],
        }
        self.board = board
        return result

    def apply(self, action):
        if action not in self.legal_actions():
            raise ValueError(f"illegal Othello action: {action}")
        if action != "pass":
            self._put(self._index(action), 1)
        self.steps += 1
        opponent = self._moves(-1)
        if opponent:
            choice = min(opponent, key=lambda index: (-self.weights[index], index))
            self._put(choice, -1)
        self.done = not self._moves(1) and not self._moves(-1)

    @property
    def score(self):
        return self.board.count(1) - self.board.count(-1)

    def render_text(self):
        symbols = {0: ".", 1: "B", -1: "W"}
        return "/".join("".join(symbols[self.board[r * 8 + c]] for c in range(8)) for r in range(8))


# Bottom-left anchored rotation states. Final-placement play does not need wall kicks.
TETROMINOES = {
    "I": (((0, 0), (1, 0), (2, 0), (3, 0)), ((0, 0), (0, 1), (0, 2), (0, 3))),
    "O": (((0, 0), (1, 0), (0, 1), (1, 1)),),
    "T": (((0, 0), (1, 0), (2, 0), (1, 1)), ((0, 0), (0, 1), (0, 2), (1, 1)),
          ((1, 0), (0, 1), (1, 1), (2, 1)), ((1, 0), (0, 1), (1, 1), (1, 2))),
    "S": (((0, 0), (1, 0), (1, 1), (2, 1)), ((1, 0), (0, 1), (1, 1), (0, 2))),
    "Z": (((1, 0), (2, 0), (0, 1), (1, 1)), ((0, 0), (0, 1), (1, 1), (1, 2))),
    "J": (((0, 0), (1, 0), (2, 0), (0, 1)), ((0, 0), (1, 0), (1, 1), (1, 2)),
          ((2, 0), (0, 1), (1, 1), (2, 1)), ((0, 0), (0, 1), (0, 2), (1, 2))),
    "L": (((0, 0), (1, 0), (2, 0), (2, 1)), ((0, 0), (0, 1), (0, 2), (1, 0)),
          ((0, 0), (0, 1), (1, 1), (2, 1)), ((0, 2), (1, 0), (1, 1), (1, 2))),
}


class Tetris:
    name = "tetris"
    width, height = 10, 20

    def __init__(self, seed=0, board=None, current=None):
        self.rng = random.Random(seed)
        self.board = [list(row) for row in board] if board is not None else [[0] * 10 for _ in range(20)]
        self.bag = []
        self.steps = 0
        self.lines = 0
        self.current = current or self._draw()
        self.done = not self._can_spawn(self.current)

    def _draw(self):
        if not self.bag:
            self.bag = list(TETROMINOES)
            self.rng.shuffle(self.bag)
        return self.bag.pop()

    @staticmethod
    def _bounds(shape):
        return max(x for x, _ in shape) + 1, max(y for _, y in shape) + 1

    def _can_spawn(self, piece):
        shape = TETROMINOES[piece][0]
        width, shape_height = self._bounds(shape)
        x = (self.width - width) // 2
        y = self.height - shape_height
        return all(not self.board[y + dy][x + dx] for dx, dy in shape)

    @staticmethod
    def _label(piece, rotation, column):
        return f"{piece}:r{rotation}:x{column}"

    @staticmethod
    def _parse(action):
        piece, rotation, column = action.split(":")
        return piece, int(rotation[1:]), int(column[1:])

    def _drop_y(self, shape, column):
        width, shape_height = self._bounds(shape)
        if column < 0 or column + width > self.width:
            return None
        y = self.height - shape_height
        if any(self.board[y + dy][column + dx] for dx, dy in shape):
            return None
        while y > 0 and all(not self.board[y - 1 + dy][column + dx] for dx, dy in shape):
            y -= 1
        return y

    def legal_actions(self):
        if self.done:
            return []
        result = []
        for rotation, shape in enumerate(TETROMINOES[self.current]):
            width, _ = self._bounds(shape)
            for column in range(self.width - width + 1):
                if self._drop_y(shape, column) is not None:
                    result.append(self._label(self.current, rotation, column))
        return result

    def _placed(self, action):
        piece, rotation, column = self._parse(action)
        if piece != self.current or not 0 <= rotation < len(TETROMINOES[piece]):
            raise ValueError(f"illegal Tetris action: {action}")
        shape = TETROMINOES[piece][rotation]
        y = self._drop_y(shape, column)
        if y is None:
            raise ValueError(f"illegal Tetris action: {action}")
        board = [row[:] for row in self.board]
        for dx, dy in shape:
            board[y + dy][column + dx] = 1
        kept = [row for row in board if not all(row)]
        cleared = self.height - len(kept)
        board = kept + [[0] * self.width for _ in range(cleared)]
        return board, cleared

    @staticmethod
    def _board_features(board):
        heights, holes = [], 0
        for column in range(10):
            occupied = [row for row in range(20) if board[row][column]]
            height = max(occupied) + 1 if occupied else 0
            heights.append(height)
            holes += sum(not board[row][column] for row in range(height))
        return {
            "holes": holes,
            "aggregate_height": sum(heights),
            "bumpiness": sum(abs(a - b) for a, b in zip(heights, heights[1:])),
            "max_height": max(heights),
        }

    def features(self, action):
        board, cleared = self._placed(action)
        before_holes = self._board_features(self.board)["holes"]
        result = self._board_features(board)
        return {"lines": cleared, "new_holes": max(0, result["holes"] - before_holes), **result}

    def apply(self, action):
        if action not in self.legal_actions():
            raise ValueError(f"illegal Tetris action: {action}")
        self.board, cleared = self._placed(action)
        self.lines += cleared
        self.steps += 1
        self.current = self._draw()
        self.done = not self._can_spawn(self.current)

    @property
    def score(self):
        return self.lines

    def render_text(self):
        rows = ["".join("#" if cell else "." for cell in row) for row in reversed(self.board)]
        return f"piece={self.current};" + "/".join(rows)


GAME_CLASSES = {"2048": Game2048, "snake": Snake, "othello": Othello, "tetris": Tetris}
