"""Rule, feature, determinism, and client tests for the action demo."""

import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from games import Game2048, Othello, Snake, Tetris
from players import FirstFeaturePlayer, HeuristicPlayer, RandomPlayer, StatimPlayer
from prompting import build_request
from run import bootstrap_interval, final_stats, main


class Game2048Tests(unittest.TestCase):
    def test_merge_order(self):
        game = Game2048(board=[[2, 2, 2, 2], [4, 0, 4, 8], [0] * 4, [0] * 4])
        board, merges, value = game._moved("left")
        self.assertEqual(board[0], [4, 4, 0, 0])
        self.assertEqual(board[1], [8, 8, 0, 0])
        self.assertEqual((merges, value), (3, 16))

    def test_features(self):
        game = Game2048(board=[[2, 2, 0, 0], [0] * 4, [0] * 4, [0] * 4])
        features = game.features("left")
        self.assertEqual(features["merges"], 1)
        self.assertEqual(features["merged_value"], 4)
        self.assertEqual(features["empty"], 15)
        self.assertTrue(features["corner"])
        game.apply("left")
        self.assertEqual(game.score, 4)

    def test_game_over_requires_no_change(self):
        board = [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]]
        self.assertTrue(Game2048(board=board).done)

    def test_spawn_draws_are_event_indexed(self):
        a = Game2048(seed=9, board=[[0] * 4 for _ in range(4)])
        b = Game2048(seed=9, board=[[2, 0, 0, 0]] + [[0] * 4 for _ in range(3)])
        a._spawn()
        b._spawn()
        self.assertEqual(a.rng.random(), b.rng.random())


class SnakeTests(unittest.TestCase):
    def test_growth(self):
        game = Snake(snake=[(4, 4), (4, 3), (4, 2)], food=(4, 5))
        game.apply("right")
        self.assertEqual(len(game.snake), 4)
        self.assertEqual(game.score, 1)

    def test_wall_and_self_death(self):
        wall = Snake(snake=[(0, 2), (0, 1), (0, 0)], food=(7, 7), direction="right")
        self.assertTrue(wall.features("up")["death"])
        wall.apply("up")
        self.assertTrue(wall.done)
        body = Snake(snake=[(2, 2), (2, 1), (1, 1), (1, 2), (1, 3), (2, 3)],
                     food=(7, 7), direction="down")
        self.assertTrue(body.features("left")["death"])

    def test_feature_distance_and_reachability(self):
        game = Snake(snake=[(4, 4), (4, 3), (4, 2)], food=(2, 4), direction="right")
        features = game.features("up")
        self.assertEqual(features["food_distance"], 1)
        self.assertFalse(features["death"])
        self.assertEqual(features["reachable"], 62)

    def test_moving_into_vacated_tail_is_safe(self):
        game = Snake(snake=[(2, 2), (2, 1), (1, 1), (1, 2)], food=(7, 7), direction="right")
        self.assertFalse(game.features("up")["death"])
        game.apply("up")
        self.assertEqual(game.snake[0], (1, 2))

    def test_filling_board_is_a_win(self):
        snake = [(0, 0)] + [(r, c) for r in range(8) for c in range(8) if (r, c) not in ((0, 0), (0, 1))]
        game = Snake(snake=snake, food=(0, 1), direction="down")
        game.apply("right")
        self.assertTrue(game.done)
        self.assertTrue(game.won)
        self.assertTrue(final_stats(game)["won"])


class OthelloTests(unittest.TestCase):
    def test_initial_legal_moves(self):
        board = [0] * 64
        board[3 * 8 + 3] = board[4 * 8 + 4] = -1
        board[3 * 8 + 4] = board[4 * 8 + 3] = 1
        self.assertEqual(Othello(board=board).legal_actions(), ["d3", "c4", "f5", "e6"])

    def test_seeded_four_ply_opening(self):
        self.assertEqual(len(Othello(3).opening_moves), 4)
        self.assertEqual(Othello(3).board, Othello(3).board)
        self.assertNotEqual(Othello(3).board, Othello(4).board)

    def test_flips_all_eight_directions(self):
        board = [0] * 64
        center = (3, 3)
        for dr, dc in Othello.directions:
            board[(center[0] + dr) * 8 + center[1] + dc] = -1
            board[(center[0] + 2 * dr) * 8 + center[1] + 2 * dc] = 1
        game = Othello(board=board)
        self.assertEqual(game.features("d4")["flips"], 8)
        game._put(27, 1)
        self.assertEqual(game.board.count(-1), 0)

    def test_pass_rule(self):
        board = [-1] * 64
        board[1] = 1
        board[2] = 0
        game = Othello(board=board)
        self.assertEqual(game.legal_actions(), ["pass"])
        game.apply("pass")
        self.assertTrue(game.done)
        self.assertEqual(game.board[2], -1)

    def test_corner_and_opponent_mobility_features(self):
        board = [0] * 64
        board[1], board[2] = -1, 1
        game = Othello(board=board)
        features = game.features("a1")
        self.assertTrue(features["corner"])
        self.assertEqual(features["flips"], 1)
        self.assertEqual(features["opp_mobility"], 0)

    def test_heuristic_beats_random_over_twenty_seeds(self):
        def mean_score(factory):
            scores = []
            for seed in range(20):
                game, player = Othello(seed), factory(seed)
                while not game.done:
                    action, _ = player.choose(game)
                    game.apply(action)
                scores.append(game.score)
            return sum(scores) / len(scores)
        self.assertGreater(mean_score(lambda _seed: HeuristicPlayer()),
                           mean_score(lambda seed: RandomPlayer(seed)))


class TetrisTests(unittest.TestCase):
    def test_rotation_and_columns(self):
        game = Tetris(current="I")
        actions = game.legal_actions()
        self.assertIn("I:r0:x6", actions)
        self.assertIn("I:r1:x9", actions)
        self.assertNotIn("I:r0:x7", actions)

    def test_collision_excludes_placement(self):
        board = [[0] * 10 for _ in range(20)]
        board[19][4] = 1
        game = Tetris(board=board, current="I")
        self.assertTrue(game.done)  # The horizontal I cannot occupy its spawn cells.

    def test_line_clear_and_features(self):
        board = [[0] * 10 for _ in range(20)]
        board[0][:8] = [1] * 8
        game = Tetris(board=board, current="O")
        features = game.features("O:r0:x8")
        self.assertEqual(features["lines"], 1)
        self.assertEqual(features["aggregate_height"], 2)
        game.apply("O:r0:x8")
        self.assertEqual(game.lines, 1)

    def test_hole_feature(self):
        board = [[0] * 10 for _ in range(20)]
        board[1][0] = 1
        game = Tetris(board=board, current="O")
        action = "O:r0:x1"
        self.assertEqual(game.features(action)["holes"], 1)


class DeterminismTests(unittest.TestCase):
    def _trajectory(self, cls, seed, turns=12):
        game = cls(seed)
        player = FirstFeaturePlayer()
        states = []
        while not game.done and len(states) < turns:
            action, _ = player.choose(game)
            states.append((game.render_text(), action))
            game.apply(action)
        return states, game.render_text(), game.score

    def test_seeded_engines_repeat(self):
        for cls in (Game2048, Snake, Othello, Tetris):
            with self.subTest(game=cls.__name__):
                self.assertEqual(self._trajectory(cls, 17), self._trajectory(cls, 17))


class PromptingTests(unittest.TestCase):
    def test_request_shape_and_tetris_limit(self):
        game = Tetris(seed=4, current="T")
        payload, actions, info = build_request(game, model="english")
        self.assertTrue(payload["state"].startswith("Tetris game. "))
        self.assertEqual(payload["questions"]["move"]["type"], "choice")
        self.assertEqual(payload["model"], "english")
        self.assertLessEqual(len(actions), 12)
        self.assertTrue(info["applied"])
        self.assertEqual(list(actions), [f"p{i}" for i in range(1, len(actions) + 1)])
        self.assertTrue(all(text is None for text in payload["questions"]["move"]["criteria"].values()))
        self.assertLessEqual(len(payload["state"]), 1800)

    def test_exact_2048_v2_sentences(self):
        game = Game2048(board=[[64, 64, 0, 0], [0] * 4, [0] * 4, [0] * 4])
        payload, _actions, _info = build_request(game)
        self.assertEqual(payload["state"],
            "2048 game. Move left merges two 64 tiles into a 128 and keeps the largest tile in the corner; 15 cells stay empty. "
            "Move right merges two 64 tiles into a 128 and keeps the largest tile in the corner; 15 cells stay empty. "
            "Move down merges nothing and keeps the largest tile in the corner; 14 cells stay empty.")

    def test_exact_snake_v2_sentences(self):
        game = Snake(snake=[(0, 2), (0, 1), (0, 0)], food=(0, 3), direction="right")
        payload, _actions, _info = build_request(game)
        self.assertEqual(payload["state"],
            "Snake game. Move up hits the wall and the snake dies. Move right eats the food. "
            "Move down is safe and moves the head farther from the food (2 steps away).")

    def test_exact_othello_v2_sentence(self):
        board = [0] * 64
        board[1], board[2] = -1, 1
        payload, _actions, _info = build_request(Othello(board=board))
        self.assertEqual(payload["state"], "Othello game. Move a1 takes a corner and flips 1 disc.")

    def test_exact_tetris_v2_sentences(self):
        board = [[0] * 10 for _ in range(20)]
        board[0][:8] = [1] * 8
        payload, actions, _info = build_request(Tetris(board=board, current="O"))
        self.assertEqual(actions["p9"], "O:r0:x8")
        self.assertEqual(payload["state"],
            "Tetris game. Placement 1 (rotation 0, column 0) clears no lines and leaves no holes; the stack is 3 rows high. "
            "Placement 2 (rotation 0, column 1) clears no lines and leaves no holes; the stack is 3 rows high. "
            "Placement 3 (rotation 0, column 2) clears no lines and leaves no holes; the stack is 3 rows high. "
            "Placement 4 (rotation 0, column 3) clears no lines and leaves no holes; the stack is 3 rows high. "
            "Placement 5 (rotation 0, column 4) clears no lines and leaves no holes; the stack is 3 rows high. "
            "Placement 6 (rotation 0, column 5) clears no lines and leaves no holes; the stack is 3 rows high. "
            "Placement 7 (rotation 0, column 6) clears no lines and leaves no holes; the stack is 3 rows high. "
            "Placement 8 (rotation 0, column 7) clears no lines and leaves 1 new hole; the stack is 3 rows high. "
            "Placement 9 (rotation 0, column 8) clears 1 line and leaves no holes; the stack is 1 row high.")

    def test_v1_is_retained(self):
        payload, actions, _info = build_request(
            Game2048(board=[[2, 2, 0, 0], [0] * 4, [0] * 4, [0] * 4]), prompt="v1")
        self.assertEqual(payload["state"]["game"], "2048")
        self.assertIsInstance(payload["questions"]["move"]["criteria"][next(iter(actions))], str)

    def test_bootstrap_uses_rounded_percentile_index(self):
        self.assertEqual(bootstrap_interval([7], seed=1, samples=2), [7.0, 7.0])

    def test_zero_games_is_rejected(self):
        with self.assertRaises(SystemExit):
            main(["--game", "snake", "--player", "random", "--games", "0", "--out", os.devnull])


class _FakeHandler(BaseHTTPRequestHandler):
    request_body = None

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        type(self).request_body = json.loads(self.rfile.read(length))
        options = list(type(self).request_body["questions"]["move"]["criteria"])
        body = json.dumps({"answers": {"move": {"type": "choice", "choice": options[0],
                                                   "probabilities": {option: 1 / len(options) for option in options}}}})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body.encode())))
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, _format, *_args):
        pass


class StatimPlayerTests(unittest.TestCase):
    def test_v2_request_exposes_one_sentence_per_option(self):
        game = Game2048(board=[[2, 2, 0, 0], [0] * 4, [0] * 4, [0] * 4])
        sentences = {}
        payload, choices, _ = build_request(game, labels="moves", sentences_out=sentences)
        self.assertEqual(set(sentences), set(choices))
        for text in sentences.values():
            self.assertIn(text, payload["state"])

    def test_fake_server_parsing(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            game = Game2048(board=[[2, 2, 0, 0], [0] * 4, [0] * 4, [0] * 4])
            player = StatimPlayer(f"http://127.0.0.1:{server.server_port}", timeout=2)
            choice, probabilities = player.choose(game)
            self.assertIn(choice, game.legal_actions())
            self.assertAlmostEqual(sum(probabilities.values()), 1)
            self.assertTrue(_FakeHandler.request_body["state"].startswith("2048 game. "))
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
