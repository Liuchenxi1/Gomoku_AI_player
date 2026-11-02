"""Run with python -m unittest discover -s tests -p test_strategy.py."""
import unittest
from gomoku import GomokuEnv
from alphazero.strategy import find_strategy_move

class StrategyTests(unittest.TestCase):
    def board(self, stones, player=-1):
        env = GomokuEnv(15)
        env.current_player = player
        for color, cells in stones:
            for row, col in cells:
                env.board[row, col] = color
        env.move_count = int((env.board != 0).sum())
        return env

    def assert_move_in(self, env, cells, reason=None):
        before = env.board.copy()
        actual = find_strategy_move(env)
        self.assertIsNotNone(actual)
        action, label = actual
        self.assertIn(divmod(action, env.size), cells)
        if reason:
            self.assertEqual(label, reason)
        self.assertTrue((before == env.board).all(), 'Strategy must not modify board')

    def test_immediate_win(self):
        env = self.board([(-1, [(7, 3), (7, 4), (7, 5), (7, 6)])])
        self.assert_move_in(env, {(7, 2), (7, 7)}, 'win')

    def test_immediate_block(self):
        env = self.board([(1, [(5, 5), (6, 6), (7, 7), (8, 8)])])
        self.assert_move_in(env, {(4, 4), (9, 9)}, 'block')

    def test_open_three_defense(self):
        env = self.board([(1, [(5, 5), (6, 6), (7, 7)])])
        self.assert_move_in(env, {(4, 4), (8, 8)}, 'prevent-fork')

    def test_create_open_four(self):
        env = self.board([(-1, [(5, 5), (6, 6), (7, 7)])])
        self.assert_move_in(env, {(4, 4), (8, 8)}, 'create-fork')

if __name__ == '__main__':
    unittest.main()
