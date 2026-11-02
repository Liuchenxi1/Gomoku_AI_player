"""Fast Gomoku threat detection used before neural-network MCTS.

Freestyle Gomoku only (five or more wins; no Renju forbidden moves).
All decisions use zero-based (row, column) coordinates and never mutate the
caller's board. This is a tactical safety layer, not a substitute for training.
"""
from __future__ import annotations

import numpy as np

DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))


def nearby_moves(board: np.ndarray, radius: int = 2) -> list[tuple[int, int]]:
    """Empty intersections close to stones, with center as the opening move."""
    size = board.shape[0]
    occupied = np.argwhere(board != 0)
    if not len(occupied):
        return [(size // 2, size // 2)]
    candidates = set()
    for r, c in occupied:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < size and 0 <= nc < size and board[nr, nc] == 0:
                    candidates.add((nr, nc))
    return sorted(candidates)


def winning_moves(board: np.ndarray, player: int, win_length: int = 5) -> set[tuple[int, int]]:
    """All empty cells that would immediately complete a win for player."""
    size = board.shape[0]
    wins = set()
    for r, c in nearby_moves(board, radius=1):
        board[r, c] = player
        try:
            for dr, dc in DIRECTIONS:
                n = 1
                for sign in (-1, 1):
                    rr, cc = r + sign * dr, c + sign * dc
                    while 0 <= rr < size and 0 <= cc < size and board[rr, cc] == player:
                        n += 1
                        rr += sign * dr
                        cc += sign * dc
                if n >= win_length:
                    wins.add((r, c))
                    break
        finally:
            board[r, c] = 0
    return wins


def open_lines_after_move(board: np.ndarray, r: int, c: int, player: int) -> tuple[int, int]:
    """Return number of open (length-3, length-4) lines through move.

    Caller temporarily places the stone. A line is open on both ends.
    """
    size = board.shape[0]
    open_three = open_four = 0
    for dr, dc in DIRECTIONS:
        count, ends = 1, 0
        for sign in (-1, 1):
            rr, cc = r + sign * dr, c + sign * dc
            while 0 <= rr < size and 0 <= cc < size and board[rr, cc] == player:
                count += 1
                rr += sign * dr
                cc += sign * dc
            if 0 <= rr < size and 0 <= cc < size and board[rr, cc] == 0:
                ends += 1
        if ends == 2:
            if count == 3:
                open_three += 1
            elif count == 4:
                open_four += 1
    return open_three, open_four


def find_strategy_move(env) -> tuple[int, str] | None:
    """Choose a forcing tactical action, or None to let MCTS decide.

    Priorities: win > emergency block > create fork > prevent enemy fork >
    create open three > stop enemy open three. A fork is a move giving at
    least two DISTINCT next-turn winning cells. This is a one-ply threat
    heuristic; it does not prove a forced win or loss.
    """
    if env.terminated:
        return None
    board = env.board.copy()
    player = int(env.current_player)
    enemy = -player
    size = env.size
    win_length = env.win_length
    own_wins = winning_moves(board, player, win_length)
    if own_wins:
        r, c = min(own_wins)
        return r * size + c, "win"
    enemy_wins = winning_moves(board, enemy, win_length)
    if enemy_wins:
        # If there are two independent winning points, the position may be
        # already lost; block one anyway rather than ignoring the danger.
        r, c = min(enemy_wins)
        return r * size + c, "block"

    choices = nearby_moves(board)
    own_forks = []
    enemy_forks = []
    own_threes = []
    enemy_threes = []
    for r, c in choices:
        board[r, c] = player
        own_open_three, _ = open_lines_after_move(board, r, c, player)
        if len(winning_moves(board, player, win_length)) >= 2:
            own_forks.append((r, c))
        if own_open_three:
            own_threes.append((r, c, own_open_three))
        board[r, c] = enemy
        enemy_open_three, _ = open_lines_after_move(board, r, c, enemy)
        if len(winning_moves(board, enemy, win_length)) >= 2:
            enemy_forks.append((r, c))
        if enemy_open_three:
            enemy_threes.append((r, c, enemy_open_three))
        board[r, c] = 0

    if own_forks:
        r, c = own_forks[0]
        return r * size + c, "create-fork"
    if enemy_forks:
        r, c = enemy_forks[0]
        return r * size + c, "prevent-fork"
    if own_threes:
        r, c, _ = max(own_threes, key=lambda x: x[2])
        return r * size + c, "create-open-three"
    if enemy_threes:
        r, c, _ = max(enemy_threes, key=lambda x: x[2])
        return r * size + c, "block-open-three"
    return None
