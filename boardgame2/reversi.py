import copy
import itertools
import numpy as np

from .env import EMPTY
from .env import BoardGameEnv
from .env import is_index, board_player_from_state


class ReversiEnv(BoardGameEnv):

    def __init__(self, board_shape=8, render_characters: str='+ox', render_mode="human"):
        super().__init__(board_shape=board_shape,
            illegal_action_mode='resign', render_characters=render_characters,
            allow_pass=False, render_mode=render_mode)  # reversi does not allow pass

    def reset(self, *, seed=None, return_info=True, options=None):
        super().reset(seed=seed, return_info=return_info, options=options)

        x, y = (s // 2 for s in self.board_shape)
        board, player = board_player_from_state(self.board)
        board[x - 1][y - 1] = board[x][y] = 1
        board[x - 1][y] = board[x][y - 1] = -1
        self.board[self.board_size] = player
        next_state = self.board
        if return_info:
            return next_state, {}
        else:
            return next_state

    def is_valid(self, state, action) -> bool:
        """
        Parameters
        ----
        state : (np.array, int)    board and player
        action : np.array   location

        Returns
        ----
        valid : bool     whether the current action is a valid action
        """
        
        board, player = board_player_from_state(state)

        if not is_index(board, action):
            return False

        if isinstance(action, int) or isinstance(action, np.integer):
            x, y = np.unravel_index(action, self.board_shape)
        else:
            x, y = action

        if board[x, y] != EMPTY:
            return False

        for dx in [-1, 0, 1]:  # loop on the 8 directions
            for dy in [-1, 0, 1]:
                if (dx, dy) == (0, 0):
                    continue
                xx, yy = x, y
                for count in itertools.count():
                    xx, yy = xx + dx, yy + dy
                    if xx < 0 or xx >= self.board_shape[0] or yy < 0 or yy >= self.board_shape[1]:
                        break
                    if not is_index(board, np.ravel_multi_index([xx, yy], self.board_shape)):
                        break
                    if board[xx, yy] == EMPTY:
                        break
                    if board[xx, yy] == -player:
                        continue
                    if count:  # and is player
                        return True
                    break
        return False

    def get_next_state(self, state, action):
        """
        Parameters
        ----
        state : (np.array, int)    board and current player
        action : np.array    location

        Returns
        ----
        next_state : (np.array, int)    next board and next player
        """

        board, player = board_player_from_state(state)
        # board = copy.deepcopy(board)

        if self.is_valid(state, action):
            x, y = np.unravel_index(action, self.board_shape)
            board[x, y] = player
            self.board[self.board_size] = -player
            for dx in [-1, 0, 1]:  # loop on the 8 directions
                for dy in [-1, 0, 1]:
                    if (dx, dy) == (0, 0):
                        continue
                    xx, yy = x, y
                    for count in itertools.count():
                        xx, yy = xx + dx, yy + dy
                        if xx < 0 or xx >= self.board_shape[0] or yy < 0 or yy >= self.board_shape[1]:
                            break
                        if not is_index(board, np.ravel_multi_index([xx, yy], self.board_shape)):
                            break
                        if board[xx, yy] == EMPTY:
                            break
                        if board[xx, yy] == player:
                            for i in range(count+1):  # overwrite
                                board[x + i * dx, y + i * dy] = player
                            break
        else:
            if action == self.PASS:
                self.board[self.board_size] = -player
        return np.array(board.reshape(self.board_size).tolist() + [-player], dtype=board.dtype)
