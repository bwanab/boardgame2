import sys
import copy

from six import StringIO
import numpy as np
import gymnasium as gym
from gymnasium import spaces


EMPTY = 0
BLACK = 1
WHITE = -1


def strfboard(board: np.array, render_characters: str='+ox', end: str='\n') -> str:
    """Format a board as a string

    Parameters
    ----
    board : np.array
    render_characters : str="+ox"
        - character at position 0 represents empty;
        - character at position 1 represents BLACK;
        - character at position -1 represents WHITE.
    end : str

    Returns
    ----
    s : str
    """
    s = ''
    for x in range(board.shape[0]):
        for y in range(board.shape[1]):
            c = render_characters[board[x][y]]
            s += c
        s += end
    return s[:-len(end)]


def is_index(board: np.array, location: np.array) -> str:
    """Check whether a location is a valid index of the board

    Parameters:
    ----
    board : np.array
    location : np.array

    Returns
    ----
    is_index : bool
    """
    # if len(location) != 2:
    #     return False
    if location < 0:
        return False
    x, y = np.unravel_index(location, board.shape)
    return x in range(board.shape[0]) and y in range(board.shape[1])


def extend_board(board: np.array) -> np.array:
    """Get the rotations of the board.

    Parameters:
    ----
    board : np.array, shape (n, n)

    Returns
    ----
    boards : np.array, shape (8, n, n)
    """
    assert board.shape[0] == board.shape[1]
    boards = np.stack([board,
            np.rot90(board), np.rot90(board, k=2), np.rot90(board, k=3),
            np.transpose(board), np.flipud(board),
            np.rot90(np.flipud(board)), np.fliplr(board)])
    return boards


class BoardGameEnv(gym.Env):

    metadata = {"render_modes": ["ansi", "human"]}
    reward_range = (-1, 1)

    PASS = -1
    RESIGN = -2

    def __init__(self, board_shape, illegal_action_mode: str='resign',
            render_characters: str='+ox', allow_pass: bool=True, render_mode = "human"):
        """Create a board game.

        Parameters
        ----
        board_shape: int or tuple    shape of the board
            - int: the same as (int, int)
            - tuple: in the form of (int, int), the two dimension of the board
        illegal_action_mode: str  What to do when the agent makes an illegal place.
            - 'resign': invalid location equivalent to resign
            - 'pass': invalid location equivalent to pass
        render_characters: str with length 3. characters used to render ('012', ' ox', etc)
        allow_pass: bool=True
            - True:  allow pass
            - False: not allow pass
        """
        self.allow_pass = allow_pass

        if illegal_action_mode == 'resign':
            self.illegal_equivalent_action = self.RESIGN
        elif illegal_action_mode == 'pass':
            self.illegal_equivalent_action = self.PASS
        else:
            raise ValueError()

        self.render_characters = {player : render_characters[player] for player \
                in [EMPTY, BLACK, WHITE]}

        if isinstance(board_shape, int):
            self.board_shape = (board_shape, board_shape)
        assert len(self.board_shape) == 2  # invalid board shape
        self.board = np.zeros(self.board_shape)
        assert self.board.size > 1  # Invalid board shape

        w,h = self.board_shape
        self.board_size = w * h

        self.observation_space = spaces.Box(low=-1, high=1, dtype=np.int8)
        self.action_space = spaces.Discrete(64)
        self.render_mode = render_mode

    def reset(self, *, seed=None, return_info=True, options=None):
        """Reset a new game episode. See gym.Env.reset()

        Parameters
        ----
        seed: Optional[int]=None
        return_info: bool=False
        options: Optional[dict]=None

        Returns
        ----
        next_state : (np.array, int)    next board and next player
        """
        self.board = np.zeros_like(self.board, dtype=np.int8)
        self.player = BLACK
        next_state = (self.board.reshape(self.board_size), self.player)
        if return_info:
            return next_state, {}
        else:
            return next_state

    def is_valid(self, state, action) -> bool:
        """Check whether the action is valid for current state.

        Parameters
        ----
        state : (np.array, int)    board and player
        action : np.array   location and skip

        Returns
        ----
        valid : bool
        """
        board, _ = state
        board = board.reshape(self.board_shape)
        if not is_index(board, action):
            return False
        x, y = action
        return board[x, y] == EMPTY

    def get_valid(self, state):
        """Get all valid locations for the current state.

        Parameters
        ----
        state : (np.array, int)    board and player

        Returns
        ----
        valid : np.array     current valid place for the player
        """
        board, _ = state
        board = board.reshape(self.board_shape)
        valid = np.zeros_like(board, dtype=np.int8)
        for x in range(board.shape[0]):
            for y in range(board.shape[1]):
                valid[x, y] = self.is_valid(state, np.array([x, y]))
        return valid

    def has_valid(self, state) -> bool:
        """Check whether there are valid locations for current state.

        Parameters
        ----
        state : (np.array, int)    board and player

        Returns
        ----
        has_valid : bool
        """
        board = state[0]
        board = board.reshape(self.board_shape)
        for x in range(board.shape[0]):
            for y in range(board.shape[1]):
                if self.is_valid(state, np.ravel_multi_index([x, y], board.shape)):
                    return True
        return False

    def get_winner(self, state):
        """Check whether the game has ended. If so, who is the winner.

        Parameters
        ----
        state : (np.array, int)   board and player. only board info is used

        Returns
        ----
        winner : None or int
            - None       The game is not ended and the winner is not determined.
            - env.BLACK  The game is ended with the winner BLACK.
            - env.WHITE  The game is ended with the winner WHITE.
            - env.EMPTY  The game is ended tie.
        """
        board, _ = state
        board = board.reshape(self.board_shape)
        for player in [BLACK, WHITE]:
            if self.has_valid((board, player)):
                return None
        return np.sign(np.nansum(board))

    def get_next_state(self, state, action):
        """Get the next state.

        Parameters
        ----
        state : (np.array, int)    board and current player
        action : np.array    location and skip indicator

        Returns
        ----
        next_state : (np.array, int)    next board and next player

        Raise
        ----
        ValueError : location in action is not valid
        """
        board, player = state
        board = board.reshape(self.board_shape)
        x, y = action
        if self.is_valid(state, action):
            board = copy.deepcopy(board)
            board[x, y] = player
        return board.reshape(self.board_size), -player

    def next_step(self, state, action):
        """Get the next observation, reward, termination, and info.

        Parameters
        ----
        state : (np.array, int)    board and current player
        action : np.array    location

        Returns
        ----
        next_state : (np.array, int)    next board and next player
        reward : float               the winner or zeros
        termination : bool           whether the game end or not
        info : {'valid' : np.array}    a dict shows the valid place for the next player
        """
        if not self.is_valid(state, action):
            action = self.illegal_equivalent_action
        if action == self.RESIGN:
            state = (np.reshape(state[0], self.board_size), state[1])
            return state, -state[1], True, {}
        while True:
            state = self.get_next_state(state, action)
            winner = self.get_winner(state)
            if winner is not None:
                return state, winner, True, {}
            if self.has_valid(state):
                break
            action = self.PASS
        return state, 0., False, {}

    def step(self, action):
        """See gym.Env.step().

        Parameters
        ----
        action : np.array    location

        Returns
        ----
        next_state : (np.array, int)    next board and next player
        reward : float        the winner or zero
        termination : bool    whether the game end or not
        truncation : bool=False
        info : dict={}
        """
        state = (self.board, self.player)
        next_state, reward, termination, info = self.next_step(state, action)
        board, self.player = next_state
        self.board = np.reshape(board, self.board_shape)
        return next_state, reward, termination, False, info

    def render(self, mode='human'):
        """See gym.Env.render()."""
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        s = strfboard(self.board, self.render_characters)
        outfile.write(s)
        if mode != 'human':
            return outfile
