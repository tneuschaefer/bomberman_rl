import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from .network import DeepQ
from settings import ROWS
from settings import COLS

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
EXPLORATION_RATE_START = 1
EXPLORATION_RATE_DECAY = 0.999
EXPLORATION_RATE_MIN = 0.1
BOMBING_LEGAL = True

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.current_round = 0

    if not os.path.isfile("model.pth"):
        self.logger.info("Setting up model from scratch.")
        self.model = DeepQ(ROWS * COLS, len(ACTIONS))
    else:
        self.logger.info("Loading model from saved state.")
        self.model = torch.load("model.pth")

    if not self.train:
        self.model.eval()
        self.device = torch.device("cpu")
        self.model.to(self.device)
    elif not os.path.isfile("target_model.pth"):
        self.logger.info("Setting up target model from model.")
        self.target_model = DeepQ(ROWS * COLS, len(ACTIONS))
        self.target_model.load_state_dict(self.model.state_dict())
    else:
        self.logger.info("Loading target model from saved state.")
        self.target_model = torch.load("target_model.pth")
        
    self.exploration_rate = EXPLORATION_RATE_START

def reset_self(self):
    #TODO
    return

def act(self, game_state: dict) -> str:
    """
    Makes a decision which ACTION to take
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # Check if we are in a different round
    if game_state["round"] != self.current_round:
        reset_self(self)
        self.current_round = game_state["round"]

    field, danger_state = pre_process(game_state)

    # chose between exploration and exploitation while in training
    if self.train:

        if np.random.rand() < self.exploration_rate:
            self.exploration_rate = update_exploration_rate(self.exploration_rate)

            self.logger.debug("Choosing action at random.")
            valid_action_list = valid_actions(danger_state, field, game_state['self'])
            if len(valid_action_list) == 0:
                self.logger.debug("No valid action available")
                valid_action_list.append('LEFT')   
            return random.choice(valid_action_list)
        
        self.exploration_rate = update_exploration_rate(self.exploration_rate)

    self.logger.debug("Querying model for action.")
    return ACTIONS[self.model.predict(state_to_features(self, field))]

def state_to_features(self, field: np.array):
    return torch.tensor(field.flatten(), dtype=torch.float32, device=self.device).unsqueeze(0)

def pre_process(game_state: dict) -> tuple[np.array, list, bool]:
    """
    Converts the game state to the input of our model.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None, None

    # will be set to true if agent is in danger of getting killed by explosion
    danger_state = False

    _, _, bomb_left, (x, y) = game_state['self']
    # -1 where the agent can't go (wall or active explosion)
    field = game_state['field']
    field = np.where(game_state['explosion_map'] != 0, -1, field)

    bombs = game_state['bombs']
    for (xb, yb), t in bombs:
        if (xb == x) and (abs(yb - y) < 4): danger_state = True
        if (yb == y) and (abs(xb - x) < 4): danger_state = True
        field[xb, yb] = 2

    coins = game_state['coins']
    for (xc, yc) in coins:
        field[xc, yc] = 3

    others = game_state['others']
    for (_, _, _, (xo, yo)) in others:
        field[xo, yo] = 4

    field[x, y] = 5

    # field with
    # -1 where wall or active explosion
    # 0 where free
    # 1 where crate
    # 2 where bomb
    # 3 where coin
    # 4 where other player
    # 5 where yourself

    # if we cut out the field to a 9x9 area focused around our agent (so they can see all possible escapes from a bomb they planted)
    # there's 81 cutouts
    # should be 81^7 different states
    # about 23 trillion

    return field, danger_state

def update_exploration_rate(rate: float) -> float:
    return max(rate * EXPLORATION_RATE_DECAY, EXPLORATION_RATE_MIN)

def valid_actions(danger_state: bool, field: np.array, self_state) -> list:
    '''
    Checks valid actions that are generally valid

    :param danger_state:  bool describing if the agent is in danger of getting blown up
    :param field: numpy array describing the playing field
    :param bomb_left: bool describing whether the agent can place a bomb
    '''
    _, _, bomb_left, (x, y) = self_state
    valid_actions = []
    if (field[x - 1, y] == 0): valid_actions.append('LEFT')
    if (field[x + 1, y] == 0): valid_actions.append('RIGHT')
    if (field[x, y - 1] == 0): valid_actions.append('UP')
    if (field[x, y + 1] == 0): valid_actions.append('DOWN')
    if bomb_left > 0 and BOMBING_LEGAL: valid_actions.append('BOMB')
    if not danger_state: valid_actions.append('WAIT')

    return valid_actions