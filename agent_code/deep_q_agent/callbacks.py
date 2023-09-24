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

from collections import deque
from random import shuffle

import settings as s

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
EXPLORATION_RATE_START = 1
EXPLORATION_RATE_DECAY = 0.999
EXPLORATION_RATE_MIN = 0.1
BOMBING_LEGAL = True

LEARN_FROM_RULE_AGENT = True

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
        self.model = DeepQ(ROWS * COLS + 2, len(ACTIONS))
    else:
        self.logger.info("Loading model from saved state.")
        self.model = torch.load("model.pth")

    if not self.train:
        self.model.eval()
        self.device = torch.device("cpu")
        self.model.to(self.device)
    elif not os.path.isfile("target_model.pth"):
        self.logger.info("Setting up target model from model.")
        self.target_model = DeepQ(ROWS * COLS + 2, len(ACTIONS))
        self.target_model.load_state_dict(self.model.state_dict())
    else:
        self.logger.info("Loading target model from saved state.")
        self.target_model = torch.load("target_model.pth")
        
    self.exploration_rate = EXPLORATION_RATE_START

    if LEARN_FROM_RULE_AGENT:
        rule_based_setup(self)

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
        self.logger.debug("---- NEW ROUND ----")
        if self.train:
            self.logger.debug(f"Exploration rate: {self.exploration_rate}")

    field, danger_state, bomb_left = pre_process(game_state)

    # chose between exploration and exploitation while in training
    if self.train:
        if np.random.rand() < self.exploration_rate:        
            if LEARN_FROM_RULE_AGENT:
                self.logger.debug("Letting rule_based_agent choose.")
                choice = rule_based_act(self, game_state)
            else:
                self.logger.debug("Choosing action at random.")            
                choice = random.choice(ACTIONS)
        else:
            self.logger.debug("Querying model for action.")
            choice = ACTIONS[self.model.predict(state_to_features(self, np.append(field.flatten(), [danger_state, bomb_left])))]

        self.exploration_rate = update_exploration_rate(self.exploration_rate)
    else:
        valid_action_list = valid_actions(danger_state, field, game_state['self'])

        self.logger.debug("Querying model for action.")
        choice = ACTIONS[self.model.predict(state_to_features(self, np.append(field.flatten(), [danger_state, bomb_left])))]
        if choice not in valid_action_list:
            self.logger.debug("Invalid action chosen. Choose randomly instead")
            choice = random.choice(valid_action_list)

    self.logger.debug(f"action: {choice}")
    return choice

def state_to_features(self, input: np.array):
    return torch.tensor(input, dtype=torch.float32, device=self.device).unsqueeze(0)

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

    return field, danger_state, bomb_left

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
    if field[x - 1, y] in [0,3]: valid_actions.append('LEFT')
    if field[x + 1, y] in [0,3]: valid_actions.append('RIGHT')
    if field[x, y - 1] in [0,3]: valid_actions.append('UP')
    if field[x, y + 1] in [0,3]: valid_actions.append('DOWN')
    if bomb_left > 0 and BOMBING_LEGAL: valid_actions.append('BOMB')
    if not danger_state: valid_actions.append('WAIT')

    if len(valid_actions) == 0:
        valid_actions.append('LEFT')

    return valid_actions

def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    if len(targets) == 0: return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]

def rule_based_setup(self):
    """Called once before a set of games to initialize data structures etc.

    The 'self' object passed to this method will be the same in all other
    callback methods. You can assign new properties (like bomb_history below)
    here or later on and they will be persistent even across multiple games.
    You can also use the self.logger object at any time to write to the log
    file for debugging (see https://docs.python.org/3.7/library/logging.html).
    """
    self.logger.debug('Successfully entered setup code')
    # Fixed length FIFO queues to avoid repeating the same actions
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0
    self.current_round = 0

def reset_self(self):
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0

def rule_based_act(self, game_state):
    """
    Called each game step to determine the agent's next action.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.
    """
    self.logger.info('Picking action according to rule set')
    # Check if we are in a different round
    if game_state["round"] != self.current_round:
        reset_self(self)
        self.current_round = game_state["round"]
    # Gather information about the game state
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']
    bomb_map = np.ones(arena.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)

    # If agent has been in the same location three times recently, it's a loop
    if self.coordinate_history.count((x, y)) > 2:
        self.ignore_others_timer = 5
    else:
        self.ignore_others_timer -= 1
    self.coordinate_history.append((x, y))

    # Check which moves make sense at all
    directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    valid_tiles, valid_actions = [], []
    for d in directions:
        if ((arena[d] == 0) and
                (game_state['explosion_map'][d] < 1) and
                (bomb_map[d] > 0) and
                (not d in others) and
                (not d in bomb_xys)):
            valid_tiles.append(d)
    if (x - 1, y) in valid_tiles: valid_actions.append('LEFT')
    if (x + 1, y) in valid_tiles: valid_actions.append('RIGHT')
    if (x, y - 1) in valid_tiles: valid_actions.append('UP')
    if (x, y + 1) in valid_tiles: valid_actions.append('DOWN')
    if (x, y) in valid_tiles: valid_actions.append('WAIT')
    # Disallow the BOMB action if agent dropped a bomb in the same spot recently
    if (bombs_left > 0) and (x, y) not in self.bomb_history: valid_actions.append('BOMB')
    self.logger.debug(f'Valid actions: {valid_actions}')

    # Collect basic action proposals in a queue
    # Later on, the last added action that is also valid will be chosen
    action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    shuffle(action_ideas)

    # Compile a list of 'targets' the agent should head towards
    cols = range(1, arena.shape[0] - 1)
    rows = range(1, arena.shape[0] - 1)
    dead_ends = [(x, y) for x in cols for y in rows if (arena[x, y] == 0)
                 and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(0) == 1)]
    crates = [(x, y) for x in cols for y in rows if (arena[x, y] == 1)]
    targets = coins + dead_ends + crates
    # Add other agents as targets if in hunting mode or no crates/coins left
    if self.ignore_others_timer <= 0 or (len(crates) + len(coins) == 0):
        targets.extend(others)

    # Exclude targets that are currently occupied by a bomb
    targets = [targets[i] for i in range(len(targets)) if targets[i] not in bomb_xys]

    # Take a step towards the most immediately interesting target
    free_space = arena == 0
    if self.ignore_others_timer > 0:
        for o in others:
            free_space[o] = False
    d = look_for_targets(free_space, (x, y), targets, self.logger)
    if d == (x, y - 1): action_ideas.append('UP')
    if d == (x, y + 1): action_ideas.append('DOWN')
    if d == (x - 1, y): action_ideas.append('LEFT')
    if d == (x + 1, y): action_ideas.append('RIGHT')
    if d is None:
        self.logger.debug('All targets gone, nothing to do anymore')
        action_ideas.append('WAIT')

    # Add proposal to drop a bomb if at dead end
    if (x, y) in dead_ends:
        action_ideas.append('BOMB')
    # Add proposal to drop a bomb if touching an opponent
    if len(others) > 0:
        if (min(abs(xy[0] - x) + abs(xy[1] - y) for xy in others)) <= 1:
            action_ideas.append('BOMB')
    # Add proposal to drop a bomb if arrived at target and touching crate
    if d == (x, y) and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(1) > 0):
        action_ideas.append('BOMB')

    # Add proposal to run away from any nearby bomb about to blow
    for (xb, yb), t in bombs:
        if (xb == x) and (abs(yb - y) < 4):
            # Run away
            if (yb > y): action_ideas.append('UP')
            if (yb < y): action_ideas.append('DOWN')
            # If possible, turn a corner
            action_ideas.append('LEFT')
            action_ideas.append('RIGHT')
        if (yb == y) and (abs(xb - x) < 4):
            # Run away
            if (xb > x): action_ideas.append('LEFT')
            if (xb < x): action_ideas.append('RIGHT')
            # If possible, turn a corner
            action_ideas.append('UP')
            action_ideas.append('DOWN')
    # Try random direction if directly on top of a bomb
    for (xb, yb), t in bombs:
        if xb == x and yb == y:
            action_ideas.extend(action_ideas[:4])

    # Pick last action added to the proposals list that is also valid
    while len(action_ideas) > 0:
        a = action_ideas.pop()
        if a in valid_actions:
            # Keep track of chosen action for cycle detection
            if a == 'BOMB':
                self.bomb_history.append((x, y))

            return a