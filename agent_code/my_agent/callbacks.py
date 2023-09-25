import os
import pickle
import random

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def rotate90(xy):
    xy = [xy[1], 16-xy[0]]
    return xy

def rotate180(xy):
    xy = [16-xy[0], 16-xy[1]]
    return xy

def rotate270(xy):
    xy = [16-xy[1], xy[0]]
    return xy

def mirroring(xy):
    xy = [xy[1], xy[0]]
    return xy


def rotated_action(action, game_state):
    if action == 'WAIT' or action == 'BOMB':
        return action
    
    xy = game_state['self'][3]
    
    if xy[0] <= 8 and xy[1] > 8:
        index = ACTIONS.index(action)
        action = ACTIONS[(index+3)%4]
    elif xy[0] > 8 and xy[1] >= 8:
        index = ACTIONS.index(action)
        action = ACTIONS[(index+2)%4]
    elif xy[0] > 8 and xy[1] < 8:
        index = ACTIONS.index(action)
        action = ACTIONS[(index+1)%4]

    if xy[0] < xy[1]:
        if action ==  'UP':
            return 'LEFT'
        elif action == 'RIGHT':
            return 'DOWN'
        elif action == 'DOWN':
            return 'RIGHT'
        elif action == 'LEFT':
            return 'UP'
    else:
        return action



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
     
    self.beta = np.zeros((4+(4*17),6))
    
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    
    self.logger.info(game_state['self'])
    
    if game_state['round'] > 10:
        random_prob = 0.05
    else:
        random_prob = 1/(0.5*game_state['round'])

    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .2, .0])

    self.logger.debug("Querying model for action.")
    return rotated_action(ACTIONS[np.argmax(state_to_features(game_state)@self.beta)], game_state)


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None
    
    xy = game_state['self'][3]
    coins = []
    bombs = []
    
    if xy[0] <= 8 and xy[1] > 8:
        xy = rotate270(xy)
        game_field = np.rot90(game_state['field'], k=3)
        explosion_map = np.rot90(game_state['explosion_map'], k=3)
        for c in game_state['coins']:
            coins.append(rotate270(c))
        for b in game_state['bombs']:
            bombs.append((rotate270(game_state['bombs'][0]),game_state['bombs'][1])
    elif xy[0] > 8 and xy[1] >= 8:
        xy = rotate180(xy)
        game_field = np.rot90(game_state['field'], k=2)
        explosion_map = np.rot90(game_state['explosion_map'], k=2)
        for c in game_state['coins']:
            coins.append(rotate180(c))
        for b in game_state['bombs']:
            bombs.append((rotate180(game_state['bombs'][0]),game_state['bombs'][1])
    elif xy[0] > 8 and xy[1] < 8:
        xy = rotate90(xy)
        game_field = np.rot90(game_state['field'], k=1)
        explosion_map = np.rot90(game_state['explosion_map'], k=1)
        for c in game_state['coins']:
            coins.append(rotate90(c))
        for b in game_state['bombs']:
            bombs.append((rotate90(game_state['bombs'][0]),game_state['bombs'][1])
    else:
        game_field = game_state['field']
        explosion_map = game_state['explosion_map']
        coins = game_state['coins']
        bombs = game_state['bombs']

    if xy[0] < xy[1]:
        xy = mirroring(xy)
        game_field = game_field.T
        for c in coins:
            c = mirroring(c)
    
    nearest_coin = (0,0)
    distance_nearest_coin = 0
    if coins:
        distance_nearest_coin = 31
        for c in coins:
            c_distance = abs(xy[0]-c[0])+abs(xy[1]-c[1])
            if c_distance <= distance_nearest_coin:
                distance_nearest_coin = c_distance
                nearest_coin = c
    
    nearest_bomb = (0,0)
    distance_nearest_bomb = 0
    if coins:
        distance_nearest_bomb = 31
        for b in bombs:
            b_distance = abs(xy[0]-c[0][0])+abs(xy[1]-c[0][1])
            if b_distance <= distance_nearest_bomb:
                distance_nearest_bomb = b_distance
                nearest_bomb = b

    X = np.array(xy)
    X = np.append(X, game_state['self'][1:3])
    X = np.append(X, game_field.flatten())
    X = np.append(X, eplosion_map.flatten())
    #X = np.append(X, nearest_coin)
    #X = np.append(X, distance_nearest_coin)
    #X = np.append(X, nearest_bomb)
    #X = np.append(X, distance_nearest_bomb)
    
    # For example, you could construct several channels of equal shape, ...
    channels = []
    for x in range(1,9):
        for y in range(1,x+1) and (x%2==0 or y%2==0): # 26 possible positions
            if xy[0] == x and xy[1] == y:
                channels.append(X)
            else:
                channels.append(np.zeros(len(X)))
                
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)
