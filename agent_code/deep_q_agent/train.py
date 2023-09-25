from collections import namedtuple, deque

import random
from typing import List
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import events as e
from .callbacks import state_to_features
from .callbacks import pre_process
from .callbacks import ACTIONS

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

EXPERIENCE_RELAY_SIZE = 10000
LEARNING_RATE = 0.7
UPDATE_FREQUENCY = 4
UPDATE_BATCH_SIZE = 128
SYNC_FREQUENCY = 400
DISCOUNT_FACTOR = 0.9

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # use GPU if available
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model.to(self.device)
    self.target_model.to(self.device)
    self.logger.info("Running on " + str(self.device))

    self.optimizer = optim.AdamW(self.model.parameters(), lr = LEARNING_RATE, amsgrad = True)

    self.experience_relay = deque(maxlen=EXPERIENCE_RELAY_SIZE)

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    After each step in the game a reward is calculated and the transition gets saved.

    With UPDATE_FREQUENCY the model gets trained and with SYNC_FREQUENCY the target model gets updated
    to the state dictionary of the main model

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    old_field, old_danger_state, old_bomb_left = pre_process(old_game_state)
    new_field, new_danger_state, new_bomb_left = pre_process(new_game_state)
    action = torch.tensor([[ACTIONS.index(self_action)]], dtype=torch.int64, device=self.device)
    reward = torch.tensor([reward_from_events(self, old_field, old_danger_state, old_bomb_left, new_field, new_danger_state, new_bomb_left, events)], dtype=torch.int64, device=self.device)
    self.experience_relay.append(Transition(state_to_features(self, np.append(old_field.flatten(), [old_danger_state, old_bomb_left])),
                                            action,
                                            state_to_features(self, np.append(new_field.flatten(), [new_danger_state, new_bomb_left])),
                                            reward))

    step = old_game_state["step"]
    if step % UPDATE_FREQUENCY == 0:
        train(self)
    if step % SYNC_FREQUENCY == 0:
        sync(self)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Trains, synchronizes and stores the 2 networks after the end of each round.

    :param self: The same object that is passed to all of your callbacks.
    """

    train(self)
    sync(self)

    torch.save(self.model, "model.pth")
    torch.save(self.target_model, "target_model.pth")


def reward_from_events(self, old_field, old_danger_state, old_bomb_left, new_field, new_danger_state, new_bomb_left, events) -> int:
    """
    Calculates the rewards.
    """
    reward_sum = 0
    if old_danger_state:
        if not new_danger_state:
            reward_sum += 500
        elif e.WAITED in events or e.BOMB_DROPPED in events:
            reward_sum -= 50
        elif e.INVALID_ACTION in events:
            reward_sum -= 40

    game_rewards = {
        e.MOVED_LEFT: 1,
        e.MOVED_RIGHT: 1,
        e.MOVED_UP: 1,
        e.MOVED_DOWN: 1,
        e.WAITED: -1,
        e.INVALID_ACTION: -10,
        e.BOMB_DROPPED: 1,
        e.CRATE_DESTROYED: 5,
        e.COIN_COLLECTED: 50,
        e.KILLED_OPPONENT: 250,
        e.KILLED_SELF: -1000,
        e.GOT_KILLED: -500,
        e.SURVIVED_ROUND: 1000
    }
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def train(self):
    '''
    Training happens here using a random sample of memorized transitions
    '''
    if len(self.experience_relay) < UPDATE_BATCH_SIZE:
        return
    
    # each transition entry contains a batch
    batch = Transition(*zip(*random.sample(self.experience_relay, UPDATE_BATCH_SIZE)))

    # Q(s_t,a)
    q_s_t_a = self.model(torch.cat(batch.state)).gather(1, torch.cat(batch.action))

    with torch.no_grad():
        # V(s_{t+1}) using target model
        v_s_t_1 = self.target_model(torch.cat(batch.next_state)).max(1)[0]
    # Q(s_{t+1},a) expected Q values
    q_s_t_1_a = (v_s_t_1 * DISCOUNT_FACTOR) + torch.cat(batch.reward)

    self.optimizer.zero_grad()

    # Huber loss
    criterion = nn.HuberLoss()
    loss = criterion(q_s_t_a, q_s_t_1_a.unsqueeze(1))
    loss.backward()

    nn.utils.clip_grad_value_(self.model.parameters(), 90)
    self.optimizer.step()

def sync(self):
    '''
    Updates the target model with the state dictionary from the main model
    '''
    self.target_model.load_state_dict(self.model.state_dict())