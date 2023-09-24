from collections import namedtuple, deque

import pickle
import random
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import events as e
from .callbacks import state_to_features
from .callbacks import pre_process

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

EXPERIENCE_RELAY_SIZE = 10000
LEARNING_RATE = 0.7
UPDATE_FREQUENCY = 4
UPDATE_BATCH_SIZE = 128
SYNC_FREQUENCY = 1000
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
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    old_field, old_danger_state = pre_process(old_game_state)
    new_field, new_danger_state = pre_process(new_game_state)
    reward = reward_from_events(self, old_field, old_danger_state, new_field, new_danger_state, events)
    self.experience_relay.append(Transition(state_to_features(self, old_field),
                                            self_action,
                                            state_to_features(self, new_field),
                                            reward))
    step = old_game_state["step"]
    if step % UPDATE_FREQUENCY == 0:
        train(self)
    if step % SYNC_FREQUENCY == 0:
        sync(self)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    #last_features, old_danger_state = state_to_features(last_game_state)
    #reward = reward_from_events(self, events)
    #self.experience_relay.append(Transition(torch.tensor(last_features, device=self.device).unsqueeze(0),
    #                                        last_action,
    #                                        None,
    #                                        reward))
    #train(self)
    #if last_game_state["step"] % SYNC_FREQUENCY == 0:
    #    sync(self)

    # Store the model
    torch.save(self.model, "model.pth")
    torch.save(self.target_model, "target_model.pth")


def reward_from_events(self, old_field, old_danger_state, new_field, new_danger_state, events) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    reward_sum = 0
    if old_danger_state and not new_danger_state:
        reward_sum += 500

    game_rewards = {
        e.MOVED_LEFT: 1,
        e.MOVED_RIGHT: 1,
        e.MOVED_UP: 1,
        e.MOVED_DOWN: 1,
        e.INVALID_ACTION: -10,
        e.BOMB_DROPPED: 2,
        e.CRATE_DESTROYED: 5,
        e.COIN_COLLECTED: 50,
        e.KILLED_OPPONENT: 250,
        e.KILLED_SELF: -1000,
        e.GOT_KILLED: -500,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def train(self):
    if len(self.experience_relay) < UPDATE_BATCH_SIZE:
        return
    
    batch = random.sample(self.experience_relay, UPDATE_BATCH_SIZE)
    batch = Transition(*zip(*batch))

    q_s_t_a = self.model(torch.cat(batch.state)).gather(1, torch.cat(batch.action))

    with torch.no_grad():
        v_s_t_1 = self.target_model(torch.cat(batch.next_state)).max(1)[0]
    # Compute the expected Q values
    q_s_t_1_a = (v_s_t_1 * DISCOUNT_FACTOR) + torch.cat(batch.reward)

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    self.loss = criterion(q_s_t_a, q_s_t_1_a.unsqueeze(1))

    # Optimize the model
    self.optimizer.zero_grad()
    self.loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
    self.optimizer.step()

def sync(self):
    self.target_model.load_state_dict(self.model.state_dict())