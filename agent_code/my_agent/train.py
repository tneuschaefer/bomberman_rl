from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


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

    # Idea: Add your own events to hand out rewards
    if ...:
        events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))


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
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)
    
    x_UP = list()
    x_RIGHT = list()
    x_DOWN = list()
    x_LEFT = list()
    x_WAIT = list()
    x_BOMB = list()
    for transition in self.transitions:
        if transition.action = 'UP':
            x_up.append(transition)
        else if transition.action = 'RIGHT':
            x_up.append(transition)
        else if transition.action = 'DOWN':
            x_up.append(transition)
        else if transition.action = 'LEFT':
            x_up.append(transition)
        else if transition.action = 'WAIT':
            x_up.append(transition)
        else if transition.action = 'BOMB':
            x_up.append(transition)
    
    state_matrix_UP = np.vstack([transition.state for transition in x_UP])
    state_matrix_RIGHT = np.vstack([transition.state for transition in x_RIGHT])
    state_matrix_DOWN = np.vstack([transition.state for transition in x_DOWN])
    state_matrix_LEFT = np.vstack([transition.state for transition in x_LEFT])
    state_matrix_WAIT = np.vstack([transition.state for transition in x_WAIT])
    state_matrix_BOMB = np.vstack([transition.state for transition in x_BOMB])
    
    next_state_matrix_UP = np.vstack([transition.next_state for transition in x_UP])
    next_state_matrix_RIGHT = np.vstack([transition.next_state for transition in x_RIGHT])
    next_state_matrix_DOWN = np.vstack([transition.next_state for transition in x_DOWN])
    next_state_matrix_LEFT = np.vstack([transition.next_state for transition in x_LEFT])
    next_state_matrix_WAIT = np.vstack([transition.next_state for transition in x_WAIT])
    next_state_matrix_BOMB = np.vstack([transition.next_state for transition in x_BOMB])
    
    reward_vector_UP = np.vstack([transition.reward for transition in x_UP])
    reward_vector_RIGHT = np.vstack([transition.reward for transition in x_RIGHT])
    reward_vector_DOWN = np.vstack([transition.reward for transition in x_LEFT])
    reward_vector_LEFT = np.vstack([transition.reward for transition in x_DOWN])
    reward_vector_WAIT = np.vstack([transition.reward for transition in x_WAIT])
    reward_vector_BOMB = np.vstack([transition.reward for transition in x_BOMB])
    
    beta[:,0] = beta[:,0] + 1/len(x_UP) sum(state_matrix_UP.T * ((reward_vector_UP + GAMA * max(beta * next_state_matrix_UP)) - state_matrix*beta[:,0]))
    beta[:,1] = beta[:,1] + 1/len(x_RIGHT) sum(state_matrix_RIGHT.T * ((reward_vector_RIGHT + GAMA * max(beta * next_state_matrix_RIGHT)) - state_matrix*beta[:,1]))
    beta[:,2] = beta[:,2] + 1/len(x_DOWN) sum(state_matrix_DOWN.T * ((reward_vector_DOWN + GAMA * max(beta * next_state_matrix_LEFT)) - state_matrix*beta[:,2]))
    beta[:,3] = beta[:,3] + 1/len(x_LEFT) sum(state_matrix_LEFT.T * ((reward_vector_LEFT + GAMA * max(beta * next_state_matrix_DOWN)) - state_matrix*beta[:,3]))
    beta[:,4] = beta[:,4] + 1/len(x_WAIT) sum(state_matrix_WAIT.T * ((reward_vector_WAIT + GAMA * max(beta * next_state_matrix_WATI)) - state_matrix*beta[:,4]))
    beta[:,5] = beta[:,5] + 1/len(x_BOMB) sum(state_matrix_BOMB.T * ((reward_vector_BOMB + GAMA * max(beta * next_state_matrix_BOMB)) - state_matrix*beta[:,5]))


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
