from rl.player_lib import Player
from random import random, randint
from operator import itemgetter
from time import sleep

class Player2048(Player):
    mapping = {0:"'Up'",\
            1:"'Down'",\
            2:"'Left'",\
            3:"'Right'"
        }

    def __init__(self, name, **kwargs):
        super(Player2048, self).__init__(name, **kwargs)
        self.current_state = None
        self.current_action = None

    def ask_action(self, current_state):
        '''
        Ask for the reward
        '''
        action_to_do = None
        max_reward = 0

        if self.is_training and not self.alg=="SARSA":
            for action in [0, 1, 2, 3]:
                next_state, is_done, reward = self.interface.ask_game('commands',\
                                                Player2048.mapping[action],\
                                                current_state)
                if reward == max_reward:
                    if isinstance(action_to_do, int):
                        if random() < 0.5:
                            action_to_do = action
                    
                    else:
                        action_to_do = randint(0,3)

                elif reward > max_reward:
                    action_to_do = action
                    max_reward = reward
        else:
            # print(current_state)
            current_state = self.compute_state(current_state)

            if current_state in self.Q_matrix:
                # print(current_state)
                action_to_do = max(self.Q_matrix[current_state].items(), key=itemgetter(1))[0]
            else:
                action_to_do = randint(0,3)
                # print(action_to_do)
            # print(action_to_do)
        return action_to_do, max_reward


    def make_action(self, next_state):
        if not self.is_training:
            # initial state
            self.current_state = next_state
        # print(self.current_state
        action_to_do, reward = self.ask_action(self.current_state)
        self.current_action = action_to_do
        self.present_reward = reward

        if self.interface.game_mode() == 'text':
            # print("Current state")
            # print(self.current_state)
            # print("[Choice " + str(self.alg) + " algorithm]: " + Player2048.mapping[self.current_action])
            # Player2048.stack += 1
            return self.action_function(action_to_do,'text')
        
        if self.interface.game_mode() == 'gui':
            # sleep(0.05)
            # print("[Choice " + str(self.alg) + " algorithm]: " + Player2048.mapping[self.current_action])
            self.action_function(action_to_do,'gui')

    def update(self, next_state, reward):
        if self.current_state is None:
            self.current_state = next_state

        else:
            past = self.compute_state(self.current_state)
            self.current_state = next_state
            now = self.compute_state(self.current_state)
            self.vfunction(past, now, self.current_action, reward, **self.vfunction_args)