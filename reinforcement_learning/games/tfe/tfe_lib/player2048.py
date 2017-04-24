from reinforcement_learning.rl.player_lib import Player

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

    def ask_Q_action(self, current_state):
        '''
        Ask for the reward
        '''
        action_to_do = None
        max_reward = 0

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

        return action_to_do, max_reward

    def ask_SARSA_action(self, current_state):
        pass

    def make_action(self):
        time.sleep(2)
        if self.alg == "Q":
            action_to_do, reward = self.ask_Q_action(self.current_state)
        
        elif self.alg == "SARSA":
            action_to_do, reward = self.ask_SARSA_action(self.current_state)

        self.current_action = action_to_do
        self.present_reward = reward

        print("[Choice %s algorithm]: " + Player2048.mapping[self.current_action])%(self.alg)
        self.action_function(action_to_do)

    def update(self, next_state, reward):
        if self.current_state is None:
            self.current_state = next_state
        else:

            past = self.compute_state(self.current_state)
            self.current_state = next_state
            now = self.compute_state(self.current_state)

            self.vfunction(past, now, self.current_action, reward, **self.vfunction_args)

    def compute_info(self):
        return {
            'player_actions_number':self.actions,
            'player_limited_actions':self.limit,
            'value_function': self.vfunction.__name__,
            'Q_matrix': self.Q_matrix,
            'player_name': self.player_name
        } 