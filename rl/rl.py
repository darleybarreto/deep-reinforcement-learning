import numpy as np
from random import random, randint
import pyautogui
import time

def ValueFunctionFactory(**kwargs):
	'''
	v_factory = ValueFunctionFactory(Q=Q_m, alpha=0.75, gamma=0.9)
	
	Q_learning = v_factory.createFunction('Q')
	SARSA = v_factory.createFunction('SARSA')

	v_factory_r = ValueFunctionFactory(shape_Q=(5,5), alpha=0.75, gamma=0.9)
	Q_m = v_factory_r.returnQ_matrix()

	Q_learning_r = v_factory_r.createFunction('Q')
	SARSA_r = v_factory_r.createFunction('SARSA')
	'''

	alpha = kwargs.get('alpha', 0.5)
	gamma = kwargs.get('gamma', 1.)

	Q_matrix =  kwargs.get('Q', False)
	if not Q_matrix:
		if kwargs.get('shape_Q', False):
			n_states, n_actions =  kwargs['shape_Q']
			if not (type(n_states) == int and type(n_actions) == int):
				raise Exception("Both n_states and n_actions must be int")
		else:
			raise Exception("shape_Q not found while Q is None")

		Q_matrix = np.random.rand(n_states, n_actions)


	def __Q(actual_state, next_state, action, reward, **lwargs):
		'''
		Q-Learning function

		Readings:
			http://artint.info/html/ArtInt_265.html
			http://www.cse.unsw.edu.au/~cs9417ml/RL1/algorithms.html

		Simple Q-Learning: 
			Q(s, a) = (1 - alpha)*Q(s,a) + alpha(R(s, a s') + gamma*max[a'](Q(s', a'))

		Epsilon-greedy Q-Learning:
			For a small given probability epsilon, dont use max[a']
			Q(s, a) = (1 - alpha)*Q(s,a) + alpha(R(s, a s') + gamma*any[a'](Q(s', a'))

		Exploratory Q-Learning:
			Let fe(u,n) = u + k/n, and
			u = max[a'](Q(s',a'))
			n = max number of choosing u state
			k = any given constant

			Q(s, a) = (1 - alpha)*Q(s,a) + alpha(R(s, a s') + gamma*fe(u,n))
		'''

		mode = lwargs.get('mode', None)
		epsilon = lwargs.get('epsilon', 0.001)
		ef = lwargs.get('ef', None)
		k = 10

		if not mode== 'ef':

			if mode == "simple":
				sample = alpha*(reward + gamma*(np.amax(Q_matrix[next_state])))

			elif mode == "epsilon":
				# e-greedy
				if random() <= epsilon:
					sample = alpha*(reward + gamma*(random.choice(Q_matrix[next_state], size=1, replace=False)))

				else:
					sample = alpha*(reward + gamma*(np.amax(Q_matrix[next_state])))

			Q_matrix[actual_state, action] = (1 - alpha)*Q_matrix[actual_state, action]+sample


		elif mode == 'ef':
			# exploratory function
			pass

		else:
			raise Exception("No mode for Q-Learning named %s found")%(mode)

	def __SARSA():
		'''
		'''
		pass

	def createFunction(alg):
		if alg == 'Q':
			return __Q

		elif alg == 'SARSA':
			return __SARSA

		else:
			raise Exception("No algorithm named %s found")%(alg)

	return createFunction, Q_matrix

def ActionFactory():
	
	def action2048(action):
		mapping = {0:'up',\
			1:'down',\
			2:'left',\
			3:'right'
		}
		pyautogui.press(mapping[action])

	def createAction(action_type):
		if action_type == '2048':
			return action2048
		else:
			raise Exception("No action function named %s found")%(action_type)

	return createAction


class Player(object):
    def __init__(self, name, **kwargs):
        self._compute_score = 0
        self.actions = kwargs.get('actions', 1000)
        self.limit = kwargs.get('limit', False)
        self.Q_matrix = kwargs.get('Q_matrix', None)
        self.alg = kwargs.get('alg', 'Q')
        self.interface = None
        self.player_name = name

        self.vfunction_args = {
			'mode':kwargs.get('mode', 'simple'),\
			'epsilon':kwargs.get('epsilon', None),\
			'ef':kwargs.get('ef', None)
		}

        factory, Q_matrix = ValueFunctionFactory(Q=self.Q_matrix,\
            shape_Q=kwargs.get('shape_Q', False),\
            alpha=kwargs.get('alpha', 0.5),\
            gamma=kwargs.get('gamma', 1.),\
        )

        self.action_function = ActionFactory()(kwargs.get('action_type', None))

        self.vfunction = factory(self.alg)
        if not self.Q_matrix:
            self.Q_matrix = Q_matrix

    def compute_info(self):
        return {
            'player_actions_number':self.actions,
            'player_limited_actions':self.limit,
            'value_function': self.vfunction.__name__,
            'Q_matrix': self.Q_matrix,
            'player_name': self.player_name
        }


    def attach_interface(self, interface):
    	interface.connect_player(self)
    	self.interface = interface


class Player2048(Player):

	mapping = {0:"'Up'",\
			1:"'Down'",\
			2:"'Left'",\
			3:"'Right'"
		}

	def __init__(self, name, **kwargs):
		super(Player2048, self).__init__(name, **kwargs)
		self.actual_state = None
		self.present_action = None

	def ask_max_reward(self, actual_state):
		'''
		Ask for the reward
		'''
		action_to_do = None
		max_reward = 0

		for action in [0, 1, 2, 3]:
			next_state, is_done, reward = self.interface.ask_game('commands',\
											Player2048.mapping[action],\
											actual_state)
			# print("ação: " + str(action))
			# print("recompensa " + str(reward))
			if reward == max_reward:
				if action_to_do:
					if random() < 0.5:
						action_to_do = action
				
				else:
					action_to_do = randint(0,3)

			elif reward > max_reward:
				action_to_do = action
				max_reward = reward

		return action_to_do, max_reward

	def make_action(self):
		time.sleep(2)
		action_to_do, reward = self.ask_max_reward(self.actual_state)
		self.present_action = action_to_do
		self.present_reward = reward
		print("Choice: " + Player2048.mapping[self.present_action])
		self.action_function(action_to_do)

	def update(self, next_state, reward):
		if self.actual_state is None:
			self.actual_state = next_state
		else:
			past = self.compute_state(self.actual_state)
			self.actual_state = next_state
			now = self.compute_state(self.actual_state)

			self.vfunction(past, now, self.present_action, reward, **self.vfunction_args)



	def compute_state(self, state):
		return int(state.sum())


	def compute_info(self):
		return {
            'player_actions_number':self.actions,
            'player_limited_actions':self.limit,
            'value_function': self.vfunction.__name__,
            'Q_matrix': self.Q_matrix,
            'player_name': self.player_name
        }