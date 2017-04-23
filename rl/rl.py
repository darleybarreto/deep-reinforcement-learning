import numpy as np
import pyautogui

def ValueFunctionFactory(**kwargs):
	'''
	Readings:
		http://www.cse.unsw.edu.au/~cs9417ml/RL1/index.html

	v_factory = ValueFunctionFactory(Q=Q_m, alpha=0.75, gamma=0.9)
	
	Q_learning = v_factory.createFunction('Q')
	SARSA = v_factory.createFunction('SARSA')

	v_factory_r = ValueFunctionFactory(shape_Q=(5,4), alpha=0.75, gamma=0.9)
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


	def __Q(current_state, next_state, action, reward, **lwargs):
		'''
		Q-Learning function

		Readings:
			http://artint.info/html/ArtInt_265.html

		Simple Q-Learning: 
			Q(s, a) = (1 - alpha)*Q(s,a) + alpha(R(s, a s') + gamma*max[a'](Q(s', a'))

		Epsilon-greedy Q-Learning:
			For a small given probability epsilon, dont use max[a']
			Q(s, a) = (1 - alpha)*Q(s,a) + alpha(R(s, a s') + gamma*any[a'](Q(s', a'))

		Softmax Q-Learning:
			Istead of choosing randomly with the same probability for selecting
			each move other than the best one, the agente chooses randomly among
			all available actions, but according to some probability weighting system.
			Q(s, a) = (1 - alpha)*Q(s,a) + alpha(R(s, a s') + gamma*(e^(Q(s,a)/tau)/sum(e^(Q(s,a)/tau))

			Where tau is the temperature specifying how randomly values should be chosen.
			When tau is high, the actions are chosen in almost equal amounts. 
			As the temperature is reduced, the highest-valued actions are more likely to be chosen and, 
			in the limit as tau -> 0, the best action is always chosen.

		Exploratory Q-Learning:
			Let fe(u,n) = u + k/n, and
			u = max[a'](Q(s',a'))
			n = max number of choosing u state
			k = any given constant

			Q(s, a) = (1 - alpha)*Q(s,a) + alpha(R(s, a s') + gamma*fe(u,n))
		'''

		mode = lwargs.get('mode', 'simple')

		if not mode== 'ef':

			elif mode == "simple":
				sample = alpha*(reward + gamma*(np.amax(Q_matrix[next_state])))

			elif mode == "greddy":
				# e-greedy
				epsilon = lwargs.get('epsilon', 0.001)
				if np.random.uniform(0, 1) <= epsilon:
					sample = alpha*(reward + gamma*(np.random.choice(Q_matrix[next_state])))

				else:
					sample = alpha*(reward + gamma*(np.amax(Q_matrix[next_state])))

			# elif mode == 'softmax':
			# 	tau = lwargs.get('tau', 0.01)
			# 	values = np.exp(Q_matrix[current_state] / tau)
		 #        values = values / np.sum(values)

		 #        sample = alpha*(reward + gamma*(np.random.choice(actions, p=values))

			Q_matrix[current_state, action] = (1 - alpha)*Q_matrix[current_state, action]+sample


		elif mode == 'ef':
			k = lwargs.get('k', 10)
			# exploratory function
			pass

		else:
			raise Exception("No mode for Q-Learning named %s found")%(mode)

	def createFunction(alg):
		if alg == 'Q' and alg == 'SARSA':
			return __Q

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
    def __init__(self, name, interface, **kwargs):
        self._compute_score = 0
        self.actions = kwargs.get('actions', 1000)
        self.limit = kwargs.get('limit', False)
        self.Q_matrix = kwargs.get('Q_matrix', None)
        self.alg = kwargs.get('alg', 'Q')
        self.interface = None
        self.player_name = name

        self.vfunction_args = {
			'mode':kwargs.get('mode', 'simple')
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

        self.__attach_interface(interface)

    def compute_info(self):
        return {
            'player_actions_number':self.actions,
            'player_limited_actions':self.limit,
            'value_function': self.vfunction.__name__,
            'Q_matrix': self.Q_matrix,
            'player_name': self.player_name
        }


    def __attach_interface(self, interface):
    	interface.connect_player(self)
    	self.interface = interface

    def compute_state(self, state):
        return int(state.sum())

class PlayerGameInterface(object):
    def __init__(self):
        self.player = None
        self.game = None

    def connect_player(self, player):
        self.player = player

    def connect_game(self, game):
        self.game = game

    def ask_game(self, resource, query, current_state):
        resource  = self.game.__dict__.get(resource, None)
        if resource:
            result = resource.get(query, None)(current_state)
            return result

    def update_state(self, next_state, reward):
        self.player.update(next_state, reward)
        self.player.make_action()

    def compute_info(self):
        return self.player.compute_info()

    def is_conected(self):
        return self.player and self.game