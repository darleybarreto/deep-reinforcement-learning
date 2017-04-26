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
		Q_matrix = {}

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
		if current_state not in Q_matrix:
			Q_matrix[current_state] = {i:r for i, r in enumerate(np.random.randn(4))}

		if next_state not in Q_matrix:
			Q_matrix[next_state] = {i:r for i, r in enumerate(np.random.randn(4))}

		mode = lwargs.get('mode', 'simple')
		if not mode== 'ef':

			if mode == "simple":
				sample = alpha*(reward + gamma*(np.amax(list(Q_matrix[next_state].values()))))

			elif mode == "greddy":
				# e-greedy
				epsilon = lwargs.get('epsilon', 0.001)
				if np.random.uniform(0, 1) <= epsilon:
					sample = alpha*(reward + gamma*(np.random.choice(list(Q_matrix[next_state].values()))))

				else:
					sample = alpha*(reward + gamma*(np.amax(list(Q_matrix[next_state].values()))))

			# elif mode == 'softmax':
			# 	tau = lwargs.get('tau', 0.01)
			# 	values = np.exp(Q_matrix[current_state] / tau)
		 #        values = values / np.sum(values)

		 #        sample = alpha*(reward + gamma*(np.random.choice(actions, p=values))

			Q_matrix[current_state][action] = (1 - alpha)*Q_matrix[current_state][action]+sample
			# print(Q_matrix[current_state][action])

		elif mode == 'ef':
			k = lwargs.get('k', 10)
			# exploratory function
			pass

		else:
			raise Exception("No mode for Q-Learning named %s found")%(mode)

	def createFunction(alg):
		if alg == 'Q' or alg == 'SARSA':
			return __Q

		else:
			raise Exception("No algorithm named %s found"%(alg))

	return createFunction, Q_matrix

def ActionFactory():
	
	def action2048(action, mode):
		mapping = {0:'up',\
			1:'down',\
			2:'left',\
			3:'right'
		}
		if mode == 'text':
			return "'" + mapping[action] +"'"
		elif mode =='gui':
			pyautogui.press(mapping[action])

		else:
			raise Exception("No mode for action 2048 named %s found"%(mode))

	def createAction(action_type):
		if action_type == '2048':
			return action2048
		else:
			raise Exception("No action function named %s found"%(action_type))

	return createAction
