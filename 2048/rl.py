import numpy as np
from random import random

def ValueFunctionFactory(Q=None, **kwargs):
	'''
	v_factory = ValueFunctionFactory(Q=Q_m, alpha=0.75, gamma=0.9)
	
	Q_learning = v_factory.createFunction('Q')
	SARSA = v_factory.createFunction('SARSA')

	v_factory_r = ValueFunctionFactory(shape_Q=(5,5), alpha=0.75, gamma=0.9)
	Q_m = v_factory_r.returnQ_matrix()

	Q_learning_r = v_factory_r.createFunction('Q')
	SARSA_r = v_factory_r.createFunction('SARSA')
	'''

	alpha = kwargs.get('alpha', default=0.5)
	gamma = kwargs.get('gamma', default=1.)

	if not kwargs['Q']: 
		if kwargs.get('shape_Q', default=False):
			n_states, n_actions =  kwargs['shape_Q']
			if not (type(n_states) == int and type(n_actions) == int):
				raise Exception("Both n_states and n_actions must be int")
		else:
			raise Exception("shape_Q not found while Q is None")

		Q_matrix = np.random.rand(n_states, n_actions)

	else:
		n_states, n_actions =  kwargs['Q'].shape
		Q_matrix = kwargs['Q']
		del kwargs['Q']


	def __Q(actual_state, next_state, action, **lwargs):

		mode = lwargs.get('mode', default=None)
		epsilon = lwargs.get('epsilon', default=None)
		ef = lwargs.get('ef', default=None)

		if not mode:

		elif mode == "epsilon":
			# e-greedy
			if random() <= epsilon:
				sample = 0
			else:
				sample = 0

		elif mode == 'ef':
			# exploration function
			pass

	def __SARSA():


	def createFunction(alg):
		if alg == 'Q':
			return __Q

		elif alg == 'SARSA':
			return __SARSA

		else:
			raise Exception("No algorithm named %s found")%(alg)

	def returnQ_matrix():
		return Q_matrix