from rl.function_lib import ValueFunctionFactory

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