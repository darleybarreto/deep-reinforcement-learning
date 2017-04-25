from game2048  import Game2048, pickle
from player2048 import Player2048 as Player
from rl.player_lib import PlayerGameInterface as PGI

_game = '2048'
name = 'simple_q_2048'
# interface = PGI()

# p = Player(name,
# 			interface=interface,
# 			is_training=True,
# 			action_type=game,
# 			mode='simple')

# game = Game2048()('text')(interface)
# game.start_game()

for i in range(10):
	print("Episode %d"%(i))
	interface = PGI(episode=i)
	
	if i == 0:
		Q_matrix = None

	else:
		with open(name + '.pickle', 'rb') as handle:
			Q_matrix = pickle.load(handle)['Q_matrix']

   	# training
	print("Training")
	p = Player(name,
				Q_matrix=Q_matrix,
				interface=interface,
				is_training=True,
				action_type=_game,
				mode='simple')

	game = Game2048()('text')(interface)
	game.start_game()
	print("End training")
	print("Saving agent")

	with open(name + '.pickle', 'rb') as handle:
		Q_matrix = pickle.load(handle)['Q_matrix']

	print("Agent saved")

	for j in range(10):
		# playing
		print("Playing game #%d in episode %d"%(i,j))
		p = Player(name,
				Q_matrix=Q_matrix,
				interface=interface,
				is_training=False,
				action_type=_game,
				mode='simple')

		game.new_game()
		game.start_game()