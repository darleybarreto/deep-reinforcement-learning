from game2048  import Game2048
from player2048 import Player2048 as Player
from rl.player_lib import PlayerGameInterface as PGI
import os
import dill as pickle

_game = '2048'
name = 'simple_q_2048'

# interface = PGI(episode=1)
# p = Player(name,
# 			interface=interface,
# 			is_training=True,
# 			action_type=_game,
# 			mode='greddy',
# 			alg='Q')

# game = Game2048(1,shape=4)('gui')(1,interface=interface)
# game.start_game()

n_episodes = 100
n_train = 100
n_plays = 10
alg = 'SARSA'
model = 'simple'
play_matches = {}

for i in range(n_episodes):
	play_matches.update({i:{'training': None, 'testing': None}})

	print("Episode %d"%(i))
	interface = PGI(episode=i)
	

	if i == 0:
		Q_matrix = None


	# else:
	# 	with open(name + '.pickle', 'rb') as handle:
	# 		Q_matrix = pickle.load(handle)['Q_matrix']

   	# training
	print("Beginning Training")
	game = Game2048(0)('text')(0, interface=interface)
	for j in range(n_train):
		# print(j)
		p = Player(name,
					alg=alg,
					Q_matrix=Q_matrix,
					interface=interface,
					is_training=True,
					action_type=_game,
					mode=model)
		game.start_game()
	play_matches[i]['training'] = {'wins': game.wins, 'loses': game.loses}

	# print("Ending training")
	Q_matrix = p.Q_matrix
	print("Beginning testing")
	game = Game2048(0)('text')(0, interface=interface)
	for j in range(n_plays):
		# print(j)
		# playing
		# print("Playing game #%d in episode %d"%(i,j))
		p = Player(name,
				alg=alg,
				Q_matrix=Q_matrix,
				interface=interface,
				is_training=False,
				action_type=_game,
				mode=model)

		game.start_game()
	# print("Ending testing")
	play_matches[i]['testing'] = {'wins': game.wins, 'loses': game.loses} 

print("Saving agent")
# with open(name + '.pickle', 'wb') as handle:
# 	pickle.dump(p, handle, protocol=pickle.HIGHEST_PROTOCOL)
game.save_data()
print("Agent saved")

print("Saving log")
with open('log' + '.pickle', 'wb') as handle:
	pickle.dump(play_matches, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Log saved")