from puzzle import GameGrid2048 as GameGrid
from puzzle import PlayerGameInterface as PGI
from rl.rl import Player2048 as Player

game = '2048'
algorithms = ['simple_q', 'egreedy_q', 'ef_q','sarsa']

interface = PGI()

players = {i + "_" + game: Player(i + "_" + game, action_type=game, shape_Q=(176,4)) for i in algorithms}

players['simple_q_2048'].attach_interface(interface)

gamegrid = GameGrid(interface)
gamegrid.start_game()
# while True:
# 	p = Player(actions=100)
# 	gamegrid = GameGrid(p)
	
# 	for i in range(1000):
# 		gamegrid.start_game()
# 	gamegrid.save_data()
# 	del gamegrid