from tfe_lib.game2048  import Game2048
from tfe_lib.player2048 import Player2048 as Player
from rl.rl import PlayerGameInterface as PGI

game = '2048'
# algorithms = ['simple_q', 'egreedy_q', 'ef_q','sarsa']

interface = PGI()

p = Player('simple_q_2048',
			action_type=game,
			shape_Q=(192,4),
			mode='simple',
			interface=interface)

# players = {i + "_" + game: Player(i + "_" + game, action_type=game, shape_Q=(176,4)) for i in algorithms}

game = Game2048('gui')(interface)
game.start_game()
# while True:
# 	p = Player(actions=100)
# 	gamegrid = GameGrid(p)
	
# 	for i in range(1000):
# 		gamegrid.start_game()
# 	gamegrid.save_data()
# 	del gamegrid