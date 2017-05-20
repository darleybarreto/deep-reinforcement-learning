from flappy_bird_pygame import flappybird
from dqn import create_model
import os


if __name__ == '__main__':
	save_dqn_path = None
	save_txt_path = None
	# save_dqn_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_files', 'dqn_model.pickle')
	# save_txt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_files', 'socres.txt')
	# txt = open(save_txt_path,"w")

	episode = 0
	episodes = 5000
	shape, fully_connected, actions = flappybird.flappy_bird_model()

	flappybird_main = flappybird.init_main(save_dqn_path, create_model(actions, shape, fully_connected, path=save_dqn_path))
	
	while episode < episodes:
		print("Beginning episode #%s"%episode)
		score = flappybird_main()
		episode += 1
		# txt.write(score + " ")

	# txt.close() 