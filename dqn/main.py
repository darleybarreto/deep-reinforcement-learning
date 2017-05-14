from flappy_bird_pygame import flappybird
from dqn import create_model
import os

if __name__ == '__main__':
	save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_files')
	
	episodes = 0
	learning_rate = 
	shape = [
			[],
			[],
			[],
			]

	while True:
		print("Beginning episode #%s"%episodes)
		flappybird.main(save_path, create_model(shape, learning_rate))
		episodes += 1