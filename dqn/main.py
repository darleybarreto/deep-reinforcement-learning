from flappy_bird_pygame import flappybird
from dqn import create_model
import os


if __name__ == '__main__':
	save_path = None
	# save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_files', '')
	episodes = 0
	# this shape below needs fixing
	shape = [
			[8,8,4],
			[8,8,4],
			[8,8,4],
			]

	while True:
		print("Beginning episode #%s"%episodes)
		flappybird.main(save_path, create_model(shape, path=save_path))
		episodes += 1