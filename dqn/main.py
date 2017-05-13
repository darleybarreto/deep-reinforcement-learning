from flappy_bird_pygame import flappybird
import dqn
import os

if __name__ == '__main__':
	save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_files')

	while True:
		flappybird.main()