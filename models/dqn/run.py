from .dqn import create_model
from tqdm import trange
import os
import sys

sys.path.insert(0,'../..')
from utils import menu

def train_test(opt, model, save_dqn, load_dqn, save_txt_path, **kwargs):
	txt = open(save_txt_path,"w")
	
	if opt == "train":
		show_display = kwargs.get("display", False)
		episodes = float(kwargs.get("episodes",5000))
		observations = float(kwargs.get("step",100))

		shape, fully_connected, actions = model[1].build_model()
		dqn = create_model(actions, shape, fully_connected, path=load_dqn)
		
		game_main = model[1].init_main(save_dqn, dqn, observations,display=show_display)
		
		# while episode < episodes:
		if episodes == float("inf"):
			while True:
				print("Beginning episode #%s"%episode)
				score = game_main()
				txt.write(str(score) + " ")

		else:
			for episode in trange(int(episodes),desc='Episodes'):
				# print("Beginning episode #%s"%episode)
				score = game_main()
				txt.write(str(score) + " ")

	elif opt == "test":
		pass

	txt.close()

def main(game, opt, **kwargs):
	save_dqn_path = game[0] + "_dqn_model.pickle"
	save_txt_path = game[0] + "_dqn_scores.txt"

	save_dqn_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_files', save_dqn_path)
	save_txt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_files', save_txt_path)
	load_dqn_path = None
	# save_txt_path = None
	train_test(opt, game, save_dqn_path, load_dqn_path, save_txt_path, **kwargs)
	# os.system("shutdown now -h")

if __name__ == '__main__':
	game, opt = menu()
	main(game, opt)