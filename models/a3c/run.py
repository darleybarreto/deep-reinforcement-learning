from .a3c import create_model
from tqdm import trange
import os
import sys

sys.path.insert(0,'../..')
from utils import menu

def train_test(opt, model, save_a3c, load_a3c, save_txt_path, **kwargs):
	txt = open(save_txt_path,"w")
	
	if opt == "train":
		show_display = kwargs.get("display", False)
		episodes = float(kwargs.get("episodes",5000))
		observations = float(kwargs.get("step",100))

		shape, fully_connected, actions = model[1].build_model()
		A3CModel = create_model(actions, shape, fully_connected, path=load_a3c)
		
		game_main = model[1].init_main(save_a3c, A3CModel, observations,display=show_display)
		
		# while episode < episodes:
		for episode in trange(episodes,desc='Episodes'):
			# print("Beginning episode #%s"%episode)
			score = game_main()
			txt.write(str(score) + " ")

	elif opt == "test":
		pass

	txt.close()


def main(model, opt, **kwargs):
	save_a3c = model[0] + "_a3c_model.pickle"
	save_txt_path = model[0] + "_a3c_scores.txt"

	save_a3c = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_files', save_a3c)
	save_txt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_files', save_txt_path)
	load_a3c = None
	# save_txt_path = None
	train_test(opt, model, save_a3c, load_a3c, save_txt_path, **kwargs)
	# os.system("shutdown now -h")

if __name__ == '__main__':
	model, opt = menu()
	main(model, opt)