from .a3c import create_model
from tqdm import trange
from .shared_opt import SharedAdam

def train(rank, model, optimizer, save_a3c, load_a3c, save_txt_path, **kwargs):
	
	A3CModel = create_model(actions, shape, fully_connected, path=load_a3c)
	model = A3CModel.model()
	model.train()

	episodes = float(kwargs.get("episodes",5000))
	show_display = kwargs.get("display", False)
	observations = float(kwargs.get("step",100))	
	show_display = kwargs.get("display", False)

	txt = open(save_txt_path,kwargs.get("access_scores", "w"))

	game_main = A3CModel.a3c_main(save_a3c, shared_model, model,\
						steps, train=False, display=show_display)
		
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
		
	txt.close()