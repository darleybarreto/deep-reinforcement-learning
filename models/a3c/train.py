from .a3c import create_model
from tqdm import trange
from .shared_opt import SharedAdam

def train(rank, shared_model, optz, save_a3c, load_a3c, save_txt_path, game, kwargs):
	
	A3CModel = create_model(game.build_model_a3c())
	select_action, perform_action, a3cmodel,save_model = A3CModel

	a3cmodel.eval()

	episodes = float(kwargs.get("episodes",5000))
	show_display = kwargs.get("display", False)
	steps = float(kwargs.get("step",1000))

	if load_a3c:
		mode = "a"
	else:
		mode = "w"

	# txt = open(save_txt_path, mode)

	game_main = game.a3c_main(save_a3c,\
								shared_model,\
								a3cmodel,\
								steps,\
								select_action,\
								perform_action,\
								save_model,\
								optimizer=optz,\
								train=False,\
								display=show_display)
		
	if episodes == float("inf"):
		episode = 0
		while True:
			print("Beginning episode #%s"%episode)
			score, rewards = game_main()
			# txt.write(str(score) + " ")
			episode += 1

	else:
		for episode in trange(int(episodes),desc='Episodes'):
			# print("Beginning episode #%s"%episode)
			score = game_main()
			# txt.write(str(score) + " ")
		
	