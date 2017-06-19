from .a3c import create_model
from tqdm import trange
from .shared_opt import SharedAdam
import torch

def test(shared_model, save_a3c, load_a3c, save_txt_path, game, kwargs):

	shape, fully_connected_shape, lstm_shape, possible_actions_shape = game.build_model_a3c()

	A3CModel = create_model((shape, fully_connected_shape, lstm_shape, possible_actions_shape))
	
	select_action, perform_action, a3cmodel,save_model = A3CModel

	a3cmodel.eval()

	episodes = float(kwargs.get("episodes",5000))
	show_display = kwargs.get("display", True)
	steps = kwargs.get("steps", 1000)

	if load_a3c:
		mode = "a"
		shared_model.load_state_dict(torch.load(load_a3c))
		
	else:
		mode = "w"

	txt = open(save_txt_path, mode)
	save_a3c = None
	game_main = game.a3c_main(save_a3c,\
								shared_model,\
								a3cmodel,\
								steps,\
								select_action,\
								perform_action,\
								save_model,\
								train=False,\
								display=show_display)

	episode = 0 

	while episode < episodes:
		episode += 1
		print("Shared model >> Beginning episode #%s"%episode)
		score = game_main(lstm_shape)
		txt.write(str(score) + " ")
		print("Shared model >> Ending episode with score:",score)