from .a3c import create_model
from tqdm import trange
from .shared_opt import SharedAdam

def test(shared_model, save_a3c, load_a3c, **kwargs):
	
	A3CModel = create_model(actions, shape, fully_connected, path=load_a3c)
	model = A3CModel.model()
	model.eval()

	episodes = float(kwargs.get("episodes",5000))
	show_display = kwargs.get("display", False)
	
	game_main = A3CModel.a3c_main(save_a3c, shared_model, model, steps, train=False, display=show_display)

	episode = 0 

	while episode < episodes:
		episode += 1
		print("Beginning episode #%s"%episode)
		score = game_main()
		print("Ending episode with score:",score)