# from flappy_bird_pygame import flappybird
from games import *
from dqn import create_model
from tqdm import trange
import os
import sys

games ={1:["FlappyBird", flappybird],\
		2:["Pong",pong],\
		3:["Catcher",catcher],\
		4:["WaterWorld",waterworld],\
		5:["Snake",snake],\
		6:["PuckWorld",puckworld],\
		7:["Pixelcopter",pixelcopter],\
		8:["Quit"]}


def menu():
	while True:
		os.system('clear')
		print("Select an option:")
		
		for k in games:
			print(k, games[k][0])

		op = input()
		
		try:
			op = int(op)
		except Exception as e:
			continue

		choice = games.get(op,None)
		
		if not choice:
			continue

		if choice[0] == "Quit":
			os.system('clear')
			sys.exit()
		
		break
	os.system('clear')
	return choice

if __name__ == '__main__':
	
	model = menu()
	
	save_dqn_path = model[0] + "_dqn_model.pickle"
	save_txt_path = model[0] + "_scores.txt"

	save_dqn_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_files', save_dqn_path)
	save_txt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_files', save_txt_path)
	load_dqn_path = None
	# save_txt_path = None
	txt = open(save_txt_path,"w")
	show_display = False
	episodes = 5000
	shape, fully_connected, actions = model[1].build_model()

	game_main = model[1].init_main(save_dqn_path, create_model(actions, shape, fully_connected, path=load_dqn_path),display=show_display)
	
	# while episode < episodes:
	for episode in trange(episodes,desc='Episodes'):
		# print("Beginning episode #%s"%episode)
		score = game_main()
		txt.write(str(score) + " ")

	txt.close()
	# os.system("shutdown now -h")