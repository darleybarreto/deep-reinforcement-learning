from .dqn import create_model
from tqdm import trange
import os
import sys

sys.path.insert(0, '../..')
from utils import menu


def train_test(opt, model, save_dqn, load_dqn, save_txt_path, **kwargs):
    shape, fully_connected, actions = model[1].build_model()
    dqn = create_model(actions, shape, fully_connected, path=load_dqn)
    episodes = float(kwargs.get("episodes", 5000))
    observations = float(kwargs.get("steps", 1000))

    if load_dqn:
        mode = "a"
    else:
        mode = "w"

    if opt == "train":
        txt = open(save_txt_path, kwargs.get("access_scores", mode))

        show_display = kwargs.get("display", False)
        game_main = model[1].init_main(save_dqn, dqn, display=show_display)

        if episodes == float("inf"):
            episode = 0
            while True:
                print("Beginning episode #%s" % episode)
                score = game_main(observations)
                txt.write(str(score) + " ")
                episode += 1
                print("Ending episode with score %d" % (score))

        else:
            for episode in trange(int(episodes), desc='Episodes'):
                # print("Beginning episode #%s"%episode)
                score = game_main(observations)
                txt.write(str(score) + " ")

        txt.close()

    elif opt == "test":

        game_main = model[1].init_main(save_dqn, dqn, train=False, display=True)
        episode = 0

        while episode < episodes:
            episode += 1
            print("Beginning episode #%s" % episode)
            score = game_main(observations)
            print("Ending episode with score:", score)


def main(game, opt, **kwargs):
    save_dqn_path = game[0] + "_dqn_model.pickle"
    save_txt_path = game[0] + "_dqn_scores.txt"

    save_dqn_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_files', save_dqn_path)
    save_txt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_files', save_txt_path)

    if kwargs.get("load_path", False):
        load_dqn_path = save_dqn_path
    else:
        load_dqn_path = None

    train_test(opt, game, save_dqn_path, load_dqn_path, save_txt_path, **kwargs)


# os.system("shutdown now -h")

if __name__ == '__main__':
    game, opt = menu()
    main(game, opt)
