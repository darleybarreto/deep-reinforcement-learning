import numpy as np
from ple import PLE
from ..util import *
from ple.games.flappybird import FlappyBird, K_w

possible_actions = {0: K_w, 1: None}

def init_main(save_path, model, train=True):
    """The application's entry point.

    If someone executes this module (instead of importing it, for
    example), this function is called.
    """
    push_to_memory, select_action, perform_action, optimize, save_model = model

    reward_alive = 0

    fps = 30  # fps we want to run at
    frame_skip = 2
    num_steps = 1
    force_fps = False  # slower speed
    display_screen = True
    
    game = FlappyBird()
    
    p = PLE(game, fps=fps, frame_skip=frame_skip, num_steps=num_steps,
        force_fps=force_fps, display_screen=display_screen)

    p.init()

    def flappy_bird_action(action):
        # reward, action
        return p.act(action)
    
    def main():
        nonlocal reward_alive

        reward = 0

        x_t = extract_image(p.getScreenRGB(),(80,80))

        stack_x = np.stack((x_t, x_t, x_t, x_t), axis=0)

        while p.game_over() == False:        

            x_t = extract_image(p.getScreenRGB(),(80,80))
            
            x_t = np.reshape(x_t, (1, 80, 80))

            st = np.append(stack_x[:3, :, :], x_t, axis=0)

            play(flappy_bird_action, st, select_action, perform_action, possible_actions)
                        
            if train:
                r, action = train_and_play(flappy_bird_action, st, select_action, perform_action, possible_actions, optimize)
                reward += r
                push_to_memory(stack_x, action, st, reward)
            
            else:
                play(flappy_bird_action, st, select_action, perform_action, possible_actions)
            
            stack_x = st

            reward_alive += 0.1
            reward += reward_alive

        score = p.score()
        p.reset_game()
        reward -= 10
        save_model(save_path)
        return score

    return main


def build_model():
    # The input of the first layer corresponds to
    # the number of most recent frames stacked together as describe in the paper
    shape = [
                [4, 32, 8, 4],    # first layer
                [32,64,4,2],      # second layer
                [64,64,3,1],      # third layer
            ]

    fully_connected = [2304,512]

    return shape, fully_connected, len(possible_actions)