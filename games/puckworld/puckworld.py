import torch
import numpy as np
from ple import PLE
from ..util import *
from ple.games.puckworld import PuckWorld, K_w, K_a, K_s, K_d
from torch.autograd import Variable

possible_actions = {0:None}

possible_actions.update({i:a for i,a in enumerate([K_w, K_a, K_s, K_d], start=1)})

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def init_main(save_path, model, steps, train=True, display=False):
    """The application's entry point.

    If someone executes this module (instead of importing it, for
    example), this function is called.
    """
    push_to_memory, select_action, perform_action, optimize, save_model = model

    fps = 30  # fps we want to run at
    frame_skip = 2
    num_steps = 1
    force_fps = False  # slower speed
    
    game = PuckWorld(width=256, height=256)
    
    p = PLE(game, fps=fps, frame_skip=frame_skip, num_steps=num_steps,
        force_fps=force_fps, display_screen=display)

    p.init()

    def p_action(action):
        # reward, action
        return p.act(action)
    
    def main():
        nonlocal steps

        reward = 0

        x_t = extract_image(p.getScreenRGB(),(80,80))

        stack_x = np.stack((x_t, x_t, x_t, x_t), axis=0)

        while p.game_over() == False and steps > 0:         
            steps -= 1         

            x_t = extract_image(p.getScreenRGB(),(80,80))
            
            x_t = np.reshape(x_t, (1, 80, 80))

            st = np.append(stack_x[1:4, :, :], x_t, axis=0)
                        
            if train:
                r, action, _, _, _ = train_and_play(p_action, st, select_action, perform_action, possible_actions, optimize,None,{})
                reward += r
                push_to_memory(stack_x, action, st, reward)
            
            else:
                play(p_action, st, select_action, perform_action, possible_actions, None,{})
            
            stack_x = st

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

def a3c_main(save_path, shared_model,\
            model,\
            steps,\
            select_action,\
            perform_action,\
            save_model,\
            optimizer=None,\
            train=True,\
            display=False):



    fps = 30  # fps we want to run at
    frame_skip = 2
    num_steps = 1
    force_fps = False  # slower speed
    
    game = PuckWorld(width=256, height=256)
    
    p = PLE(game, fps=fps, frame_skip=frame_skip, num_steps=num_steps,
        force_fps=force_fps, display_screen=display)

    p.init()

    def p_action(action):
        # reward, action
        return p.act(action)
    
    def main():
        nonlocal steps

        rewards = []
        values = []
        log_probs = []
        rewards = []
        entropies = []

        # reward = 0
        x_t = extract_image(p.getScreenRGB(),(80,80))

        stack_x = np.stack((x_t, x_t, x_t, x_t), axis=0)
        model.load_state_dict(shared_model.state_dict())

        cx = Variable(torch.zeros(1, 256), volatile=True)
        hx = Variable(torch.zeros(1, 256), volatile=True)

        while p.game_over() == False and steps > 0:        
            steps -= 1

            x_t = extract_image(p.getScreenRGB(),(80,80))
            
            x_t = np.reshape(x_t, (1, 80, 80))

            st = np.append(stack_x[1:4, :, :], x_t, axis=0)
                        
            if train:
                r, action, hx, cx, info_dict = train_and_play(p_action, st,\
                                                    select_action, perform_action,\
                                                    possible_actions, opt_nothing, \
                                                    model, {"isTrain":True, "hx":hx,"cx":cx})
                rewards.append(r)
                # reward += r

                entropies.append(info_dict["entropies"])
                values.append(info_dict["values"])
                log_probs.append(info_dict["log_probs"])

            else:
                _, _, hx, cx, _ = play(p_action, st, select_action,\
                    perform_action, possible_actions, model, {"hx":hx,"cx":cx, "isTrain":False})
            
            stack_x = st

        if isTrain:
            state = torch.from_numpy(stack_x)
            R = torch.zeros(1, 1)
            if steps > 0:
                value, _, _ = model((Variable(state.unsqueeze(0)), (hx, cx)))

            values.append(Variable(R))
            policy_loss = 0
            value_loss = 0
            R = Variable(R)
            gae = torch.zeros(1, 1)

            for i in reversed(range(len(rewards))):
                R = args.gamma * R + rewards[i]
                advantage = R - values[i]
                value_loss = value_loss + 0.5 * advantage.pow(2)

                # Generalized Advantage Estimataion
                delta_t = rewards[i] + args.gamma * \
                    values[i + 1].data - values[i].data
                gae = gae * args.gamma * args.tau + delta_t

                policy_loss = policy_loss - \
                    log_probs[i] * Variable(gae) - 0.01 * entropies

            optimizer.zero_grad()

            (policy_loss + 0.5 * value_loss).backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 40)

            ensure_shared_grads(model, shared_model)
            optimizer.step()

        score = p.score()
        p.reset_game()
        # reward -= 1
        # save_model(save_path)
        return score, rewards

    return main

def build_model_a3c():
    # The input of the first layer corresponds to
    # the number of most recent frames stacked together as describe in the paper
    shape = [
                [4, 32, 5, 1, 2],    # first layer
                [32, 32, 5, 1, 1],    # second
                [32, 64, 4, 1, 1],   # third layer
                [64, 64, 3, 1, 1],   # fourth layer
            ]

    lstm = [1024, 512]
    fully_connected = [512,1]

    return shape, fully_connected, lstm, len(possible_actions)

def opt_nothing():
    pass