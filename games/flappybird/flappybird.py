import torch
import numpy as np
from ple import PLE
from ..util import *
from ple.games.flappybird import FlappyBird, K_w
from torch.autograd import Variable

possible_actions = {0: K_w, 1: None}

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def init_main(save_path, model, train=True,display=False):

    push_to_memory, select_action, perform_action, optimize, save_model = model

    fps = 30  # fps we want to run at
    frame_skip = 2
    num_steps = 1
    force_fps = False  # slower speed

    game = FlappyBird()
    
    p = PLE(game, fps=fps, frame_skip=frame_skip, num_steps=num_steps,
        force_fps=force_fps, display_screen=display)

    p.init()

    def p_action(action):
        # reward, action
        return p.act(action)
    
    def main(steps):

        frames_steps = 40
        reward_alive = 0

        x_t = extract_image(p.getScreenRGB(),(80,80))

        steps = steps * frames_steps

        stack_x = np.stack((x_t, x_t, x_t, x_t), axis=0)

        while p.game_over() == False and steps > 0:         
            try:
                steps -= 1

                x_t = extract_image(p.getScreenRGB(),(80,80))
                
                x_t = np.reshape(x_t, (1, 80, 80))

                st = np.append(stack_x[1:4, :, :], x_t, axis=0)

                if train:
                    reward, action, _, _, _ = train_and_play(p_action, st, select_action, perform_action, possible_actions, optimize,None,{})
                    
                    reward_alive += 0.1
                    reward += reward_alive
                    
                    push_to_memory(stack_x, action, st, reward)
                
                else:
                    play(p_action, st, select_action, perform_action, possible_actions, None,{})
                
                stack_x = st

            except Exception as e:
                print("Exception >>", e)
                print("Saving model")
                save_model(save_path)
                raise Exception


        with open("stackFrames.txt","wb") as f:
            np.savetxt(f,np.column_stack(stack_x),fmt='%1.10f')

        score = p.score()
        p.reset_game()
        if train: save_model(save_path)
        return score

    return main


def a3c_main(save_path, shared_model,\
            model,\
            select_action,\
            perform_action,\
            save_model,\
            optimizer=None,\
            train=True,\
            display=False,\
            gamma =.99,\
            tau=1.):

    fps = 30  # fps we want to run at
    frame_skip = 2
    num_steps = 1
    force_fps = False  # slower speed
    
    game = FlappyBird()
    
    p = PLE(game, fps=fps, frame_skip=frame_skip, num_steps=num_steps,
        force_fps=force_fps, display_screen=display)

    p.init()

    def p_action(action):
        # reward, action
        return p.act(action)
    
    def main(lstm_shape, steps):      
        reward_alive = 0
        values = []
        log_probs = []
        rewards = []
        entropies = []
        frames_step = 40
        steps =  steps * frames_step

        x_t = extract_image(p.getScreenRGB(),(80,80))

        stack_x = np.stack((x_t, x_t, x_t, x_t), axis=0)
        model.load_state_dict(shared_model.state_dict())

        cx = Variable(torch.zeros(1, lstm_shape[-1]))
        hx = Variable(torch.zeros(1, lstm_shape[-1]))

        try:
            while p.game_over() == False and steps > 0:        
                steps -= 1

                x_t = extract_image(p.getScreenRGB(),(80,80))
                
                x_t = np.reshape(x_t, (1, 80, 80))

                st = np.append(stack_x[1:4, :, :], x_t, axis=0)
                            
                if train:
                    reward, action, hx, cx, info_dict = train_and_play(p_action, st,\
                                                        select_action, perform_action,\
                                                        possible_actions, opt_nothing, \
                                                        model, {"isTrain":True, "hx":hx,"cx":cx})
                    reward_alive += 0.1
                    reward += reward_alive
                    rewards.append(reward)
                    # reward += r

                    entropies.append(info_dict["entropies"])
                    values.append(info_dict["values"])
                    log_probs.append(info_dict["log_probs"])

                else:
                    _, _, hx, cx, _ = play(p_action, st, select_action,\
                        perform_action, possible_actions, model, {"hx":hx,"cx":cx, "isTrain":False})
                
                stack_x = st
            np.save("stack.txt",stack_x)
            if train:
                state = torch.from_numpy(stack_x)
                R = torch.zeros(1, 1)
                if steps > 0:
                    value, _, _ = model((Variable(state.unsqueeze(0).float()), (hx, cx)))

                values.append(Variable(R))
                policy_loss = 0
                value_loss = 0
                R = Variable(R)
                gae = torch.zeros(1, 1)

                for i in reversed(range(len(rewards))):
                    R = gamma * R + rewards[i]
                    advantage = R - values[i]
                    value_loss = value_loss + 0.5 * advantage.pow(2)

                    # Generalized Advantage Estimataion
                    delta_t = rewards[i] + gamma * \
                        values[i + 1].data - values[i].data
                    gae = gae * gamma * tau + delta_t

                    policy_loss = policy_loss - \
                        log_probs[i] * Variable(gae) - 0.01 * entropies[i]

                optimizer.zero_grad()

                (policy_loss + 0.5 * value_loss).backward()
                torch.nn.utils.clip_grad_norm(model.parameters(), 40)

                ensure_shared_grads(model, shared_model)
                optimizer.step()
        
        except Exception as e:
            print("Exception >>", e)
            print("Saving model")
            save_model(save_path)

        score = p.score()
        p.reset_game()
        if train: save_model(shared_model,save_path)
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