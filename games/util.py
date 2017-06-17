import cv2
import os

def extract_image(image_data, size):
    # get the image as a numpy array
    # image_data = pygame.surfarray.array3d(display_surface)
    # pygame.image.save(display_surface, "screenshot.jpg")
    # resizing the image and change color to grayscale
    # x_t = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
    x_t = cv2.cvtColor(cv2.resize(image_data, size), cv2.COLOR_BGR2GRAY)
    # x_t = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)

    # The threshold function applies fixed-level thresholding to a single-channel array
    # threshold(array,threshold,maximum value,thresholding type)
    _, x_t = cv2.threshold(x_t,50,255,cv2.THRESH_BINARY)
    return x_t

def play(f_action, state, select_action, perform_action, possible_actions, model, kwargs):
    hx = kwargs.get("hx", None)
    cx = kwargs.get("hx", None)
    isT = kwargs.get("isTrain", False)
    action, hx, cx, info_dict = select_action(state, hx, cx, model, isT)
    reward = perform_action(f_action, possible_actions, action)
    return reward, action, hx, cx, info_dict

def train_and_play(f_action, state, select_action, perform_action, possible_actions, optimize, model,kwargs):
    reward, action, hx, cx, info_dict = play(f_action, state, select_action, perform_action, possible_actions,model,kwargs)
    optimize()
    return reward, action, hx, cx, info_dict