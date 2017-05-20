import cv2

def extract_image(image_data, size):
    # get the image as a numpy array
    # image_data = pygame.surfarray.array3d(display_surface)
    # pygame.image.save(display_surface, "screenshot.jpg")
    # resizing the image and change color to grayscale
    x_t = cv2.cvtColor(cv2.resize(image_data, size), cv2.COLOR_BGR2GRAY)
    # x_t = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
    # The threshold function applies fixed-level thresholding to a single-channel array
    # threshold(array,threshold,maximum value,thresholding type)
    _, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)

    return x_t

def play(state, select_action, perform_action, possible_actions):
    action = select_action(state)
    # print(action)
    perform_action(possible_actions, action[0][0])
    return action

def train_and_play(state, select_action, perform_action, possible_actions, optimize):
    action = play(state, select_action, perform_action, possible_actions)
    optimize()
    return action
