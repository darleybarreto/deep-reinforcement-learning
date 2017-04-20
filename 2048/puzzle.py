from tkinter import *
from logic import *
import os
import pyautogui

SIZE = 500
GRID_LEN = 4
GRID_PADDING = 10

BACKGROUND_COLOR_GAME = "#92877d"
BACKGROUND_COLOR_CELL_EMPTY = "#9e948a"
BACKGROUND_COLOR_DICT = {   2:"#eee4da", 4:"#ede0c8", 8:"#f2b179", 16:"#f59563", \
                            32:"#f67c5f", 64:"#f65e3b", 128:"#edcf72", 256:"#edcc61", \
                            512:"#edc850", 1024:"#edc53f", 2048:"#edc22e" }
CELL_COLOR_DICT = { 2:"#776e65", 4:"#776e65", 8:"#f9f6f2", 16:"#f9f6f2", \
                    32:"#f9f6f2", 64:"#f9f6f2", 128:"#f9f6f2", 256:"#f9f6f2", \
                    512:"#f9f6f2", 1024:"#f9f6f2", 2048:"#f9f6f2" }
FONTGRID = ("Verdana", 40, "bold")
FONTHEADER = ("Verdana", 30)
KEY_UP_ALT = "\'\\uf700\'"
KEY_DOWN_ALT = "\'\\uf701\'"
KEY_LEFT_ALT = "\'\\uf702\'"
KEY_RIGHT_ALT = "\'\\uf703\'"

KEY_UP = "'Up'"
KEY_DOWN = "'Down'"
KEY_LEFT = "'Left'"
KEY_RIGHT = "'Right'"

class Player(object):
    
    ACTION = pyautogui.press

    def __init__(self, alg='Q', Q_matrix=None, actions=1000, limit=False, **kwargs):
        self._compute_score = 0
        self.actions = actions
        self.limit = limit
        self.Q_matrix = Q_matrix
        self.alpha = kwargs.get('alpha', default=0.5)
        self.gamma = kwargs.get('gamma', default=1.)

        self.vfunction = ValueFunctionFactory(self.Q_matrix,\
            shape_Q=kwargs.get('shape_Q', default=False),\
            alpha=self.alpha,\
            gamma=self.gamma,\
        ).createFunction(alg)(
            mode = kwargs.get('mode', default=None),\
            epsilon = kwargs.get('epsilon', default=None),\
            ef = kwargs.get('ef', default=None)
            )


    def make_action(self):
        pass
        # Player.ACTION('left')

    def update_state(self, env, reward, terminal_state):
        # self._compute_score = score
        # self.make_action()
        # if terminal_state:
        #     if terminal_state = "Win!":
        #         pass

        #     else:
        #         pass

        print(env)

    def save_date(self, file_name):
        logging = {
            'player_actions_number':self.actions,
            'player_limited_actions':self.limit,
            'value_function': self.vfunction.__name__,
            'Q_matrix': self.Q_matrix,
            'is_epsilon': bool(self.epsilon)
        }

        np.save(file_name, logging)


class GameGrid(Frame):
    def __init__(self, player=None):
        Frame.__init__(self)
        self._end_game = False
        self.game_number = 0
        self.player = player
        self.score = 0
        self.actual_reward = 0
        self.max_given_reward = 0

        self.grid()
        self.master.title('2048')
        self.master.bind("<Key>", self.key_down)

        self.backround = Frame(self, bg=BACKGROUND_COLOR_GAME, width=SIZE, height=SIZE)
        self.backround.grid()

        score_label = Label(master=self.backround,bg=BACKGROUND_COLOR_GAME, text="Score:", font=FONTHEADER, width=6, height=1)
        score_label.grid()

        self.ncompute_score_label = Label(master=self.backround,bg=BACKGROUND_COLOR_GAME, text=str(self.score), font=FONTHEADER, width=6, height=1)
        self.ncompute_score_label.grid()
        
        

        #self.gamelogic = gamelogic
        self.commands = {   KEY_UP: up, KEY_DOWN: down, KEY_LEFT: left, KEY_RIGHT: right,
                            KEY_UP_ALT: up, KEY_DOWN_ALT: down, KEY_LEFT_ALT: left, KEY_RIGHT_ALT: right }

        self.grid_cells = []
        self.init_grid()
        self.init_matrix()
        self.update_grid_cells()

        self.update_and_notify()

    def init_grid(self):

        table = Frame(self.backround, bg=BACKGROUND_COLOR_GAME, width=SIZE, height=SIZE)
        table.grid()
        for i in range(GRID_LEN):
            grid_row = []
            for j in range(GRID_LEN):
                cell = Frame(table, bg=BACKGROUND_COLOR_CELL_EMPTY, width=SIZE/GRID_LEN, height=SIZE/GRID_LEN)
                cell.grid(row=i, column=j, padx=GRID_PADDING, pady=GRID_PADDING)
                t = Label(master=cell, text="", bg=BACKGROUND_COLOR_CELL_EMPTY, justify=CENTER, font=FONTGRID, width=4, height=2)
                t.grid()
                grid_row.append(t)

            self.grid_cells.append(grid_row)

    def gen(self):
        return randint(0, GRID_LEN - 1)

    def init_matrix(self):
        self.matrix = new_game(4)

        self.matrix=add_two(self.matrix)
        self.matrix=add_two(self.matrix)

    def update_grid_cells(self):
        for i in range(GRID_LEN):
            for j in range(GRID_LEN):
                new_number = self.matrix[i][j]
                if new_number == 0:
                    self.grid_cells[i][j].configure(text="", bg=BACKGROUND_COLOR_CELL_EMPTY)
                else:

                    self.grid_cells[i][j].configure(text=str(new_number), bg=BACKGROUND_COLOR_DICT[new_number], fg=CELL_COLOR_DICT[new_number])
        self.update_idletasks()
    
    def update_and_notify(self):
        self.ncompute_score_label['text'] = str(self.score)
        if not self._end_game:
            if self.player:
                self.actual_reward = self.score - self.actual_reward

        self.player.update_state(self.matrix,actual_reward,self._end_game)

    def finish(self, result):
        self._end_game = result

        self.quit()

    def start_game(self):
        self.game_number += 1
        self.mainloop()

    def compute_score(self, before_state):
        self.matrix[np.where( self.matrix != before_state)]
        # self.score +=  int(np.sum(self.matrix[np.where( self.matrix > 2)]))

    def key_down(self, event):
        key = repr(event.keysym)
        if key in self.commands:
            before_state = self.matrix
            self.matrix, done = self.commands[repr(event.keysym)](self.matrix)
            # self.compute_score(before_state)
            if done:
                self.matrix = add_two(self.matrix)
                self.update_grid_cells()
                done=False

                result = game_state(self.matrix)
                if result:
                    self.grid_cells[1][1].configure(text="You",bg=BACKGROUND_COLOR_CELL_EMPTY)
                    self.grid_cells[1][2].configure(text=result,bg=BACKGROUND_COLOR_CELL_EMPTY)

                    self.finish(result)
                self.update_and_notify()

    def save_data(self, file_name):
        self.player.save_data(file_name)