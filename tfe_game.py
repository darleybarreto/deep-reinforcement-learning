from tkinter import *
from tfe_logic import *
from rl.rl import Player
from random import random, randint
import time

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


class Player2048(Player):

    mapping = {0:"'Up'",\
            1:"'Down'",\
            2:"'Left'",\
            3:"'Right'"
        }

    def __init__(self, name, **kwargs):
        super(Player2048, self).__init__(name, **kwargs)
        self.current_state = None
        self.current_action = None

    def ask_max_reward(self, current_state):
        '''
        Ask for the reward
        '''
        action_to_do = None
        max_reward = 0

        for action in [0, 1, 2, 3]:
            next_state, is_done, reward = self.interface.ask_game('commands',\
                                            Player2048.mapping[action],\
                                            current_state)
            if reward == max_reward:
                if isinstance(action_to_do, int):
                    if random() < 0.5:
                        action_to_do = action
                
                else:
                    action_to_do = randint(0,3)

            elif reward > max_reward:
                action_to_do = action
                max_reward = reward

        return action_to_do, max_reward

    def make_action(self):
        time.sleep(2)
        if self.alg == "Q":
            action_to_do, reward = self.ask_max_reward(self.current_state)
        
        elif self.alg == "SARSA":
            pass

        self.current_action = action_to_do
        self.present_reward = reward
        print("Choice: " + Player2048.mapping[self.current_action])
        self.action_function(action_to_do)

    def update(self, next_state, reward):
        if self.current_state is None:
            self.current_state = next_state
        else:

            past = self.compute_state(self.current_state)
            self.current_state = next_state
            now = self.compute_state(self.current_state)

            self.vfunction(past, now, self.current_action, reward, **self.vfunction_args)

    def compute_info(self):
        return {
            'player_actions_number':self.actions,
            'player_limited_actions':self.limit,
            'value_function': self.vfunction.__name__,
            'Q_matrix': self.Q_matrix,
            'player_name': self.player_name
        }     


class GameGrid2048(Frame):
    def __init__(self, interface=None):
        Frame.__init__(self)
        self.wins = 0
        self.loss = 0
        self.game_number = 0
        self.score = 0
        self.actual_reward = 0
        self.interface = None

        self.grid()
        self.master.title('2048')
        self.master.bind("<Key>", self.key_down)
        self.master.bind('<Control-c>', self.keyinterrupt)

        self.background = None
        self.ncompute_score_label = 0

        self.commands = {   KEY_UP: up, KEY_DOWN: down, KEY_LEFT: left, KEY_RIGHT: right,
                            KEY_UP_ALT: up, KEY_DOWN_ALT: down, KEY_LEFT_ALT: left, KEY_RIGHT_ALT: right }
        self.__attach_interface(interface)
        self.new_game()

    def new_game(self):
        if self.background:
            self.background.grid_forget()
            self.background.destroy()

        self.background = Frame(self, bg=BACKGROUND_COLOR_GAME, width=SIZE, height=SIZE)
        self.background.grid()

        score_label = Label(master=self.background,bg=BACKGROUND_COLOR_GAME, text="Score:", font=FONTHEADER, width=6, height=1)
        score_label.grid()

        self.ncompute_score_label = Label(master=self.background,bg=BACKGROUND_COLOR_GAME, text=str(self.score), font=FONTHEADER, width=6, height=1)
        self.ncompute_score_label.grid()

        self.score = 0
        self.ncompute_score_label['text'] = str(self.score)
        self.grid_cells = []
        self.init_grid()
        self.init_matrix()
        self.update_grid_cells()

        self.update_and_notify()

    def init_grid(self):

        table = Frame(self.background, bg=BACKGROUND_COLOR_GAME, width=SIZE, height=SIZE)
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

        if self.interface and self.interface.is_conected():
            self.actual_reward = self.score - self.actual_reward
            self.interface.update_state(self.matrix, self.actual_reward)

    def finish(self, result):
        self.game_number += 1

        if result == "Win!": self.wins += 1
        elif result == "Lose!": self.loss += 1

        self.new_game()

    def start_game(self):
        self.mainloop()

    def compute_score(self, score_earned):
        self.score +=  score_earned


    def key_down(self, event):
        key = repr(event.keysym)
        if key in self.commands:
            self.matrix, done, score_earned = self.commands[repr(event.keysym)](self.matrix)
            self.compute_score(score_earned)
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

    def save_data(self, file_name=None):
        if self.interface.is_conected():
            data_player = self.interface.compute_info()
            data_player.update({'game_number': self.game_number,\
                'wins': self.wins,\
                'loss': self.loss
                })
            if not file_name: file_name = data_player['player_name']
            np.save(file_name, data_player)

    def keyinterrupt(self, event):
        self.save_data()
        self.quit()

    def __attach_interface(self, interface):
        if interface: interface.connect_game(self)
        self.interface = interface