from tkinter import *
from logic2048 import *
from ZODB import FileStorage, DB
import transaction


KEY_UP_ALT = "\'\\uf700\'"
KEY_DOWN_ALT = "\'\\uf701\'"
KEY_LEFT_ALT = "\'\\uf702\'"
KEY_RIGHT_ALT = "\'\\uf703\'"

KEY_UP = "'Up'"
KEY_DOWN = "'Down'"
KEY_LEFT = "'Left'"
KEY_RIGHT = "'Right'"

storage = FileStorage.FileStorage('DB/qstorage.fs')
db = DB(storage)

class Game2048(object):

    def __init__(self, game_number, shape=4,interface=None):
        self.shape = shape
        self.wins = 0
        self.game_number = 0
        self.loses = 0
        self.score = 0
        self.previous_score = 0
        self.interface = interface
        self.attach_interface(interface)
        self.commands = {   KEY_UP: up, KEY_DOWN: down, KEY_LEFT: left, KEY_RIGHT: right,
                            KEY_UP_ALT: up, KEY_DOWN_ALT: down, KEY_LEFT_ALT: left, KEY_RIGHT_ALT: right }
        

    def __call__(self, mode):
        self.mode = mode

        if mode == 'text':
            return GameText2048

        elif mode == 'gui':
            return GameGui2048
    
    def init_matrix(self):
        self.matrix = new_game_matrix(self.shape)

        self.matrix=add_two(self.matrix)
        self.matrix=add_two(self.matrix)

    def update_and_notify(self):
        finished = False
        if self.interface and self.interface.is_conected():
            if self.mode == 'text':
                while not finished:
                    re = self.previous_score
                    self.previous_score = self.score
                    finished = self.action_made(self.interface.update_state(self.matrix, self.previous_score - re - 10))
            
            else:
                re = self.previous_score
                self.previous_score = self.score
                self.ncompute_score_label['text'] = str(self.score)
                self.interface.update_state(self.matrix, self.previous_score - re - 10)

    def action_made(self, action):
        raise Exception("Not implemented")

    def save_data(self, file_name=None):
        if self.interface.is_conected():
            data_player = self.interface.compute_info()
            # data_player.update({'wins': self.wins, 'loses':self.loses})

            if not file_name:
                # if self.interface.is_training():
                file_name = data_player['player_name']

                # else:
                #     file_name = data_player['player_name'] +\
                #     '_episode_'+ str(data_player['episode']) + '_game_'\
                #     +str(self.game_number)
            connection = db.open()
            root = connection.root()
            root['player'] = data_player
            transaction.commit()
            connection.close()
            db.close()
            storage.close()
            # with open(file_name + '.qmatrix', 'wb') as handle:
            #     dill.dump(data_player, handle)

    def compute_score(self, score_earned):
        self.score +=  score_earned

    def finish(self, result):
        if result == "Win!": self.wins += 1
        else: self.loses += 1

        return self.keyinterrupt(None)


    def keyinterrupt(self, event):
        # self.save_data()
        if self.mode=='gui': self.quit()
        return True

    def attach_interface(self, interface):
        if interface: interface.connect_game(self)
        self.interface = interface



class GameText2048(Game2048):
    def __init__(self, game_number, interface=None):
        super(GameText2048, self).__init__(game_number, interface=interface)
        self.mode = 'text'

    def new_game(self):
        self.score = 0
        self.init_matrix()
        self.update_and_notify()
        self.game_number += 1

    def start_game(self):
        self.new_game()

    def action_made(self, action):
        # for a after action in text mode
        self.matrix, done, score_earned = self.commands[action.title()](self.matrix)
        # print(score_earned)
        self.compute_score(score_earned)
        finished = False
        if done:
            self.matrix = add_two(self.matrix)
            done=False
            result = game_state(self.matrix)
            if result:
                finished = self.finish(result)
        # print(self.score)
        return finished
        # self.update_and_notify()


class GameGui2048(Game2048, Frame):

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



    def __init__(self, game_number, interface=None):
        super(GameGui2048, self).__init__(game_number, interface=interface)
        self.mode = 'gui'

        Frame.__init__(self)
        self.grid()
        self.master.title('2048')
        self.master.bind("<Key>", self.action_made)
        self.master.bind('<Control-c>', self.keyinterrupt)

        self.background = None
        self.ncompute_score_label = 0

        self.new_game()


    def new_game(self):
        self.init_matrix()
        self.game_number += 1
        if self.background:
            self.background.grid_forget()
            self.background.destroy()

        self.background = Frame(self, bg=GameGui2048.BACKGROUND_COLOR_GAME, width=GameGui2048.SIZE, height=GameGui2048.SIZE)
        self.background.grid()

        score_label = Label(master=self.background,bg=GameGui2048.BACKGROUND_COLOR_GAME, text="Score:", font=GameGui2048.FONTHEADER, width=6, height=1)
        score_label.grid()

        self.ncompute_score_label = Label(master=self.background,bg=GameGui2048.BACKGROUND_COLOR_GAME, text=str(self.score), font=GameGui2048.FONTHEADER, width=6, height=1)
        self.ncompute_score_label.grid()

        self.score = 0
        self.ncompute_score_label['text'] = str(self.score)
        self.grid_cells = []
        self.init_grid()
        self.update_grid_cells()

        self.update_and_notify()

    def init_grid(self):

        table = Frame(self.background, bg=GameGui2048.BACKGROUND_COLOR_GAME, width=GameGui2048.SIZE, height=GameGui2048.SIZE)
        table.grid()
        for i in range(GameGui2048.GRID_LEN):
            grid_row = []
            for j in range(GameGui2048.GRID_LEN):
                cell = Frame(table, bg=GameGui2048.BACKGROUND_COLOR_CELL_EMPTY, width=GameGui2048.SIZE/GameGui2048.GRID_LEN, height=GameGui2048.SIZE/GameGui2048.GRID_LEN)
                cell.grid(row=i, column=j, padx=GameGui2048.GRID_PADDING, pady=GameGui2048.GRID_PADDING)
                t = Label(master=cell, text="", bg=GameGui2048.BACKGROUND_COLOR_CELL_EMPTY, justify=CENTER, font=GameGui2048.FONTGRID, width=4, height=2)
                t.grid()
                grid_row.append(t)

            self.grid_cells.append(grid_row)

    def update_grid_cells(self):
        for i in range(GameGui2048.GRID_LEN):
            for j in range(GameGui2048.GRID_LEN):
                new_number = self.matrix[i][j]
                if new_number == 0:
                    self.grid_cells[i][j].configure(text="", bg=GameGui2048.BACKGROUND_COLOR_CELL_EMPTY)
                else:

                    self.grid_cells[i][j].configure(text=str(new_number), bg=GameGui2048.BACKGROUND_COLOR_DICT[new_number], fg=GameGui2048.CELL_COLOR_DICT[new_number])
        self.update_idletasks()

    def start_game(self):
        self.mainloop()

    def action_made(self, event):
        key = repr(event.keysym)
        if key in self.commands:
            self.matrix, done, score_earned = self.commands[key](self.matrix)
            self.compute_score(score_earned)
            if done:
                self.matrix = add_two(self.matrix)
                self.update_grid_cells()
                done=False
                result = game_state(self.matrix)
                if result:
                    self.grid_cells[1][1].configure(text="You",bg=GameGui2048.BACKGROUND_COLOR_CELL_EMPTY)
                    self.grid_cells[1][2].configure(text=result,bg=GameGui2048.BACKGROUND_COLOR_CELL_EMPTY)
                    self.finish(result)
            self.update_and_notify()