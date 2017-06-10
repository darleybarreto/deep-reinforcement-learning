import numpy as np
from random import randint

def new_game_matrix(n):
    matrix = np.zeros((n,n), dtype=np.uint64)

    return matrix

def add_two(mat):
    a=randint(0,len(mat)-1)
    b=randint(0,len(mat)-1)
    while(mat[a][b]!=0):
        a=randint(0,len(mat)-1)
        b=randint(0,len(mat)-1)
    mat[a][b]=2
    return mat


def game_state(mat):


    for i in range(len(mat)):
        for j in range(len(mat[0])):
            if mat[i][j]==2048:
                return 'Win!'

    for i in range(len(mat)):
        for j in range(len(mat[0])):
            if mat[i][j]==0:
                return ''

    for i in range(len(mat)-1): 
        for j in range(len(mat[0])-1):
            if mat[i][j]==mat[i+1][j] or mat[i][j+1]==mat[i][j]:
                return ''

    for k in range(len(mat)-1):
        if mat[len(mat)-1][k]==mat[len(mat)-1][k+1]:
            return ''
            
    for j in range(len(mat)-1):
        if mat[j][len(mat)-1]==mat[j+1][len(mat)-1]:
            return ''

    return 'Lose!'


def cover_up(mat):
    new=np.zeros(mat.shape, dtype=np.uint64)
    done=False
    for i in range(4):
        count=0
        for j in range(4):
            if mat[i][j]!=0:
                new[i][count]=mat[i][j]
                if j!=count:
                    done=True
                count+=1
    return new, done

def merge(mat):
    done=False
    earn = 0
    for i in range(4):
         for j in range(3):
             if mat[i][j]==mat[i][j+1] and mat[i][j]!=0:
                 mat[i][j]*=2
                 earn += mat[i][j]
                 mat[i][j+1]=0
                 done=True
    return mat, done, int(earn)


def up(game):
        game=game.transpose()
        game,done=cover_up(game)
        temp=merge(game)
        game=temp[0]
        done=done or temp[1]
        game=cover_up(game)[0]
        game=game.transpose()
        return game, done, temp[2]

def down(game):
        game=np.fliplr(game.transpose())
        game,done=cover_up(game)
        temp=merge(game)
        game=temp[0]
        done=done or temp[1]
        game=cover_up(game)[0]
        game=np.transpose(np.fliplr(game))
        return game, done, temp[2]

def left(game):
        game,done=cover_up(game)
        temp=merge(game)
        game=temp[0]
        done=done or temp[1]
        game=cover_up(game)[0]
        return game, done, temp[2]

def right(game):
        game=np.fliplr(game)
        game,done=cover_up(game)
        temp=merge(game)
        game=temp[0]
        done=done or temp[1]
        game=cover_up(game)[0]
        game=np.fliplr(game)
        return game, done, temp[2]