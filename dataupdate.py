import numpy as np
from game import Game
from displays import Display, IPythonDisplay
from agents import Agent, RandomAgent, ExpectiMaxAgent
from expectimax import board_to_move

f=open('test11.txt','w+')
for i in range(2):
    game = Game(4,score_to_win = 2048,random=False)
    search_func = board_to_move
    while(game.end == 0):
         suggest = search_func(game.board)
         board = game.board
         ARRS = []
         
         for i in range(4):
              jointsFrame = board[i] #每行
              ARRS.append(jointsFrame)
              for Ji in range(4):
                   strNum = str(jointsFrame[Ji])
                   f.write(strNum)
                   f.write(' ')
              f.write('\n')
         f.write(str(suggest))
         f.write('\n')
         game.move(suggest)   
f.close()

