import numpy as np
from game2048.model256 import CNN256
from game2048.model512 import CNN512
from game2048.model1024 import CNN1024
from game2048.model2048 import CNN2048
from game2048.agents import Agent
from torchvision import transforms
import torch 

cnn256 = CNN256()
cnn512 = CNN512()
cnn1024 = CNN1024()
cnn2048 = CNN2048()
map_location=lambda storage,loc: storage

cnn256.load_state_dict(torch.load('game2048/256cnn_params.pkl',map_location=map_location))
cnn512.load_state_dict(torch.load('game2048/512cnn_params.pkl',map_location=map_location))
cnn1024.load_state_dict(torch.load('game2048/1024cnn_params.pkl',map_location=map_location))
cnn2048.load_state_dict(torch.load('game2048/2048cnn_params.pkl',map_location=map_location))

map_table = {2**i: i for i in range(1,16)}
map_table[0] = 0
def OneHotEncoder(InputArr):
     codearr = np.zeros((16,4,4),dtype=float)
     for p in range(0,4):
           for q in range(0,4):
                codearr[map_table[int(InputArr[p,q])],p,q] = 1
     return codearr

class MyAgent(Agent):
    def __init__(self, game, display=None):
        super().__init__(game, display)

    def step(self):
        arr=OneHotEncoder(self.game.board)
        Input=np.expand_dims(arr,axis=0)
        arr_tensor=torch.from_numpy(Input).float()
        if self.game.score<256:
            output=cnn256(arr_tensor)
            direction = torch.max(output,1)[1]
        elif (self.game.score >=256) and (self.game.score <512):
            output=cnn512(arr_tensor)
            direction = torch.max(output,1)[1]
        elif (self.game.score >=512) and(self.game.score <1024):
            output=cnn1024(arr_tensor)
            direction = torch.max(output,1)[1]
        elif (self.game.score >=1024) and (self.game.score <2048):
            output=cnn2048(arr_tensor)
            direction = torch.max(output,1)[1]
        return direction
