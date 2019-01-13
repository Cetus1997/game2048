import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import torch 
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
import numpy as np
from game2048.dataConfig import MyDataset
from torch.autograd import Variable
import torch.nn.functional as F


root = "./"

FILTERS=128
EPOCH = 100
LR = 0.0001
BATCH_SIZE = 256


train_data = MyDataset(txt =root+'data256.txt',transform = transforms.ToTensor())

train_loader = DataLoader(dataset = train_data,batch_size = BATCH_SIZE, shuffle = False)


class CNN256(nn.Module):
     def __init__(self):
          super(CNN256, self).__init__()
          self.conv1 = nn.Sequential(
                     nn.Conv2d(in_channels=16, out_channels=FILTERS, kernel_size=(4,1),stride=1, padding=0),
                     nn.ReLU(),#1*4*128
                     )
          self.conv2 = nn.Sequential(
                     nn.Conv2d(in_channels=16, out_channels=FILTERS, kernel_size=(1,4), stride=1, padding=0),
                     nn.ReLU(),#4*1*128
                     )
          self.conv3 = nn.Sequential(
            	     nn.Conv2d(in_channels=16, out_channels=FILTERS, kernel_size=(2,2), stride=1, padding=0),
                     nn.ReLU(), #outputsize[3*3*128]
          )
          self.conv4 = nn.Sequential(
                     nn.Conv2d(in_channels=16, out_channels=FILTERS, kernel_size=(3,3), stride=1, padding=0), 
                     nn.ReLU(),    #  2*2*128
          )
          self.conv5 = nn.Sequential(
                     nn.Conv2d(in_channels=16, out_channels=FILTERS, kernel_size=(4,4), stride=1, padding=0), #outputsize[1*1*128]
                     nn.ReLU(),
          )
          self.fc1 = nn.Linear(2816,512)
          self.fc2 = nn.Linear(512,128)
          self.fc3 = nn.Linear(128,4)
     def forward(self, x):
         m = nn.ReLU()
         conv41 = self.conv1(x)
         conv41 = conv41.view(conv41.size(0), -1)
         conv14 = self.conv2(x)
         conv14 = conv14.view(conv14.size(0), -1)
         conv22 = self.conv3(x)
         conv22 = conv22.view(conv22.size(0), -1)

         conv33 = self.conv4(x)
         conv33 = conv33.view(conv33.size(0), -1)
         conv44 = self.conv5(x)
         conv44 = conv44.view(conv44.size(0), -1)
         hidden=torch.cat((conv41,conv14,conv22,conv33,conv44),1)
         fc1 = self.fc1(hidden)
         fc1 = m(fc1)
         fc2 = self.fc2(fc1)
         fc2 = m(fc2)
         output = self.fc3(fc2)
         return output

cnn = CNN256()
cnn.cuda()
cnn.load_state_dict(torch.load('256cnn_params.pkl'))

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)



for epoch in range(EPOCH):
     print('epoch{}'.format(epoch+1))
     #training--------------------------
     train_loss = 0
     train_acc = 0
     for batch_x, batch_y in train_loader:
           batch_x = Variable(batch_x).cuda()
           batch_y = Variable(batch_y).cuda()
           output = cnn(batch_x.float())
           loss = loss_func(output,batch_y)
           train_loss+=loss.item()
           pred = torch.max(output,1)[1].cuda()
           train_correct = (pred == batch_y).sum()
           train_acc += train_correct.item()
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
     print('Train Loss: {:.6f}, Acc:{:.6f}'.format(train_loss/(len(train_data)), train_acc/(len(train_data))))
     torch.save(cnn.state_dict(),"256cnn_params.pkl")

'''

