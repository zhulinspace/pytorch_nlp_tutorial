import torch
from dataset import *
from helper import *
import torch.nn as nn
from model import RNN
import math
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

n_iters = 100000
print_every = 5000
plot_every = 500
all_losses = []
total_loss = 0 # Reset every plot_every iters
start=time.time()

rnn=RNN(n_letters,128,n_letters)
criterion=nn.NLLLoss()
lr=0.0005
optimizer=torch.optim.SGD(rnn.parameters(),lr=lr)

def train(category_tensor,input_line_tensor,target_line_tensor):
    target_line_tensor.unsqueeze_(-1)#[line_length] ----> [line_length,1]
    #a.squeeze(N)去掉a中指定维数N为1的维度
    #a.unsqueeze(N)给a指定维度N加上维数为1的维度
    hidden=rnn.initHidden()

    optimizer.zero_grad()

    loss=0

    for i  in range(input_line_tensor.size(0)):
        output,hidden=rnn(category_tensor,input_line_tensor[i],hidden)
        l=criterion(output,target_line_tensor[i])
        loss+=l

    loss.backward()

    optimizer.step()

    return output,loss.item()/input_line_tensor.size(0)

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

for iter in range(1,n_iters+1):
    output,loss=train(*randomTrainingExample())
    total_loss+=loss
    if iter % print_every ==0:
        print('%s (%d %d%%) %.4f'%(timeSince(start),iter,iter/n_iters*100,loss))

    if iter % plot_every==0:
        all_losses.append(total_loss/plot_every)
        total_loss=0


plt.figure()
plt.plot(all_losses)
plt.show()