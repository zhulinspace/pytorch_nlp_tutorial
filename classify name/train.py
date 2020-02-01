import torch
from dataset import *
import os
from model import *

import random
import time
import math

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


n_hidden=128
n_iters=100000
print_every=5000
plot_every=1000
learning_rate=0.005
'''network config'''
rnn=RNN(n_letters,n_hidden,n_categories)
criterion=nn.NLLLoss()
optimizer=torch.optim.SGD(rnn.parameters(),lr=learning_rate)

def categoryFromOutput(output):
    # output =1 x n_categories
    top_n,top_i=output.topk(1)
    # topk 用来求tensor中某个前k大或者前k小的值以及对应的index
    # type(top_i)=tensor e.g.[[0]]
    categories_i=top_i[0].item()
    return all_categories[categories_i],categories_i

'''get more training example'''

def randomChoice(l):
    #输入是一个list
    return l[random.randint(0,len(l)-1)]

def randomTrainingExample():
    category=randomChoice(all_categories)
    line=randomChoice(category_lines[category])
    category_tensor=torch.tensor([all_categories.index(category)],dtype=torch.long)
    # list index() 函数用于从列表中找出某个值第一个匹配项的索引位置
    line_tensor=lineToTensor(line)
    return category,line,category_tensor,line_tensor

def train(category_tensor,line_tensor):
    '''
    each loop of training will:
    - create input and target tensor
    - create a zeroed initial hidden state
    - read each letter in and
      - keep hidden state for next letter
    - back-propagate
    - return the output and loss
    '''
    hidden=rnn.initHidden()

    optimizer.zero_grad()

    # rnn.zero_grad()

    for i  in range(line_tensor.size()[0]):
        output,hidden=rnn(line_tensor[i],hidden)

    loss=criterion(output,category_tensor)
    loss.backward()

    # for p in rnn.parameters():
    #     p.data.add_(-learning_rate,p.grad.data)
    optimizer.step()

    return output,loss.item()

def timeSince(since):
    now=time.time()
    s=now-since
    m=math.floor(s/60)
    s-=m*60
    return '%dm %ds'% (m,s)


#--------------training part-------------
current_loss=0
all_losses=[]
start=time.time()

for iter in range(1,n_iters+1):
    category,line,category_tensor,line_tensor=randomTrainingExample()
    output,loss=train(category_tensor,line_tensor)
    current_loss+=loss

    if iter % print_every ==0:
        guess,guess_i=categoryFromOutput(output)
        correct='√' if guess==category else '×(%s)'% category
        print('%d %d%% (%s) %.4f %s / %s %s'
              %(iter,iter/n_iters*100,timeSince(start),loss,line,guess,correct))

        if iter % plot_every==0:
            all_losses.append(current_loss/plot_every)
            current_loss=0

print(all_losses)
plt.figure()
plt.plot(all_losses)

plt.show()
torch.save(rnn, 'char-rnn-classification.pt')

