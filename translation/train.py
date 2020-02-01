import torch
from dataset import *
from model import *
from torch import optim
import torch.nn as nn
import random
SOS_token = 0
EOS_token = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device=torch.device('cpu')
teacher_forcing_ratio=0.5
import time
import math
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def train(input_tensor,target_tensor,encoder,decoder,\
          encoder_optimizer,decoder_optimizer,\
          criterion,max_length=MAX_LENGTH):
    encoder_hidden=encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length=input_tensor.size(0)
    target_length=target_tensor.size(0)
    # print('input_length',input_length)
    # print('target tensor',target_length)

    encoder_outputs=torch.zeros(max_length,encoder.hidden_size,device=device)

    loss=0

    for ei in range(input_length):
        encoder_output,encoder_hidden=encoder(input_tensor[ei],encoder_hidden)
        encoder_outputs[ei]=encoder_output[0,0] #1x1x256
        #encoder_outputs[ei]=vector of 256 dim

    decoder_input=torch.tensor([[SOS_token]],device=device)

    decoder_hidden=encoder_hidden

    use_teacher_forcing=True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for di in range(target_length):
            # decoder_output,decoder_hidden=decoder(decoder_input,decoder_hidden)
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss+=criterion(decoder_output,target_tensor[di])
            decoder_input=target_tensor[di]
    else:
        for di in range(target_length):
            # decoder_output,decoder_hidden=decoder(decoder_input,decoder_hidden)
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv,topi=decoder_output.topk(1)
            decoder_input=topi.squeeze().detach()

            loss+=criterion(decoder_output,target_tensor[di])
            if decoder_input.item()==EOS_token:
                break
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()/target_length
1


def trainIters(encoder,decoder,n_iters,print_every=1000,plot_every=100,lr=0.01):
    start=time.time()
    plot_losses=[]
    print_loss_total=0
    plot_loss_total=0

    encoder_optimizer=optim.SGD(encoder.parameters(),lr=lr)
    decoder_optimizer=optim.SGD(decoder.parameters(),lr=lr)
    training_pairs=[tensorFromPair(random.choice(pairs),input_lang,output_lang)
                    for i in range(n_iters)]

    criterion=nn.NLLLoss()
    '''
    NLLoss输入是一个在C个类别上的概率分布，还有一个目标类别索引
    一般要在网络上加上logsoftmax层，如果不想加可以直接使用crossentropy()
    
    '''

    for iter in range(1,n_iters+1):
        training_pair=training_pairs[iter-1]
        input_tensor=training_pair[0]

        target_tensor=training_pair[1]



        loss=train(input_tensor,target_tensor,encoder,
                   decoder,encoder_optimizer,decoder_optimizer,criterion)

        print_loss_total+=loss
        plot_loss_total+=loss

        if iter % print_every==0:
            print_loss_avg=print_loss_total/print_every
            print_loss_total=0

            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))


        if iter%plot_every==0:
            plot_loss_avg=plot_loss_total/plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total=0

    showPlot(plot_losses)


'''train'''
print(device)
input_lang,output_lang,pairs=prepareData('eng','fra',True)
hidden_size=256
encoder=EncoderRNN(input_lang.n_words,hidden_size).to(device)
attn_decoder=AttnDecoderRNN(hidden_size,output_lang.n_words,dropout_p=0.1).to(device)
# decoder=DecoderRNN(hidden_size,output_lang.n_words).to(device)
trainIters(encoder,attn_decoder,75000,print_every=5000)
torch.save(encoder,'encoder.pt')
torch.save(attn_decoder,'attn_deocer.pt')

