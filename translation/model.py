import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from dataset import MAX_LENGTH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device=torch.device('cpu')
class EncoderRNN(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(EncoderRNN,self).__init__()
        self.hidden_size=hidden_size
        '''
        input(*)： a list of indices
        output(*,H): where * is the input shape and H=embedding dim
        首先初始化embedding层得到词嵌入矩阵 embedding
        然后在forward部分输出包含索引的list得到word vectoe输入到gru中
        '''
        self.embedding=nn.Embedding(input_size,hidden_size)
        self.gru=nn.GRU(hidden_size,hidden_size)

    def forward(self, input,hidden):
        # print("....encoder.....")
        # print('input size',input.size())
        embedded=self.embedding(input).view(1,1,-1)
        # print("embedded size",embedded.size())
        # print('init hidden size',hidden.size())
        output=embedded
        output,hidden=self.gru(output,hidden)
        # print('encoder output size',output.size()) 1x1x256
        return output,hidden

    def initHidden(self):
        return torch.zeros(1,1,self.hidden_size,device=device)


class DecoderRNN(nn.Module):
    def __init__(self,hidden_size,output_size):
        super(DecoderRNN,self).__init__()
        self.hidden_size=hidden_size

        self.embedding=nn.Embedding(output_size,hidden_size)
        self.gru=nn.GRU(hidden_size,hidden_size)
        self.out=nn.Linear(hidden_size,output_size)
        self.softmax=nn.LogSoftmax(dim=1)

    def forward(self,input,hidden):
         output=self.embedding(input).view(1,1,-1)

         output=F.relu(output)
         output,hidden=self.gru(output,hidden)
         output=self.out(output[0])
         output=self.softmax(output) # output=  1 x output_lang_n_words
         # softmax returns a tensor of the same dimension and shapes as the input with values in the range[0,1]
         # print("decoder.....")
         # print('output-size',output.size())
         # print('hidden size',hidden.size())
         return output,hidden

    def initHidden(self):
        return torch.zeros(1,1,self.hidden_size,device=device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        alignment_scores=self.attn(torch.cat((embedded[0], hidden[0]), 1))#embedded[0]=1x256,hidden[0]=1x256
        # alignment_scores [1,max_length]

        attn_weights = F.softmax(alignment_scores, dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),#1x1x max_length
                                 encoder_outputs.unsqueeze(0))#1xmax_lengthxhidden_size

        # attn_applied=context vector 1 x max length x hidden size
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)








