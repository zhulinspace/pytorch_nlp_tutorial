from __future__ import unicode_literals,print_function,division
from io import open
import unicodedata
import string
import re
import random
import torch
device=torch.device("cuda"if torch.cuda.is_available() else 'cpu')
# device=torch.device('cpu')
SOS_token=0
EOS_token=1

class Lang:
    def __init__(self,name):
        '''

        :param name: 语言的名字
        '''
        self.name=name
        self.word2index={}
        self.word2count={}
        self.index2word={0:"SOS",1:'EOS'}
        self.n_words=2 # count SOS and EOS

    def addSentence(self,sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self,word):
        if word not in self.word2index:
            self.word2index[word]=self.n_words
            self.word2count[word]=1
            self.index2word[self.n_words]=word
            self.n_words+=1
        else:
            self.word2count[word]+=1

# Turn a unicode string to plain ASCii
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD',s)
        if unicodedata.category(c)!='Mn'
    )

# 变小写，strip,移除非字母以及非.?!符号，同时对.?!进行替换为' .'或' ?'或' !'
def normalizeString(s):
    # strip([chars]):移除字符串头尾指定的字符序列（chars）
    s=unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    # [.!?]匹配里面的字符 \1表示反向引用（从左往右数第一个左括号对应的内容）
    # [^m]匹配除m以外的字符 +表示一次或多次匹配
    # re.sub(pattern,repl,s) 根据给出的pattern在s中匹配并替换
    return s

# s='I ... am test??##'
# s = re.sub(r"([.!?])", r" \1", s)
# print(s) #I  . . . am test ? ?##
# s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
# print(s) #I . . . am test ? ?

'''
1.readlangs():
read data file,split into lines,split lines into pairs,and make lang instance ,normalize
2.filterPairs():
filter by length and content
3.lang.add_sentence
make word lists from sentences in pairs
'''
def readLangs(lang1,lang2,reverse=False):
    print("Reading lines...")

    # read file and split into lines
    lines=open('../data/%s-%s.txt'%(lang1,lang2),encoding='utf-8').\
        read().strip().split('\n')

    # split every line into pairs and normalize
    pairs=[[normalizeString(s) for s in l.split('\t')]for l in lines]

    # make lang instance
    if reverse:
        pairs=[list(reversed(p)) for p in pairs]
        input_lang=Lang(lang1)
        output_lang=Lang(lang2)
    else:
        input_lang=Lang(lang1)
        output_lang=Lang(lang2)
    return input_lang,output_lang,pairs

MAX_LENGTH=10
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def filterPair(p):
    return len(p[0].split(' '))<MAX_LENGTH and \
        len(p[1].split(' '))<MAX_LENGTH and \
        p[1].startswith(eng_prefixes)
        # 如果字符串以指定的前缀开始，则返回true，否则返回false


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


'''完整准备数据过程'''
def prepareData(lang1,lang2,reverse=False):
    input_lang,output_lang,pairs=readLangs(lang1,lang2,reverse)
    print("read %s sentence pairs"%len(pairs))
    pairs=filterPairs(pairs)
    print("TRimmed to %s sentence"%len(pairs))
    print("counting words ....")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("finish counting words")
    print(input_lang.name,input_lang.n_words)
    print(output_lang.name,output_lang.n_words)

    return input_lang,output_lang,pairs

# input_lang,output_lang,pairs=prepareData('eng','fra',True)
# print(random.choice(pairs))
# random.choice(seq) 返回列表，元组或字符串的随机项
'''
准备训练数据
对每个pair,都需要input tensor（indexes of the words in the input sentence） 和 target tensor(indexes of the words in the target sentence)
另外每个seq结尾都会加上EOS token
'''
def indexesFromSentence(lang,sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang,sentence):
    indexes=indexesFromSentence(lang,sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes,dtype=torch.long,device=device).view(-1,1)

def tensorFromPair(pair,input_lang,output_lang):
    input_tensor=tensorFromSentence(input_lang,pair[0])
    target_tensor=tensorFromSentence(output_lang,pair[1])
    return (input_tensor,target_tensor)
