import random
from dataset import *
import torch

def randomChoice(l):
    return l[random.randint(0,len(l)-1)]

def randomTrainPair():
    category=randomChoice(all_categories)
    line=randomChoice(category_lines[category])
    return category,line

def categoryTensor(category):
    li=all_categories.index(category)
    tensor=torch.zeros(1,n_categories)
    tensor[0][li]=1
    return tensor

def inputTensor(line):
    tensor=torch.zeros(len(line),1,n_letters)
    for li in range(len(line)):
        letter=line[li]
        tensor[li][0][all_letters.find(letter)]=1
    return tensor

def targetTensor(line):
    letter_indexes=[all_letters.find(line[li]) for li in range(1,len(line))]
    letter_indexes.append(n_letters-1) # EOS
    return torch.LongTensor(letter_indexes)

def randomTrainingExample():
    category,line=randomTrainPair()
    category_tensor=categoryTensor(category)
    input_line_tensor=inputTensor(line)
    target_line_tensor=targetTensor(line)
    return category_tensor,input_line_tensor,target_line_tensor

randomTrainingExample()

