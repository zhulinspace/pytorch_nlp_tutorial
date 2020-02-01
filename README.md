
# pytorch_nlp_tutorial

### classify name

预测名字属于哪一种语言

RNN由两个线性层组成。输入[input,hidden],输出[output,hidden]

每个timestamp 输入都是一个letter（eg.‘ a’），用one_hot vector表示,输出output是在所有语言类别的概率分布

![](https://github.com/zhulinspace/pytorch_nlp_tutorial/blob/master/img/classfy_name_network'.png)

输入序列（line),line ,eg for 'Adam' ,输入line需转化为line_tensor，其大小为[line_length x1 x n_letters]，多余的维度1是因为在pytorch中，所有数据都必须是batch_size的  ：That extra 1 dimension is because PyTorch assumes everything is in batches - we’re just using a batch size of 1 here. 



```python
    # 输入一个序列
    for i  in range(line_tensor.size()[0]):
        output,hidden=rnn(line_tensor[i],hidden)

    loss=criterion(output,category_tensor)
```



这里的时间长度T=Line_length,即会有Line_length个output,输入一个序列时，我们选取最后时间节点的output作为预测的语言类别，和真实的语言类别一起放入loss function中

train set: line_tensor ,target_category_tensor

由于是选取最后一个时间节点的output作为最后结果，即对一个epoch，只需要计算最后output和category的loss

### generating name

生成姓名，也可以生成句子。

在每个时间节点，输入（category,current letter, hidden state),输出(next letter,next hidden state),

![](https://github.com/zhulinspace/pytorch_nlp_tutorial/blob/master/img/generate_name_network.png)


所以我们的训练集必须包括以下：category，input_letters,target_letters

如下图line-e.g.for"Kasparov<EOS>"

input_letters如下图包含line的第一个letter到最后一个letter,而target_letters包含line的第二个letter到EOS结束符号

- category_tensor: 大小为[1 x n_categories],one_hot vector,
- input_line_tensor:大小为[line_length,1,n_letters] an array of one_hot vector
- target_line_tensor:为一维Longtensor,其大小为Line_length,元素为letter在all_letters的索引


![](https://github.com/zhulinspace/pytorch_nlp_tutorial/blob/master/img/generate_name_input_output.png)

因为有EOS符号，所以在处理训练集会有很多种不同的方法，这里选择处理方式如下，另input_tensor是定长的，而输出长度不是固定的。



```python
all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1 # Plus EOS marker
# One-hot matrix of first to last letters (not including EOS) for input
def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

# LongTensor of second letter to end (EOS) for target
def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1) # EOS
    return torch.LongTensor(letter_indexes)
```



我们在每一步时间节点都进行了预测，因此需要把每个时间节点的loss累积，而autograd可以累积每一步的loss并在最后后向传播即可计算梯度。每个时间节点的输出是在字典所有字母上的概率分布。


```python
def train(category_tensor,input_line_tensor,target_line_tensor):
    target_line_tensor.unsqueeze_(-1)#[line_length] ----> [line_length,1]
   
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
#以上为一个epoch的训练
```



generate部分：对每个时间节点产生的output串在一起就是生成的名字，步骤如下：

- Create tensors for input category, starting letter, and empty hidden state
- Create a string `output_name` with the starting letter
- Up to a maximum output length,
  - Feed the current letter to the network
  - Get the next letter from highest output, and next hidden state
  - If the letter is EOS, stop here
  - If a regular letter, add to `output_name` and continue
- Return the final name

### Translation

translate french-->english

字典L={ word in language,EOS,SOS}   EOS：结束符号 SOS:开始符号

pairs:[ line_countx2 ]

准备数据的完整过程：

1.读取txt,分割成lines,在将lines分成pairs

2.规范化文本，根据长度和内容进行过滤

3.从句子中得到字典(word2index,index2word)

然后可以准备训练数据，即每个pair里面的input和output都变成tensor，而且是以每个word在字典里的索引表示。而且最后都会加上eos token

注意一点由于引入了embedding层，即在不同时间节点输入到encoder中是一个字典索引，通过embeded层变成一个hidden_size维的特征向量，再输入到rnn中。

#### model

seq2seq结构：将input_tensor输入到encoder,可以得到一个context vector，理想情况下可以包含整个输入语句的含义，Decoder预测输出翻译后的语句

![](https://github.com/zhulinspace/pytorch_nlp_tutorial/blob/master/img/seq2seq.png)

下面是encoder部分的网络结构，其中gru可以换成LSTM

![](https://github.com/zhulinspace/pytorch_nlp_tutorial/blob/master/img/encoder-network.png)



下面是没有attention机制的decoder

![](https://github.com/zhulinspace/pytorch_nlp_tutorial/blob/master/img/decoder-network.png)



training:

input语句输入到encoder,即在所有的时间节点，encoder RNN对于输入语句每个词，每次都会产生output和最新的hidden state,然后把SOS token作为decoder的初始input输入，把encoder最新的hidden state（即context vector）作为decoder的初始hidden

"Teacher forcing"是利用真实的输出作为decoder的next input,而不是利用decoder自己的guess作为next input,这种方法可以让网络快速收敛。设置teacher_forcing_ratio 来表示多少使用比率



##### decoder based on attention

下面这张图是基于attention的decoder的网络结构，在自然语言处理中，注意力机制可以使模式对于输入的语句的不同部分赋予不同的权重，有利于decoder解码，同时这也更加符合认知，我们在翻译某个词的时候，观察输入语句不同部分的侧重使不一样的，如果仅仅使用普通的RNN作为decoder,则默认输入语句的不同部分权重一样。

note:下面这张图的网络结构是来自于pytorch tutorial,链接在文章最低端，其计算对齐分数的方式和下面讲解的都不同。

![](https://github.com/zhulinspace/pytorch_nlp_tutorial/blob/master/img/attention-decoder-network.png)

##### 注意力机制：

![](https://github.com/zhulinspace/pytorch_nlp_tutorial/blob/master/img/types.png)

Bahdanau和Luong最主要的区别是

1.alignment score计算方式

2.在decoder结构被引入注意力机制的位置，Luong在一开始就利用RNN，然后利用输出的new hidden state用于后续步骤中，而Bahdanau则是在最后面利用RNN,计算output

##### Bahdanan Attention

1. 在encoder每个时间节点记录encoder output

2. 计算对齐分数（alignment scores）

   假设hidden size是3，输入语句有两个word,即有两个encoder_output.

   则encoder_outputs :[input_length ,hidden_size]

   encoder outputs和decoder latest hidden state各自经过一个全连接层后cat,经过tanh函数，在乘以权重矩阵$W_{alignment}$得到Aligment scores vector ，每个元素是对不同的encoder output的分数。总的公式如下：
   $$
   score_{aligment}=W_{combined}⋅tanh(W_{decoder}⋅
   H_{decoder}+W_{encoder}⋅
   H_{encoder})
$$
   其实可以这样理解计算对齐分数的方式：拿所有encoder output里所有vector和decoder hidden state计算相似度，越相似的分数越高，即我们应该把注意力放在分数高的encoder outputs部分
   
3. 利用softmax层得到attention weights

4. 计算语义向量（ context vector）

   将encoder outputs乘以attention weights得到context vetcor

5. 解码并得到output

   将context vector和embedding of previous decoder output做cat,和previous decoder hidden一起feed到RNN中，得到new hidden state 后经过softmax得到new output(下面图中利用了全连接层得到概率分布，然后选择最大的概率得到类别索引)

6. 在每个时间节点重复步骤2-5，直到decoder输出eos token或者输出已经超过定义的最大长度 （max-length）

   

   下面这张图详细描述了具体过程
![](https://github.com/zhulinspace/pytorch_nlp_tutorial/blob/master/img/flow.png)


##### Luong attention

1.在encoder每个时间节点记录encoder output

2.解码得到new hidden state

在每个时间节点， 把先前decoder生成的hidden state 和 embeded output放入decoder RNN生成新的hidden state，若是初始decoder,其input可以是sos token,其hidden state 可以随机初始化，或用encoder最后一个时间节点的hidden state.其初始化方式有很多种

3.计算对齐分数

用decoder new hidden state 和encoder outputs计算对齐分数

Loung attention有三种计算注意力的方式：

- Dot ：encoder outputs和decoder hidden矩阵相乘
$$
  socre_{alignment}=H_{encoder} ⋅ H_{decoder}
$$
  
- general ：和dot类似,只不过加了一个权重矩阵
$$
  socre_{alignment}=W（H_{encoder} ⋅ H_{decoder}）
$$
  
- concat：这种方式和Bahdanau attention很类似只不过没有各自的参数矩阵，表明encoder outputs和decoder hidden state共享权重矩阵$W_{combined}$
$$
  score_{aligment}=W ⋅
  tanh(W_{combined}(H_{encoder}+H_{decoder}))
  $$



4. softmax对齐分数 同上

5. 计算语义向量 同上

6. 得到output

   语义向量和decoder new hidden state进行cat,然后送入线性层（作用和分类器最后一层类似）得到output

7. 重复2-6 ...同上

### Ref

[ https://blog.floydhub.com/attention-mechanism/ ]( https://blog.floydhub.com/attention-mechanism/ )

[ https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html ]( https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html )