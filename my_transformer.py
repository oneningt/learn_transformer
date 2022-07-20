from audioop import bias
from json import encoder
from turtle import forward

from click import password_option
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math

## 句子的输入部分，
sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']


# Transformer Parameters
# Padding Should be Zero
## 构建词表
src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4}
src_vocab_size = len(src_vocab)

tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6}
tgt_vocab_size = len(tgt_vocab)

src_len = 5 # length of source
tgt_len = 5 # length of target

## 模型参数
d_model = 512  # Embedding Size
d_ff = 2048  # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 6  # number of Encoder of Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()  ## 编码层
        self.decoder = Decoder()  ## 解码层
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False) ## 输出层 d_model 是我们解码层每个token输出的维度大小，之后会做一个 tgt_vocab_size 大小的softmax
    def forward(self, enc_inputs, dec_inputs):
        ## 这里有两个数据进行输入，一个是enc_inputs 形状为[batch_size, src_len]，主要是作为编码段的输入，一个dec_inputs，形状为[batch_size, tgt_len]，主要是作为解码端的输入

        ## enc_inputs作为输入 形状为[batch_size, src_len]，输出由自己的函数内部指定，想要什么指定输出什么，可以是全部tokens的输出，可以是特定每一层的输出；也可以是中间某些参数的输出；
        ## enc_outputs就是主要的输出，enc_self_attns这里没记错的是QK转置相乘之后softmax之后的矩阵值，代表的是每个单词和其他单词相关性；
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)

        ## dec_outputs 是decoder主要输出，用于后续的linear映射； dec_self_attns类比于enc_self_attns 是查看每个单词对decoder中输入的其余单词的相关性；dec_enc_attns是decoder中每个单词对encoder中每个单词的相关性；
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)

        ## dec_outputs做映射到词表大小
        dec_logits = self.projection(dec_outputs) # dec_logits : [batch_size x src_vocab_size x tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns


class Encoder(nn.Module):
    def __init__(self) -> None:
        super(Encoder, self).__init__()
        # 使用ModuleList对多个encoder进行堆叠，因为后续的encoder并没有使用词向量和位置编码，所以抽离出来
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)]) 
        # torch.nn.Embedding(numembeddings,embeddingdim)的意思是创建一个词嵌入模型，numembeddings代表一共有多少个词,
        # embedding_dim代表你想要为每个词创建一个多少维的向量来表示它
        # 字向量构建
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        # 位置编码
        self.pos_embedding = PositionalEncoding(d_model)

    def forward(self, enc_inputs):
        """
        enc_inputs: [batch_size, src_len]
        """
        enc_outputs = self.src_embedding(enc_inputs)  # [batch_size, src_len, d_model]
        enc_outputs = self.pos_embedding(enc_outputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, src_len, d_model]
        # Encoder输入序列的pad mask矩阵
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)  # [batch_size, src_len, src_len]
        enc_self_attns = []  # 在计算中不需要用到，它主要用来保存你接下来返回的attention的值（这个主要是为了你画热力图等，用来看各个词之间的关系
        for layer in self.layers:  # for循环访问nn.ModuleList对象
            # 上一个block的输出enc_outputs作为当前block的输入
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs,
                                               enc_self_attn_mask)  # 传入的enc_outputs其实是input，传入mask矩阵是因为你要做self attention
            enc_self_attns.append(enc_self_attn)  # 这个只是为了可视化
        return enc_outputs, enc_self_attns

class EncoderLayer(nn.Module):
    def __init__(self) -> None:
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        """
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]  mask矩阵(pad mask or sequence mask)
        """
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        # 第一个enc_inputs * W_Q = Q
        # 第二个enc_inputs * W_K = K
        # 第三个enc_inputs * W_V = V
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                               enc_self_attn_mask)  # enc_inputs to same Q,K,V（未线性变换前）
        enc_outputs = self.pos_ffn(enc_outputs)
        # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn

class Decoder(nn.Module):
    def __init__(self) -> None:
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)  # Decoder输入的embed词表
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])  # Decoder的blocks

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        """
        dec_inputs: [batch_size, tgt_len]
        enc_inputs: [batch_size, src_len]
        enc_outputs: [batch_size, src_len, d_model]   # 用在Encoder-Decoder Attention层
        """
        dec_outputs = self.tgt_emb(dec_inputs)  # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1) # [batch_size, tgt_len, d_model]
        # Decoder输入序列的pad mask矩阵（这个例子中decoder是没有加pad的，实际应用中都是有pad填充的）
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs) # [batch_size, tgt_len, tgt_len]
        # Masked Self_Attention：当前时刻是看不到未来的信息的
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs)# [batch_size, tgt_len, tgt_len]

        # Decoder中把两种mask矩阵相加（既屏蔽了pad的信息，也屏蔽了未来时刻的信息）
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask),
                                      0) # [batch_size, tgt_len, tgt_len]; torch.gt比较两个矩阵的元素，大于则返回1，否则返回0

        # 这个mask主要用于encoder-decoder attention层
        # get_attn_pad_mask主要是enc_inputs的pad mask矩阵(因为enc是处理K,V的，求Attention时是用v1,v2,..vm去加权的，要把pad对应的v_i的相关系数设为0，这样注意力就不会关注pad向量)
        #                       dec_inputs只是提供expand的size的
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)  # [batc_size, tgt_len, src_len]

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            # Decoder的Block是上一个Block的输出dec_outputs（变化）和Encoder网络的输出enc_outputs（固定）
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                             dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        # dec_outputs: [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attns, dec_enc_attns

class DecoderLayer(nn.Module):
    def __init__(self) -> None:
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        """
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        """
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs,
                                                        dec_self_attn_mask)  # 这里的Q,K,V全是Decoder自己的输入
        # dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs,
                                                      dec_enc_attn_mask)  # Attention层的Q(来自decoder) 和 K,V(来自encoder)
        dec_outputs = self.pos_ffn(dec_outputs)  # [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn, dec_enc_attn  # dec_self_attn, dec_enc_attn这两个是为了可视化的


class MultiHeadAttention(nn.Module):
    def __init__(self) -> None:
        super(MultiHeadAttention, self).__init__()
        """
        nn.Linear的用法
        y=wx+b
        __init__(self, in_features, out_features, bias=True)
        in_features:前一层网络神经元的个数
        out_features: 该网络层神经元的个数
        以上两者决定了weight的形状[out_features , in_features]
        bias: 网络层是否有偏置，默认存在，且维度为[out_features ],若bias=False,则该网络层无偏置。

        接下来看一下，输入该网络层的形状(N, *, in_features)，其中N为批量处理过成中每批数据的数量，
        *表示，单个样本数据中间可以包含很多维度，但是单个数据的最后一个维度的形状一定是in_features
        """
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        #
        self.W_O = nn.Linear(d_v * n_heads, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        """
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        """
        residual, batch_size = input_Q, input_Q.size(0)
        # 多头参数矩阵在一起进行线性变换，然后拆成多个头
        # B: batch_size, S:seq_len, D:dim
        # (B, S, D) -proj->(B, S, D_new) -split ->(B, S, Head, W) -trans ->(B, Head, S, W)
        """
        在pytorch中view函数的作用为重构张量的维度，相当于numpy中resize()的功能
        有的时候会出现torch.view(-1)或者torch.view(参数a，-1)这种情况
        a=torch.Tensor([[[1,2,3],[4,5,6]]])
        print(a.view(-1))  变一维结构
        tensor([1., 2., 3., 4., 5., 6.])

        a=torch.Tensor([[[1,2,3],[4,5,6]]])
        a=a.view(3,2)
        print(a)
        a=a.view(2,-1)
        print(a)
        tensor([[1., 2.],
        [3., 4.],
        [5., 6.]])
        tensor([[1., 2., 3.],
        [4., 5., 6.]])

        由上面的案例可以看到，如果是torch.view(参数a，-1)，
        则表示在参数b未知，参数a已知的情况下自动补齐列向量长度，在这个例子中a=2，tt3总共由6个元素，则b=6/2=3
        """
        # Q [batch_size, n_heads, len_q, d_k]
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        # K [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        # V [batch_size, n_heads, len_q, d_v]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        # 多头mask，所以mask矩阵扩充为4维
        # attn_mask:[batch_size, seq_len, seq_len] -> [batch_size, n_heads, seq_len, seq_len]
        """
        repeat 相当于一个broadcasting的机制
        a = torch.arange(4).reshape([2, 2])
        print(a)
        # tensor([[0, 1],
        #         [2, 3]]) 

        step1_a = a.repeat([2, 1])
        print(step1_a)
        # tensor([[0, 1],
        #         [2, 3],
        #         [0, 1],
        #         [2, 3]])
        """
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        # context: [batch_size, n_heads, len_q, d_v] attn_scores: [batch_size, n_heads, len_q, d_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        # 接下来拼接不同头的输出向量
        # context: [batch_size, n_heads, len_q, d_v] -> [batch_size, len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)

        # 全连接层使得多头attention的输出为seq_len * d_model
        output = self.W_O(context)  # [batch_size, len_q, d_model]
        return LayerNorm(d_model)(output + residual), attn

        

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask=None, dropout=None):
        scores = torch.matmul(Q, K.transpose(-1, -2) / np.sqrt(d_k))  # scaled ==> / np.sqrt(d_k)
        #  DotProductAttention  ==> torch.matmul(Q, K.transpose(-1, -2)
        #  masked
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        #  softmax 注意力分配 
        attn_scores = nn.Softmax(dim=-1)(scores)

        context = torch.matmul(attn_scores, V)

        return context, attn_scores


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000) -> None:
        super(PositionalEncoding, self).__init__()

        # dropout解释为:在训练过程的前向传播中，让每个神经元以一定概率p处于不激活的状态。以达到减少过拟合的效果。
        # pytorch中还表示使x每个位置的元素都有一定概率归0，以此来模拟现实生活中的某些频道的数据缺失，以达到数据增强的目的
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        # torch.range(start=1, end=6) 的结果是会包含end的，而torch.arange(start=1, end=6)的结果并不包含end
        # a.unsqueeze(1) 在索引1对应位置增加一个维度
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        #PE(pos,2i)=sin(pos/10000^(2i/dmodel))   PE(pos,2i+1)=cos(pos/10000^(2i/dmodel))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1) # question转置 异议

        self.register_buffer('pe', pe)
        # 这里值得注意的是调用了Module.register_buffer函数。这个函数的作用是创建一个buffer，比如这里把pi保存下来。
        # register_buffer通常用于保存一些模型参数之外的值，比如在BatchNorm中，我们需要保存running_mean(Moving Average)，
        # 它不是模型的参数(不用梯度下降)，但是模型会修改它，而且在预测的时候也要使用它。这里也是类似的，pe是一个提前计算好的常量，
        # 我们在forward要用到它。我们在构造函数里并没有把pe保存到self里，但是在forward的时候我们却可以直接使用它(self.pe)。
        # 如果我们保存(序列化)模型到磁盘的话，PyTorch框架也会帮我们保存buffer里的数据到磁盘，这样反序列化的时候能恢复它们

    #question
    def forward(self, x):
        x = x + self.pe[:x.size(0), :] 
        return self.dropout(x)


"""
FFN(x)=max(0,xW1+b1)W2+b2
"""
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self) -> None:
        super(PoswiseFeedForwardNet, self).__init__()
        """
        Sequential与ModuleList的区别
        1.nn.Sequential内部实现了forward函数，因此可以不用写forward函数。而nn.ModuleList则没有实现内部forward函数。
        2.nn.Sequential可以使用OrderedDict对每层进行命名
        3.nn.Sequential里面的模块按照顺序进行排列的，所以必须确保前一个模块的输出大小和下一个模块的输入大小是一致的。
        而nn.ModuleList 并没有定义一个网络，它只是将不同的模块储存在一起，这些模块之间并没有什么先后顺序可言
        """
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        """
        inputs: [batch_size, seq_len, d_model]
        """
        residual = inputs
        outputs = self.fc(inputs)
        return LayerNorm(d_model)(outputs + residual)
        # [batch_size, seq_len, d_model]


class LayerNorm(nn.Module):
    """
    目的:方便学习
    torch.nn.LayerNorm(
            normalized_shape: Union[int, List[int], torch.Size],
            eps: float = 1e-05,
            elementwise_affine: bool = True)
    讲解torch自带的LayerNorm  
    Layer Normalization 的作用是把神经网络中隐藏层归一为标准正态分布，也就是  独立同分布，以起到加快训练速度，加速收敛的作用
    
    在使用LayerNorm时，通常只需要指定normalized_shape就可以了

    normalized_shape输入尺寸
    如果传入整数，比如4，则被看做只有一个整数的list，此时LayerNorm会对输入的最后一维进行归一化，这个int值需要和输入的最后一维一样大。

    假设此时输入的数据维度是[3, 4]，则对3个长度为4的向量求均值方差，得到3个均值和3个方差，分别对这3行进行归一化(每一行)4个数字都是均值为0，方差为1）；LayerNorm中的weight和bias也分别包含4个数字，重复使用3次，对每一行进行仿射变换(仿射变)即乘以weight中对应的数字后，然后加bias中对应的数字），并会在反向传播时得到学习。
    如果输入的是个list或者torch.Size，比如[3, 4]或torch.Size([3, 4])，则会对网络最后的两维进行归一化，且要求输入数据的最后两维尺寸也是[3, 4]。

    假设此时输入的数据维度也是[3, 4]，首先对这12个数字求均值和方差，然后归一化这个12个数字；weight和bias也分别包含12个数字，分别对12个归一化后的数字进行仿射变换(仿射变)即乘以weight中对应的数字后，然后加bias中对应的数字），并会在反向传播时得到学习。
    假设此时输入的数据维度是[N, 3, 4]，则对着N个[3,4]做和上述一样的操作，只是此时做仿射变换时，weight和bias被重复用了N次。
    假设此时输入的数据维度是[N, T, 3, 4]，也是一样的，维度可以更多。
    注意:显然LayerNorm中weight和bias的shape就是传入的normalized_shape。

    eps
    归一化时加在分母上防止除零。

    elementwise_affine
    如果设为False，则LayerNorm层不含有任何可学习参数。

    如果设为True(默认是True)则会包含可学习参数weight和bias，用于仿射变换，即对输入数据归一化到均值0方差1后，乘以weight，即bias。
    """
    def __init__(self, features, eps=1e-6) -> None:
        super(LayerNorm, self).__init__()
        """
        nn.Parameter将固定的参数加入学习的参数中
        requires_grad=True 梯度更新
        """
        self.gamma = nn.Parameter(torch.ones(features), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(features), requires_grad=True)
        self.eps = eps

    def forward(self, x):
        # -1 would thus map to the last dimension, -2 to the preceding one, etc
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)
        return self.gamma * (x - mean)/(std + self.eps) + self.beta


def get_attn_pad_mask(seq_q, seq_k):
    # pad mask的作用:在对value向量加权平均的时候，让pad对应值为0，这样不会让注意力考虑到pad向量
    """
    上面 Self Attention 的计算过程中，我们通常使用 mini-batch 来计算，也就是一次计算多句话，
    即x的维度是 [batch_size, sequence_length]，sequence_length是句长，
    而一个 mini-batch 是由多个不等长的句子组成的，我们需要按照这个 mini-batch 中最大的句长对
    剩余的句子进行补齐，一般用 0 进行填充，这个过程叫做 padding
    但这时在进行 softmax 就会产生问题。回顾 softmax 函数 ，e^0是 1，是有值的，
    这样的话 softmax 中被 padding 的部分就参与了运算，相当于让无效的部分参与了运算，
    这可能会产生很大的隐患。因此需要做一个 mask 操作，让这些无效的区域不参与运算，
    一般是给无效区域加一个很大的负数偏置


    ***Tensor.detach()和Tensor.data的区别
    实际上，detach()就是返回一个新的tensor，并且这个tensor是从当前的计算图中分离出来的。
    但是返回的tensor和原来的tensor是共享内存空间的
    Tensor.detach()和Tensor.data的区别

    Tensor.data和Tensor.detach()一样， 都会返回一个新的Tensor， 这个Tensor和原来的Tensor共享内存空间，
    一个改变，另一个也会随着改变，且都会设置新的Tensor的requires_grad属性为False。这两个方法只取出原来Tensor
    的tensor数据， 丢弃了grad、grad_fn等额外的信息。区别在于Tensor.data不能被autograd追踪到，如果你修改了Tensor.data返回的新Tensor，
    原来的Tensor也会改变， 但是这时候的微分并没有被追踪到，那么当你执行loss.backward()的时候并不会报错，但是求的梯度就是错误的！因此，
    如果你使用了Tensor.data，那么切记一定不要随便修改返回的新Tensor的值。如果你使用的是Tensor.detach()方法，当你修改他的返回值并进行求导操作
    ，会报错。 因此，Tensor.detach()是安全的。


    ***torch.eq()方法详解
    对两个张量Tensor进行逐元素的比较，若相同位置的两个元素相同，则返回True；若不同，返回False
    torch.eq(input, other, *, out=None)
    Parameters(参数):
    input :必须是一个Tensor，该张量用于比较
    other :可以是一个张量Tensor，也可以是一个值value
    return(返回值):返回一个Boolean类型的张量，对两个张量Tensor进行逐元素的比较，若相同位置的两个元素相同，则返回True；若不同，返回False。
    """
    ## 比如说，我现在的句子长度是5，在后面注意力机制的部分，我们在计算出来QK转置除以根号之后，softmax之前，我们得到的形状
    ## len_input * len*input  代表每个单词对其余包含自己的单词的影响力

    ## 所以这里我需要有一个同等大小形状的矩阵，告诉我哪个位置是PAD部分，之后在计算计算softmax之前会把这里置为无穷大；

    ## 一定需要注意的是这里得到的矩阵形状是batch_size x len_q x len_k，我们是对k中的pad符号进行标识，并没有对k中的做标识，因为没必要

    ## seq_q 和 seq_k 不一定一致，在交互注意力，q来自解码端，k来自编码端，所以告诉模型编码这边pad符号信息就可以，解码端的pad信息在交互注意力层是没有用到的；
    batch_size, len_q = seq_q.size()  # 这个seq_q只是用来expand维度的
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    # 例如:seq_k = [[1,2,3,4,0], [1,2,3,5,0]]
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k] 构成一个立方体(batch_size个这样的矩阵)


def get_attn_subsequence_mask(seq):
    """
    seq: [batch_size, tgt_len]
    """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # attn_shape: [batch_size, tgt_len, tgt_len]
    """
    triu函数的使用 => 取矩阵的上三角部分
    tril函数的使用 => 取矩阵的下三角部分
    k = 0时，表示的是主对角线。
    k<0时，是在主对角线下面
    k>0时，是在主对角上面
    """
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # 生成一个上三角矩阵
    """
    torch.from_numpy()方法把数组转换成张量，且二者共享内存，对张量进行修改比如重新赋值，那么原始数组也会相应发生改变。
    byte将tensor改为byte类型
    """
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]


def make_batch(sentences):
    input_batch = [[src_vocab[n] for n in sentences[0].split()]]
    output_batch = [[tgt_vocab[n] for n in sentences[1].split()]]
    target_batch = [[tgt_vocab[n] for n in sentences[2].split()]]
    return torch.LongTensor(input_batch), torch.LongTensor(output_batch), torch.LongTensor(target_batch)


if __name__ == '__main__':
    model = Transformer()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    enc_inputs, dec_inputs, target_batch = make_batch(sentences)

    for epoch in range(20):
        optimizer.zero_grad()
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
        loss = criterion(outputs, target_batch.contiguous().view(-1))
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()