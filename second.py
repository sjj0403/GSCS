import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math
import random

import config
import utils

#初始化权重，好的初始化能有效避免梯度消失等问题的发生（具体的初始化后期可以更改）
def init_rnn_wt(rnn):
    for names in rnn._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(rnn, name)
                wt.data.uniform_(-config.init_uniform_mag, config.init_uniform_mag)
            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(rnn, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)


def init_linear_wt(linear):
    """
    initialize the weight and bias(if) of the given linear layer
    :param linear: linear layer
    :return:
    """
    linear.weight.data.normal_(std=config.init_normal_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=config.init_normal_std)


def init_wt_normal(wt):
    """
    initialize the given weight following the normal distribution
    :param wt: weight to be normal initialized
    :return:
    """
    wt.data.normal_(std=config.init_normal_std)


def init_wt_uniform(wt):
    """
    initialize the given weight following the uniform distribution
    :param wt: weight to be uniform initialized
    :return:
    """
    wt.data.uniform_(-config.init_uniform_mag, config.init_uniform_mag)

class GraphAttentionLayer(nn.Module):
    def __init__(self,in_feature,out_feature,dropout,alpha,concat=True):
        super(GraphAttentionLayer,self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Parameter(torch.zeros(size=(self.in_feature,self.out_feature)))
        self.W.data.uniform_(-config.init_uniform_mag, config.init_uniform_mag)
        #nn.init.xavier_normal(self.W.data,gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*self.out_feature,1)))
        #nn.init.xavier_normal(self.a.data,gain=1.414)
        self.a.data.uniform_(-config.init_uniform_mag, config.init_uniform_mag)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self,inputs,adj):
        ww = self.W.unsqueeze(0)
        ww = ww.repeat(config.batch_size,1,1)
        h = torch.bmm(inputs,ww)
        #N = h.size()[0]
        #h = inputs
        aa = self.a.unsqueeze(0)

        aa = aa.repeat(config.batch_size, 1, 1)
        attn_for_self = torch.bmm(h,aa[:,0:self.out_feature,:])

        attn_for_neighs = torch.bmm(h,aa[:,self.out_feature:,:])

        dense = attn_for_self + attn_for_neighs.transpose(1,2)
        #print(attn_for_neighs.T.size())

        dense = self.leakyrelu(dense)


        zero_vec = -9e15 * torch.ones_like(dense)
        attention = torch.where(adj>0,dense,zero_vec)
        attention = F.softmax(attention,dim=1)
        attention = self.dropout(attention)
        h_prime = torch.matmul(attention,h)
        #print(h_prime.size())

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class GAT(nn.Module):
    def __init__(self,nfeat,nhid,nclass,dropout,alpha,nheads):
        super(GAT,self).__init__()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.attentions = [GraphAttentionLayer(nfeat,nhid,dropout=dropout,alpha=alpha,concat=True) for _ in range(nheads)]
        for i,attention in enumerate(self.attentions):
            self.add_module("attention_{}".format(i),attention)
        
        self.out_att = GraphAttentionLayer(nhid*nheads,nclass,dropout=dropout,alpha=alpha,concat=False)

    def forward(self, x, adj):
        x = self.dropout1(x)
        #print(x.size())
        x = torch.cat([att(x,adj) for att in self.attentions],dim=2)
        x = self.dropout2(x)
        #print(x.size())
        x = F.elu(self.out_att(x,adj))
        #print(x.size())
        #print("x")
        #return F.log_softmax(x,dim=1)
        return x

class GATEncoder(nn.Module):
    def __init__(self, vocab_size):
        super(GATEncoder, self).__init__()
        self.hidden_size = config.hidden_size
        self.embedding = nn.Embedding(vocab_size,config.embedding_dim)
        self.gat = GAT(config.nfeat,config.nhid,config.hidden_size,config.dropout,config.alpha,config.nheads)
        self.gru = nn.GRU(config.embedding_dim, self.hidden_size, bidirectional=True)
        init_wt_normal(self.embedding.weight)
        init_rnn_wt(self.gru)

    def forward(self, inputs: torch.Tensor, seq_lens: torch.Tensor, adjacent: torch.Tensor):
        embedded0 = self.embedding(inputs)
        embedded0 = embedded0.transpose(0,1)
        adjacents = torch.Tensor(adjacent).cuda()
        #adj = torch.zeros(config.batch_size,201,201).cuda()
        #adj = adjacents[:,:201,:201]
        #adj[:,200,200] = 0
        embedded = torch.zeros(config.batch_size,201,config.embedding_dim).cuda()
        _, size_em, _ = embedded0.size()
        
        
        embedded[:,:size_em,:] = embedded0
        adjacents = adjacents + adjacents.transpose(1,2)
        #embedded = embedded.transpose(0,1)


        out = self.gat(embedded,adjacents)
        out = out.transpose(0,1)
        output = out[:size_em,:,:]
        packed = pack_padded_sequence(output, seq_lens, enforce_sorted=False)
        outputs, hidden = self.gru(packed)
        #把压紧的序列再填充回来
        outputs, _ = pad_packed_sequence(outputs)  # [T, B, 2*H]
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:] #+ embedded


        return outputs

'''
class TYPEEncoder(nn.Module):
    """
    Encoder for the code sequence(bigru)
    """

    def __init__(self, vocab_size):
        super(TYPEEncoder, self).__init__()
        self.hidden_size = config.hidden_size
        self.num_directions = 2

        # vocab_size: config.code_vocab_size for code encoder, size of sbt vocabulary for ast encoder
        self.embedding = nn.Embedding(vocab_size, 50)
        self.gru = nn.GRU(50, 50, bidirectional=True)
        self.dropout = nn.Dropout(config.decoder_dropout_rate)
        #self.ff = nn.Linear(config.embedding_dim, config.hidden_size)

        init_wt_normal(self.embedding.weight)
        init_rnn_wt(self.gru)

    def forward(self, inputs: torch.Tensor, seq_lens: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """

        :param inputs: sorted by length in descending order, [T, B]
        :param seq_lens: should be in descending order
        :return: outputs: [T, B, H]
                hidden: [2, B, H]
        """
        #输入根据长度大小倒序，T表示最长序列长度
        embedded = self.embedding(inputs)   # [T, B, embedding_dim]
        embedded = self.dropout(embedded)
        #在使用深度学习 特别是LSTM进行文本分析时，经常会遇到文本长度不一样的情况，此时就需要对用一个batch中不同文本使用padding的方式进行文本长度对齐，
        # 方便将训练数据输入到LSTM模型进行训练，同时为了保证模型训练的精度，应该同时告诉LSTM相关padding的情况。
        #当使用双向RNN的时候，必须要使用pack_padded_sequence.否则的话，pytorch是无法获得序列的长度，这样也无法正确的计算双向RNN的结果。
        #enforce_sorted默认是True，被处理的序列要求按序列长度降序排序。如果是False则没有要求。
        packed = pack_padded_sequence(embedded, seq_lens, enforce_sorted=False)
        outputs, hidden = self.gru(packed)
        #把压紧的序列再填充回来
        outputs, _ = pad_packed_sequence(outputs)  # [T, B, 2*H]
    
        #embedded = embedded.transpose(0,1)
        #embedded = self.ff(embedded)
        #embedded = embedded.transpose(0,1)
        
        outputs = outputs[:, :, :50] + outputs[:, :, 50:] #+ embedded

        # outputs: [T, B, H]
        # hidden: [2, B, H]
        return outputs, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_directions, batch_size, self.hidden_size, device=config.device)
'''

class CODEEncoder(nn.Module):
    """
    Encoder for the code sequence(bigru)
    """

    def __init__(self, vocab_size):
        super(CODEEncoder, self).__init__()
        self.hidden_size = config.hidden_size
        self.num_directions = 2

        # vocab_size: config.code_vocab_size for code encoder, size of sbt vocabulary for ast encoder
        self.embedding = nn.Embedding(vocab_size, config.embedding_dim)
        self.gru = nn.GRU(config.embedding_dim, self.hidden_size, bidirectional=True)
        self.dropout = nn.Dropout(config.decoder_dropout_rate)
        #self.linear = nn.Linear(50 + self.hidden_size, self.hidden_size)

        #init_linear_wt(self.linear)
        self.ff = nn.Linear(config.embedding_dim, config.hidden_size)
        self.cc = nn.Linear(config.hidden_size*2, config.hidden_size)

        init_wt_normal(self.embedding.weight)
        init_rnn_wt(self.gru)
        init_linear_wt(self.ff)
        init_linear_wt(self.cc)

    def forward(self, inputs: torch.Tensor, seq_lens: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """

        :param inputs: sorted by length in descending order, [T, B]
        :param seq_lens: should be in descending order
        :return: outputs: [T, B, H]
                hidden: [2, B, H]
        """
        #输入根据长度大小倒序，T表示最长序列长度
        embedded = self.embedding(inputs)   # [T, B, embedding_dim]
        embedded = self.dropout(embedded)
        #hyper = torch.cat((types, embedded), dim=2)
        #hyper = self.linear(hyper)
        #hyper = F.relu(hyper)
        #hyper = self.dropout(hyper)
        #在使用深度学习 特别是LSTM进行文本分析时，经常会遇到文本长度不一样的情况，此时就需要对用一个batch中不同文本使用padding的方式进行文本长度对齐，
        # 方便将训练数据输入到LSTM模型进行训练，同时为了保证模型训练的精度，应该同时告诉LSTM相关padding的情况。
        #当使用双向RNN的时候，必须要使用pack_padded_sequence.否则的话，pytorch是无法获得序列的长度，这样也无法正确的计算双向RNN的结果。
        #enforce_sorted默认是True，被处理的序列要求按序列长度降序排序。如果是False则没有要求。
        packed = pack_padded_sequence(embedded, seq_lens, enforce_sorted=False)
        outputs, hidden = self.gru(packed)
        #把压紧的序列再填充回来
        outputs, _ = pad_packed_sequence(outputs)  # [T, B, 2*H]
        
        embedded = embedded.transpose(0,1)
        embedded = self.ff(embedded)
        embedded = embedded.transpose(0,1)
        
        outputs = self.cc(outputs) + embedded

        #print("code")
        #print(outputs.size())

        # outputs: [T, B, H]
        # hidden: [2, B, H]
        return outputs, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_directions, batch_size, self.hidden_size, device=config.device)
# no use


#注意力机制（Luong Global Attention)
class Attention(nn.Module):

    def __init__(self, hidden_size=config.hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        #线性变换 y=Ax+b   (输入样本的大小，输出样本的大小)
        self.attn = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.v = nn.Parameter(torch.rand(self.hidden_size), requires_grad=True)   # [H]
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs):
        """
        forward the net
        :param hidden: the last hidden state of encoder, [1, B, H]
        :param encoder_outputs: [T, B, H]
        :return: softmax scores, [B, 1, T]
        """
        time_step, batch_size, _ = encoder_outputs.size()
        h = hidden.repeat(time_step, 1, 1).transpose(0, 1)  # [B, T, H]
        encoder_outputs = encoder_outputs.transpose(0, 1)   # [B, T, H]

        attn_energies = self.score(h, encoder_outputs)      # [B, T]
        attn_weights = F.softmax(attn_energies, dim=1).unsqueeze(1)     # [B, 1, T]

        return attn_weights

    def score(self, hidden, encoder_outputs):
        """
        calculate the attention scores of each word
        :param hidden: [B, T, H]
        :param encoder_outputs: [B, T, H]
        :param coverage: [B, T]
        :return: energy: scores of each word in a batch, [B, T]
        """
        # after cat: [B, T, 2/3*H]
        # after attn: [B, T, H]
        # energy: [B, T, H]
        energy = F.relu(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))     # [B, T, H]
        energy = energy.transpose(1, 2)     # [B, H, T]
        #0按行进行元素重复 1按列进行元素重复
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)      # [B, 1, H]
        energy = torch.bmm(v, energy)   # [B, 1, T]
        #torch.bmm(a,b)计算两个tensor的矩阵乘法，两个tensor的维度必须为3
        return energy.squeeze(1)
        #squeeze()函数去掉一个维度  unsqueeze()函数增加一个维度


class Decoder(nn.Module):

    def __init__(self, vocab_size, hidden_size=config.hidden_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, config.embedding_dim)
        self.dropout = nn.Dropout(config.decoder_dropout_rate)
        self.code_attention = Attention()
        self.gat_attention = Attention()
        self.gru = nn.GRU(config.embedding_dim + self.hidden_size, self.hidden_size)
        self.out = nn.Linear(2 * self.hidden_size, config.nl_vocab_size)
        self.pick = nn.Linear(self.hidden_size, 1)
   
        if config.use_pointer_gen:
            self.p_gen_linear = nn.Linear(2*self.hidden_size+config.embedding_dim,1)


        init_wt_normal(self.embedding.weight)
        init_rnn_wt(self.gru)
        init_linear_wt(self.out)
        init_linear_wt(self.pick)

    def forward(self, inputs, last_hidden, code_outputs, gat_outputs, extend_type_batch, extra_zeros):
        """
        forward the net
        :param inputs: word input of current time step, [B]
        :param last_hidden: last decoder hidden state, [1, B, H]
        :param source_outputs: outputs of source encoder, [T, B, H]
        :param code_outputs: outputs of code encoder, [T, B, H]
        :param ast_outputs: outputs of ast encoder, [T, B, H]
        :param extend_source_batch: [B, T]
        :param extra_zeros: [B, max_oov_num]
        :return: output: [B, nl_vocab_size]
                hidden: [1, B, H]
                attn_weights: [B, 1, T]
        """
        embedded = self.embedding(inputs).unsqueeze(0)      # [1, B, embedding_dim]
        embedded = self.dropout(embedded)

        # get attn weights of source
        # calculate and add source context in order to update attn weights during training

        code_attn_weights = self.code_attention(last_hidden, code_outputs)  # [B, 1, T]
        code_context = code_attn_weights.bmm(code_outputs.transpose(0, 1))  # [B, 1, H]
        code_context = code_context.transpose(0, 1)     # [1, B, H]
        gat_attn_weights = self.code_attention(last_hidden, gat_outputs)  # [B, 1, T]
        gat_context = gat_attn_weights.bmm(gat_outputs.transpose(0, 1))  # [B, 1, H]
        gat_context = gat_context.transpose(0, 1)     # [1, B, H]
        poss = F.softmax(last_hidden.squeeze(0),dim=1)  #[B,H]
        poss = self.pick(poss)  #[B,1]
        poss = torch.sigmoid(poss) #[B,1]
        
        # make ratio between source code and construct is 1: 1
        context = code_context + poss*gat_context     # [1, B, H]

        p_gen = None
        if config.use_pointer_gen:
            p_gen_input = torch.cat([context,last_hidden,embedded],dim=2)
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = torch.sigmoid(p_gen)
            p_gen = p_gen.squeeze(0)

        rnn_input = torch.cat([embedded, context], dim=2)   # [1, B, embedding_dim + H]
        outputs, hidden = self.gru(rnn_input, last_hidden)  # [1, B, H] for both

        outputs = outputs.squeeze(0)    # [B, H]
        context = context.squeeze(0)    # [B, H]

        vocab_dist = self.out(torch.cat([outputs, context], 1))    # [B, nl_vocab_size]
        vocab_dist = F.softmax(vocab_dist, dim=1)     # P_vocab, [B, nl_vocab_size]

        if config.use_pointer_gen:
            vocab_dist_ = p_gen * vocab_dist
            gat_attn_weights_ = gat_attn_weights.squeeze(1)
            attn_dist = (1-p_gen)*gat_attn_weights_

            if extra_zeros is not None:
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], dim=1)
            
            final_dist = vocab_dist_.scatter_add(1,extend_type_batch,attn_dist)
            
        else:
            final_dist = vocab_dist
        
        final_dist = torch.log(final_dist + config.eps)

        return final_dist, hidden


class Model(nn.Module):

    def __init__(self, code_vocab_size, type_vocab_size, nl_vocab_size,
                 model_file_path=None, model_state_dict=None, is_eval=False):
        super(Model, self).__init__()

        # vocabulary size for encoders
        #self.code_vocab_size = code_vocab_size
        self.is_eval = is_eval

        # init models
        
        self.gat_encoder = GATEncoder(type_vocab_size)
        self.code_encoder = CODEEncoder(code_vocab_size)
        self.decoder = Decoder(nl_vocab_size)

        if config.use_cuda:
            self.gat_encoder = self.gat_encoder.cuda()
            self.code_encoder = self.code_encoder.cuda()
            self.decoder = self.decoder.cuda()

        if model_file_path:
            state = torch.load(model_file_path)
            self.set_state_dict(state)

        if model_state_dict:
            self.set_state_dict(model_state_dict)

        if is_eval:
            self.gat_encoder.eval()
            self.code_encoder.eval()
            self.decoder.eval()

    def forward(self, batch, batch_size, nl_vocab, adjacent, is_test=False):
        """

        :param batch:
        :param batch_size:
        :param nl_vocab:
        :param is_test: if True, function will return before decoding
        :return: decoder_outputs: [T, B, nl_vocab_size]
        """
        # batch: [T, B]
        code_batch, code_seq_lens, type_batch, type_seq_lens, nl_batch, nl_seq_lens = batch.get_regular_input()

        # encode
        # outputs: [T, B, H]
        # hidden: [2, B, H]
        gat_outputs = self.gat_encoder(type_batch, type_seq_lens, adjacent)
        code_outputs, code_hidden = self.code_encoder(code_batch, code_seq_lens)

        # data for decoder
        # source_hidden = source_hidden[:1]
        code_hidden = code_hidden[:1]  # [1, B, H]
        decoder_hidden = code_hidden  # [1, B, H]

        if is_test:
            return code_outputs, gat_outputs, decoder_hidden

        if nl_seq_lens is None:
            max_decode_step = config.max_decode_steps
        else:
            max_decode_step = max(nl_seq_lens)

        decoder_inputs = utils.init_decoder_inputs(batch_size=batch_size, vocab=nl_vocab)  # [B]

        extend_type_batch = None
        extra_zeros = None
        if config.use_pointer_gen:
            extend_type_batch, _, extra_zeros = batch.get_pointer_gen_input()
            decoder_outputs = torch.zeros((max_decode_step, batch_size, config.nl_vocab_size+batch.max_oov_num),device=config.device)
        else:
            decoder_outputs = torch.zeros((max_decode_step, batch_size, config.nl_vocab_size), device=config.device)

        for step in range(max_decode_step):
            # decoder_outputs: [B, nl_vocab_size]
            # decoder_hidden: [1, B, H]
            # attn_weights: [B, 1, T]
            decoder_output, decoder_hidden = self.decoder(inputs=decoder_inputs,
                                             last_hidden=decoder_hidden, 
                                             code_outputs=code_outputs, 
                                             gat_outputs=gat_outputs,
                                             extend_type_batch=extend_type_batch,
                                             extra_zeros=extra_zeros)
            decoder_outputs[step] = decoder_output

            if config.use_teacher_forcing and random.random() < config.teacher_forcing_ratio and not self.is_eval:
                # use teacher forcing, ground truth to be the next input
                decoder_inputs = nl_batch[step]
            else:
                # output of last step to be the next input
                _, indices = decoder_output.topk(1)  # [B, 1]

                if config.use_pointer_gen:
                    word_indices = indices.squeeze(1).detach().cpu().numpy()
                    decoder_inputs = []
                    for index in word_indices:
                        decoder_inputs.append(utils.tune_up_decoder_input(index, nl_vocab))
                    decoder_inputs = torch.tensor(decoder_inputs, device=config.device)
                else:
                    decoder_inputs = indices.squeeze(1).detach()  # [B]
                    decoder_inputs = decoder_inputs.to(config.device)


        return decoder_outputs

    def set_state_dict(self, state_dict):
        self.code_encoder.load_state_dict(state_dict["code_encoder"])
        self.gat_encoder.load_state_dict(state_dict["gat_encoder"])
        self.decoder.load_state_dict(state_dict["decoder"])
