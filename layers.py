import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sparse


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.dropout = dropout

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))  # FxF'
        self.attn = nn.Parameter(torch.zeros(size=(2 * out_features)))  # 2F'

        nn.init.xavier_normal_(self.W, gain=1.414)
        nn.init.xavier_normal_(self.attn, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, edge):
        """
        input: NxF
        edge: 2xE
        """
        N = input.size()[0]
        if input.is_sparse:
            h = torch.sparse.mm(input, self.W)  # (NxF) * (FxF') = NxF'
        else:
            h = torch.mm(input, self.W)

        # Self-attention (because including self edges) on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()  # Ex2F'.t() = 2F'xE
        values = edge_h.mm(self.attn)  # E
        sp_edge_h = torch.sparse_coo_tensor(edge, -self.leakrelu(values), size=(N, N))  # values() = E

        sp_edge_h = sparse.nn.functional.softmax(sp_edge_h, dim=1)
        sp_edge_h = sparse.nn.functional.dropout(sp_edge_h, p=self.dropout)

        # apply attention
        h_prime = torch.sparse.mm(sp_edge_h, h) # (NxN) * (NxF') = (NxF')


        if self.concat:
            # if this layer is not last layer
            return F.elu(h_prime)
        else:
            # if this layer is last layer
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
