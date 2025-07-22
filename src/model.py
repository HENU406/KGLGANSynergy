import math
import os
from typing import Optional

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree
import torch_geometric.nn as gnn
from torch_geometric.utils import softmax


class LGAT(gnn.MessagePassing):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_heads: int,
                 local_attn_dropout_ratio: float = 0.0,
                 local_ffn_dropout_ratio: float = 0.0):
        super().__init__(aggr='add', node_dim=0)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.local_attn_dropout_ratio = local_attn_dropout_ratio

        # Input to hidden linear transformation
        self.input_to_hidden = nn.Linear(input_dim, hidden_dim)

        # Attention mechanism layers
        self.linear_dst = nn.Linear(hidden_dim, hidden_dim)
        self.linear_src_edge = nn.Linear(2 * hidden_dim, hidden_dim)

        # Feed-forward network with hidden to output transformation
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(local_ffn_dropout_ratio),
            nn.Linear(hidden_dim, output_dim)  # Hidden to output transformation
        )

    def reset_parameters(self):
        # self.input_to_hidden.reset_parameters()
        # self.linear_dst.reset_parameters()
        # self.linear_src_edge.reset_parameters()
        # for layer in self.ffn:
        #     if isinstance(layer, nn.Linear) or isinstance(layer, nn.LayerNorm):
        #         layer.reset_parameters()

        torch.nn.init.xavier_normal_(self.input_to_hidden.weight, gain=1)
        torch.nn.init.xavier_normal_(self.linear_dst.weight, gain=1)
        torch.nn.init.xavier_normal_(self.linear_src_edge.weight, gain=1)
        for layer in self.ffn:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight, gain=1)
            elif isinstance(layer, nn.LayerNorm):
                # layer.reset_parameters()
                nn.init.ones_(layer.weight)  # weight 初始化为 1
                nn.init.zeros_(layer.bias)  # bias 初始化为 0

    def forward(self, x, edge_index, edge_attr):
        # Transform input features to hidden_dim
        x = self.input_to_hidden(x)

        # Perform local attention propagation
        local_out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        local_out = local_out.view(-1, self.hidden_dim)

        # Pass through feed-forward network
        x = self.ffn(local_out)

        return x

    def message(self, x_i, x_j, edge_attr, edge_index_i, size_i: Optional[int]):
        H, C = self.num_heads, self.hidden_dim // self.num_heads

        # Linear transformation for destination nodes
        x_dst = self.linear_dst(x_i).view(-1, H, C)

        # Linear transformation for source nodes and edge features
        m_src = self.linear_src_edge(torch.cat([x_j, edge_attr], dim=-1)).view(-1, H, C)

        # Scaled dot-product attention
        alpha = (x_dst * m_src).sum(dim=-1) / math.sqrt(C)
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
        alpha = F.dropout(alpha, p=self.local_attn_dropout_ratio, training=self.training)

        return m_src * alpha.unsqueeze(-1)


class GraphConv(nn.Module):
    def __init__(self, node_dim, hidden_channels, lgat_hidden_channels, lgat_head, num_layers=2, dropout=0.5,
                 use_bn=True, use_residual=True,
                 use_weight=True, use_init=False, use_act=True):
        super(GraphConv, self).__init__()

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(node_dim, hidden_channels))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers):
            self.convs.append(
                LGAT(hidden_channels, lgat_hidden_channels, hidden_channels, lgat_head, dropout, dropout))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.use_act = use_act

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        layer_ = []

        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        layer_.append(x)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.use_residual:
                x = x + layer_[-1]
        return x


class GGAT(nn.Module):

    def __init__(self, in_channels,
                 out_channels,
                 num_heads,
                 use_weight=True):
        super().__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        if use_weight:
            self.Wv = nn.Linear(in_channels, out_channels * num_heads)

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight

    def reset_parameters(self):
        # self.Wk.reset_parameters()
        # self.Wq.reset_parameters()
        # if self.use_weight:
        #     self.Wv.reset_parameters()

        torch.nn.init.xavier_normal_(self.Wk.weight, gain=1)
        torch.nn.init.xavier_normal_(self.Wq.weight, gain=1)
        if self.use_weight:
            torch.nn.init.xavier_normal_(self.Wv.weight, gain=1)

    def forward(self, query_input, source_input, output_attn=False):
        # feature transformation
        qs = self.Wq(query_input).reshape(-1, self.num_heads, self.out_channels)
        ks = self.Wk(source_input).reshape(-1, self.num_heads, self.out_channels)
        if self.use_weight:
            vs = self.Wv(source_input).reshape(-1, self.num_heads, self.out_channels)
        else:
            vs = source_input.reshape(-1, 1, self.out_channels)

        # normalize input
        qs = qs / torch.norm(qs, p=2)  # [N, H, M]
        ks = ks / torch.norm(ks, p=2)  # [L, H, M]
        N = qs.shape[0]

        # numerator
        kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
        attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs)  # [N, H, D]
        attention_num += N * vs

        # denominator
        all_ones = torch.ones([ks.shape[0]]).to(ks.device)
        ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
        attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)  # [N, H]

        # attentive aggregated results
        attention_normalizer = torch.unsqueeze(
            attention_normalizer, len(attention_normalizer.shape))  # [N, H, 1]
        attention_normalizer += torch.ones_like(attention_normalizer) * N
        attn_output = attention_num / attention_normalizer  # [N, H, D]

        # compute attention for visualization if needed
        if output_attn:
            attention = torch.einsum("nhm,lhm->nlh", qs, ks).mean(dim=-1)  # [N, N]
            normalizer = attention_normalizer.squeeze(dim=-1).mean(dim=-1, keepdims=True)  # [N,1]
            attention = attention / normalizer

        final_output = attn_output.mean(dim=1)

        if output_attn:
            return final_output, attention
        else:
            return final_output


class TransConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, num_heads=1,
                 dropout=0.5, use_bn=True, use_residual=True, use_weight=True, use_act=True):
        super().__init__()

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(hidden_channels))

        for i in range(num_layers):
            self.convs.append(
                GGAT(hidden_channels, hidden_channels, num_heads=num_heads, use_weight=use_weight))
            self.bns.append(nn.LayerNorm(hidden_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.use_act = use_act

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x):
        layer_ = []

        # input MLP layer
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # store as residual link
        layer_.append(x)

        for i, conv in enumerate(self.convs):
            # graph convolution with full attention aggregation
            x = conv(x, x)
            if self.use_residual:
                x = (x + layer_[i]) / 2.
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_.append(x)

        return x

    def get_attentions(self, x):
        layer_, attentions = [], []
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        layer_.append(x)
        for i, conv in enumerate(self.convs):
            x, attn = conv(x, x, output_attn=True)
            attentions.append(attn)
            if self.use_residual:
                x = (x + layer_[i]) / 2.
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            layer_.append(x)
        return torch.stack(attentions, dim=0)  # [layer num, N, N]


class MutualCrossAttention(nn.Module):
    def __init__(self, aggregate, dropout):
        super(MutualCrossAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # self.ln = nn.LayerNorm(hidden_channels)
        self.aggregate = aggregate

    def forward(self, x1, x2):
        x1 = torch.reshape(x1, (x1.shape[0], 1, x1.shape[1]))
        x2 = torch.reshape(x2, (x2.shape[0], 1, x2.shape[1]))
        # Assign x1 and x2 to query and key
        query = x1
        key = x2
        d = query.shape[-1]

        # Basic attention mechanism formula to get intermediate output A
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(d)
        output_A = torch.matmul(self.dropout(F.softmax(scores, dim=-1)), x2)
        # Basic attention mechanism formula to get intermediate output B
        scores = torch.matmul(key, query.transpose(-1, -2)) / math.sqrt(d)
        output_B = torch.matmul(self.dropout(F.softmax(scores, dim=-1)), x1)

        # Make the summation of the two intermediate outputs
        if self.aggregate == 'add':
            output = output_A + output_B    # add
        else:
            output = 0.5 * output_A + 0.5 * output_B    # avg

        output = torch.reshape(output, (x1.shape[0], x1.shape[2]))

        return output


class KGLGANSynergy(nn.Module):
    def __init__(self, node_num, edge_num, edge_dim, in_channels, hidden_channels, out_channels,
                 lgat_hidden_channels,
                 lgat_head, trans_num_layers=1, trans_num_heads=1, trans_dropout=0.5, trans_use_bn=True,
                 trans_use_residual=True, trans_use_weight=True, trans_use_act=True,
                 gnn_num_layers=1, gnn_dropout=0.5, gnn_use_weight=True, gnn_use_init=False, gnn_use_bn=True,
                 gnn_use_residual=True, gnn_use_act=True,
                 use_graph=True, graph_weight=0.8, aggregate='add'):
        super().__init__()
        self.node_embedding = nn.Embedding(node_num, in_channels, max_norm=1)
        self.edge_embedding = nn.Embedding(edge_num, edge_dim, max_norm=1)
        self.trans_conv = TransConv(in_channels, hidden_channels, trans_num_layers, trans_num_heads, trans_dropout,
                                    trans_use_bn, trans_use_residual, trans_use_weight, trans_use_act)

        self.graph_conv = GraphConv(in_channels, hidden_channels, lgat_hidden_channels, lgat_head, gnn_num_layers,
                                    gnn_dropout, gnn_use_bn,
                                    gnn_use_residual, gnn_use_weight, gnn_use_init, gnn_use_act)
        self.use_graph = use_graph
        self.graph_weight = graph_weight

        self.aggregate = aggregate

        self.MCA = MutualCrossAttention(aggregate, 0.0)

        self.fc = nn.Linear(2 * hidden_channels, hidden_channels)

        self.params1 = list(self.trans_conv.parameters())
        self.params2 = list(self.graph_conv.parameters()) if self.graph_conv is not None else []
        self.params2.extend(list(self.fc.parameters()))

    def forward(self, entity_tensor, edge_index, drug1_ids, drug2_ids, cell_ids):
        x = self.node_embedding(entity_tensor.long())
        edge_attr = self.edge_embedding(edge_index[0])

        x1 = self.trans_conv(x)
        x2 = self.graph_conv(x, edge_index, edge_attr)

        # MCA融合
        d1_f = self.MCA(x1[drug1_ids], x2[drug1_ids])
        d2_f = self.MCA(x1[drug2_ids], x2[drug2_ids])
        c_f = self.MCA(x1[cell_ids], x2[cell_ids])

        combine_drug = torch.max(d1_f, d2_f)
        logits = torch.sigmoid((combine_drug * c_f).sum(dim=1))

        return logits

    def get_attentions(self, x):
        attns = self.trans_conv.get_attentions(x)  # [layer num, N, N]

        return attns

    def reset_parameters(self):
        self.trans_conv.reset_parameters()
        if self.use_graph:
            self.graph_conv.reset_parameters()
        self.node_embedding.reset_parameters()
        self.edge_embedding.reset_parameters()
        self.fc.reset_parameters()
