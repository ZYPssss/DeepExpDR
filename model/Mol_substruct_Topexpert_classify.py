from model.SetTransformer import SAB
from model.gnn import GNNGraph
import numpy as np
from model.Topexpert import Topexpert
import torch
import torch.nn as nn
import math
from torch.nn import Parameter
from typing import List, Optional, Union
from torch import Tensor
import copy
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn import SGConv, GATConv, GCNConv, GINConv
import copy
import torch.nn.functional as F

class GlobalPooling(torch.nn.Module):
    r"""A global pooling module that wraps the usage of
    :meth:`~torch_geometric.nn.glob.global_add_pool`,
    :meth:`~torch_geometric.nn.glob.global_mean_pool` and
    :meth:`~torch_geometric.nn.glob.global_max_pool` into a single module.

    Args:
        aggr (string or List[str]): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            If given as a list, will make use of multiple aggregations in which
            different outputs will get concatenated in the last dimension.
    """
    def __init__(self, aggr: Union[str, List[str]]):
        super().__init__()

        self.aggrs = [aggr] if isinstance(aggr, str) else aggr

        assert len(self.aggrs) > 0
        assert len(set(self.aggrs) | {'sum', 'add', 'mean', 'max'}) == 4

    def forward(self, x: Tensor, batch: Optional[Tensor],
                size: Optional[int] = None) -> Tensor:
        """"""
        xs: List[Tensor] = []

        for aggr in self.aggrs:
            if aggr == 'sum' or aggr == 'add':
                xs.append(global_add_pool(x, batch, size))
            elif aggr == 'mean':
                xs.append(global_mean_pool(x, batch, size))
            elif aggr == 'max':
                xs.append(global_max_pool(x, batch, size))

        return xs[0] if len(xs) == 1 else torch.cat(xs, dim=-1)


    def __repr__(self) -> str:
        aggr = self.aggrs[0] if len(self.aggrs) == 1 else self.aggrs
        return f'{self.__class__.__name__}(aggr={aggr})'

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # attention_probs = self.dropout(attention_probs)
        #
        # context_layer = torch.matmul(attention_probs, value_layer)
        # context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # context_layer = context_layer.view(*new_context_layer_shape)
        return attention_probs

class Self_Sub_Attention(nn.Module):
    def __init__(self, hidden_size, hidden_size1):
        super(Self_Sub_Attention, self).__init__()
        self.attention_head_size = hidden_size1
        self.query = nn.Linear(hidden_size, hidden_size1)
        self.key = nn.Linear(hidden_size, hidden_size1)
        self.value = nn.Linear(hidden_size, hidden_size1)


    def forward(self, hidden_states, hidden_states1):
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states1)
        value_layer = self.value(hidden_states1)


        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(0, 1))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = torch.sigmoid(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # attention_probs = self.dropout(attention_probs)

        # context_layer = torch.matmul(attention_probs, value_layer)
        # context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # context_layer = context_layer.view(*new_context_layer_shape)
        return attention_probs

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):

        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class Embeddings1(nn.Module):
    """Construct the embeddings from protein/target, position embeddings.
    """
    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_rate):
        super(Embeddings1, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_size, hidden_size)

        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class Embeddings(nn.Module):
    """Construct the embeddings from protein/target, position embeddings.
    """
    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_rate):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_size, hidden_size)

        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        seq_length = len(input_ids)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        #position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class SelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class Intermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(Intermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.relu(hidden_states)
        return hidden_states

class Output(nn.Module):
    def __init__(self, intermediate_size, hidden_size, hidden_dropout_prob):
        super(Output, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class AdjAttenAgger(torch.nn.Module):
    def __init__(self, hidden_size, mid_dim, *args, **kwargs):
        super(AdjAttenAgger, self).__init__(*args, **kwargs)
        self.attn = SelfAttention(hidden_size, 1, 0.1)

    def forward(self, sub_feat, substructure_index):

        sub_index = substructure_index[0]
        input_mask = substructure_index[1]
        for i in range(len(sub_index)):
            sub_feat_x = sub_feat[sub_index[i]]
            sub_feat_x = sub_feat_x.unsqueeze(0)
            if(i == 0):
                drug_embeddings = sub_feat_x
            else:
                drug_embeddings = torch.cat((drug_embeddings, sub_feat_x), dim = 0)
        e_mask = input_mask.unsqueeze(1).unsqueeze(2)
        e_mask = (1.0 - e_mask) * -10000.0
        O = self.attn(drug_embeddings, e_mask)
        O = O[:, 0, 0, :]

        return O

class gate(torch.nn.Module):
    def __init__(self, emb_dim, gate_dim=300):
        super(gate, self).__init__()
        self.linear1 = nn.Linear(emb_dim, gate_dim)
        self.batchnorm = nn.BatchNorm1d(gate_dim)
        self.linear2 = nn.Linear(gate_dim, gate_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.batchnorm(x)
        x = F.relu(x)
        gate_emb = self.linear2(x)
        return gate_emb

class GraphConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, layers, K, device):
        super(GraphConv, self).__init__()
        self.conv = nn.ModuleList()
        self.layers = layers
        self.device = device
        self.K = K
        for i in range(self.layers):
            self.conv.append(SGConv(in_channels, out_channels, K=self.K))
        self.prelu = nn.PReLU(out_channels)
        self.batch = nn.BatchNorm1d(out_channels)

    def glorot(self, tensor):
        if tensor is not None:
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)

    def interaction_edge(self, adj, nan):
        a, b = torch.where(adj != nan)
        edge_index = torch.cat((a.unsqueeze(dim=-1), b.unsqueeze(dim=-1)), 1)
        edge_index = torch.cat((edge_index.T, edge_index[:, [1, 0]].T), 1)
        edge_weight = adj[a, b]
        edge_weight = torch.cat((edge_weight, edge_weight), 0)
        return edge_index, edge_weight

    def feature_initialize(self, node_num, max_dim):
        node_feat = torch.eye(node_num)
        node_feat = torch.cat((node_feat, torch.zeros([node_num, max_dim - node_num])), axis = 1) \
                    if (node_num < max_dim) else node_feat
        return node_feat

    def forward(self, cell_feat, sub_feat, maps):
        # all_num = len(cell_feat) + len(sub_feat)
        # x = nn.Parameter(torch.Tensor(all_num, len(sub_feat[0])))
        # self.glorot(x)
        x = torch.cat((cell_feat, sub_feat), dim=0)
        x = x.to(self.device)
        #x = torch.cat((cell_feat, sub_feat), dim=0)

        edge_index, edge_weight = self.interaction_edge(maps, 0.5)
        for i in range(self.layers):
            x = self.conv[i](x, edge_index, edge_weight = edge_weight)
            x = self.batch(self.prelu(x))
        cell_F = x[0:66,:]
        sub_F = x[66:,:]
        return cell_F, sub_F

class Mol_substruct_Experts(torch.nn.Module):
    def __init__(self, substruct_para, emb_dim,
        substruct_num, global_dim, substruct_dim, use_embedding=False,
        device=torch.device('cpu'), *args, **kwargs):
        super(Mol_substruct_Experts, self).__init__(*args, **kwargs)
        self.device = device
        self.use_embedding = use_embedding
        self.substruct_num = substruct_num
        self.layers = substruct_para['num_layer']

        self.drug_glofeat = substruct_para['drug_glofeat']
        self.cline_glofeat = substruct_para['cline_glofeat']
        self.cell_linear = nn.Linear(len(self.cline_glofeat[0]), substruct_dim)
        self.drug_linear = nn.Linear(len(self.drug_glofeat[0]), substruct_dim)
        if self.use_embedding:
            self.substruct_emb = torch.nn.Parameter(
                torch.zeros(substruct_num, emb_dim)
            )
        else:
            #self.substruct_encoder = GNNGraph(**substruct_para)
            input_dim_drug = 2586
            transformer_emb_size_drug = substruct_dim
            transformer_dropout_rate = 0.1
            self.emb = Embeddings(input_dim_drug,
                                  transformer_emb_size_drug,
                                  533,
                                  transformer_dropout_rate)
        self.query = torch.nn.Sequential(
            torch.nn.Tanh(),
            torch.nn.Linear(64 + 32 + 32, emb_dim)
        )
        self.substruct_rela = torch.nn.Linear(emb_dim, substruct_num)

        self.weight = Parameter(torch.Tensor(substruct_dim, substruct_dim))
        self.glorot(self.weight)
        self.K = substruct_para['k']
        self.GraphConv = GraphConv(substruct_dim, substruct_dim, self.layers, self.K,  device)
        self.fuse_drug = nn.Linear(128, 64)
        self.fuse_cell = nn.Linear(128, 64)
        self.aggregator = AdjAttenAgger(64, 128)
        self.linear1 = nn.Linear(128, 64)
        self.linear2 = nn.Linear(128, 64)
        self.predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )


    def glorot(self, tensor):
        if tensor is not None:
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)

    def forward(self, substruct_data, ddi_mask_H, average_projection, drug_cell_index, cells_merged, drugs_merged, substructure_index):


        substruct_embeddings = self.emb(substruct_data['batched_data'][0])
        cells_merged = cells_merged
        use_cell_id = drug_cell_index[:, 1].long()
        use_drug_id = drug_cell_index[:, 0].long()

        sub_index = substructure_index[0]
        mask = substructure_index[1]
        ddi_mask_H = ddi_mask_H.float()
        sub_sum = 1 / torch.sum(ddi_mask_H, dim=1)
        sub_sum = sub_sum.reshape((len(sub_sum), 1))
        score_x = ddi_mask_H * sub_sum


        cells_merged_x = self.cell_linear(self.cline_glofeat)
        drugs_merged_x = self.drug_linear(self.drug_glofeat)

        drug_emb = torch.matmul(average_projection, drugs_merged)
        drug_emb = drug_emb.repeat(len(cells_merged), 1)
        cell_drug_emb = torch.cat((cells_merged, drug_emb), dim=1)
        query = self.query(cell_drug_emb)
        substruct_weight = self.substruct_rela(query)

        substruct_embeddings_x = torch.matmul(score_x.T, drugs_merged_x)
        substruct_weight = torch.matmul(cells_merged_x, torch.matmul(self.weight, substruct_embeddings_x.T))
        substruct_weight = torch.sigmoid(substruct_weight)
        cells_merged, substruct_embeddings = self.GraphConv(cells_merged, substruct_embeddings, substruct_weight)
        cell_drug_embeddings_x = torch.cat((cells_merged_x[use_cell_id], drugs_merged_x[use_drug_id]), dim=1)
        score = self.aggregator(substruct_embeddings, substructure_index)
        for i in range(len(sub_index)):
            sub_feat = substruct_embeddings[sub_index[i]]
            drug_feat = torch.matmul(score[i], sub_feat)
            drug_feat = drug_feat.unsqueeze(0)
            if(i == 0):
                drug_embeddings = drug_feat
            else:
                drug_embeddings = torch.cat((drug_embeddings, drug_feat), dim=0)

        # cells_merged = cells_merged_x
        # drug_embeddings = drugs_merged_x
        use_cell = cells_merged[use_cell_id]
        use_drug = drug_embeddings[use_drug_id]

        cell_drug_embeddings = torch.cat((use_cell, use_drug), dim=1)
        cell_drug_embeddings = self.linear1(cell_drug_embeddings)
        cell_drug_embeddings_x = self.linear2(cell_drug_embeddings_x)
        #ic50_predict = self.predictor(cell_drug_embeddings)
        ic50_predict = self.predictor(torch.cat((cell_drug_embeddings, cell_drug_embeddings_x), dim=1))
        ic50_predict = torch.sigmoid(ic50_predict)

        return ic50_predict

class Mol_substruct_Topexpert(torch.nn.Module):
    def __init__(
        self, global_para, substruct_para, emb_dim,
        substruct_num, global_dim, substruct_dim, num_tr_scf, use_embedding=False,
        device=torch.device('cpu'), *args, **kwargs
    ):
        super(Mol_substruct_Topexpert, self).__init__(*args, **kwargs)
        self.device = device
        self.use_embedding = use_embedding
        #self.global_encoder = GNNGraph(**global_para)\
        self.emb_dim = global_dim
        self.num_tasks = 1
        self.gate_dim = 50
        self.num_experts = 3

        self.T = 10
        self.Mol_substruct_experts = nn.ModuleList()
        for i in range(self.num_experts):
             self.Mol_substruct_experts.append(Mol_substruct_Experts(substruct_para=substruct_para,emb_dim=emb_dim, global_dim=global_dim, substruct_dim=substruct_dim, substruct_num=substruct_num, device=device).to(device))


    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)
        if self.use_embedding:
            torch.nn.init.xavier_uniform_(self.substruct_emb)

    def forward(
        self, substruct_data, mol_data,
        ddi_mask_H, average_projection, e, drug_cell_emb, substructure_index, cells_merged, drugs_merged, drug_cell_index
    ):

        for i in range(self.num_experts):
            if(i == 0):
                ic50_predicts = self.Mol_substruct_experts[i](substruct_data, ddi_mask_H, average_projection, drug_cell_index, cells_merged, drugs_merged, substructure_index)
            else:
                w = self.Mol_substruct_experts[i](substruct_data, ddi_mask_H, average_projection, drug_cell_index, cells_merged, drugs_merged, substructure_index)
                ic50_predicts = torch.cat((ic50_predicts, w), dim=1)
        ic50_predicts = ic50_predicts.view(-1, self.num_tasks, self.num_experts)

        return ic50_predicts

class CLassfiy_gate(torch.nn.Module):
    def __init__(self, global_dim, num_tr_scf, global_para, device):
        super(CLassfiy_gate, self).__init__()
        self.emb_dim = global_dim
        self.device = device
        self.num_tasks = 1
        self.gate_dim = 50
        self.num_experts = 3
        self.global_encoder = GNNGraph(**global_para)
        self.gate = gate(self.emb_dim, self.gate_dim)

        self.cluster = nn.Parameter(torch.Tensor(self.num_experts, self.gate_dim))
        torch.nn.init.xavier_normal_(self.cluster.data)
        ## optimal transport
        self.scf_emb = nn.Parameter(torch.Tensor(num_tr_scf, self.gate_dim))
        torch.nn.init.xavier_normal_(self.scf_emb.data)
        self.cos_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)


    def forward(self, mol_data):
        global_embeddings = self.global_encoder(**mol_data)
        z = self.gate(global_embeddings)
        q = self.get_q(z)
        return z, q

    def target_distribution(self, q):
        weight = q ** 2 / q.sum(0)
        p = (weight.t() / weight.sum(1)).t()
        return p

    def alignment_loss(self, scf_idx, q):
        e = self.scf_emb[scf_idx]
        e = e.unsqueeze(dim=-1)
        mu = torch.transpose(self.cluster, 1, 0).unsqueeze(dim=0)
        loss = torch.mean(torch.sum(q * (1 - self.cos_similarity(e, mu)), dim=1))
        return loss

    def get_q(self, z):
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster, 2), 2))
        q = q.pow((1.0 + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q

    def get_num_experts(self):
        return self.num_experts

    def assign_head(self, q):
        q_idx = torch.argmax(q, dim=-1)  # N x 1
        if self.training:
            g = F.gumbel_softmax((q + 1e-10).log(), tau=10, hard=False, dim=-1)
            g = torch.unsqueeze(g, 1)
            g = g.repeat(1, self.num_tasks, 1)  # N x tasks x heads
            return g, q_idx  # N x tasks x heads // N // N
        else:
            q = torch.unsqueeze(q, 1)
            q = q.repeat(1, self.num_tasks, 1)  # N x tasks x heads
            return q, q_idx  # N x tasks x heads // N // N
