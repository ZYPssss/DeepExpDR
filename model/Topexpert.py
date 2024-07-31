from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from torch.nn import ParameterList, Parameter
import torch.nn.functional as F
import math
torch.manual_seed(42)

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

class expert(torch.nn.Module):
    def __init__(self, emb):
        super(expert, self).__init__()
        self.clf = nn.Sequential(
            nn.Linear(emb, 128),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.clf(x)
        return x

class Topexpert(torch.nn.Module):  # expert 를 parallel 하게

    def __init__(self, drug_emb_dim, num_experts):
        super(Topexpert, self).__init__()

        self.emb_dim = drug_emb_dim
        self.num_tasks = 1
        self.gate_dim = 50
        self.num_experts = num_experts
        self.gate = gate(self.emb_dim, self.gate_dim)
        self.cluster = nn.Parameter(torch.Tensor(self.num_experts, self.gate_dim))
        torch.nn.init.xavier_normal_(self.cluster.data)
        ## optimal transport
        self.cos_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

        self.T = 10
        # self.criterion = criterion
        self.experts_w = nn.Parameter(torch.empty(self.emb_dim, self.num_tasks * self.num_experts))
        self.experts_b = nn.Parameter(torch.empty(self.num_tasks * self.num_experts))
        self.reset_experts()

    def reset_experts(self):
        torch.nn.init.kaiming_uniform_(self.experts_w, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.emb_dim)
        torch.nn.init.uniform_(self.experts_b, -bound, bound)

    def forward(self, gnn_out, gate_input):

        ## multi-head mlps
        gnn_out = torch.unsqueeze(gnn_out, -1)  # N x emb_dim x 1
        gnn_out = gnn_out.repeat(1, 1, self.num_tasks * self.num_experts)  # N x emb_dim x (tasks * experts)

        clf_logit = torch.sum(gnn_out * self.experts_w, dim=1) + self.experts_b  # N x (tasks * experts)

        clf_logit = clf_logit.view(-1, self.num_tasks, self.num_experts)  # N  x tasks x num_experts
        ## multi-head mlps
        z = self.gate(gate_input)
        q = self.get_q(z)
        q, q_idx = self.assign_head(q)
        #scores = torch.sum(torch.sigmoid(clf_logit) * q, dim=-1)
        scores = torch.sum(torch.sigmoid(clf_logit) * q, dim=-1)
        return scores

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

        return g, q_idx  # N x tasks x heads // N // N

    def get_q(self, z):
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster, 2), 2))
        q = q.pow((1.0 + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q

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
