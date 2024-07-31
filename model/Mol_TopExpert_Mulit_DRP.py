import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Mol_substruct_Topexpert import Mol_substruct_Topexpert
from model.Topexpert import Topexpert
torch.manual_seed(42)
class MLP(nn.Sequential):
    def __init__(self):
        input_dim_gene = 49751
        hidden_dim_gene = 64
        mlp_hidden_dims_gene = [4096, 2048, 1024, 256, 64]
        super(MLP, self).__init__()
        layer_size = len(mlp_hidden_dims_gene) + 1
        dims = [input_dim_gene] + mlp_hidden_dims_gene + [hidden_dim_gene]
        self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(layer_size)])

    def forward(self, v):
        # predict
        v = v.float()
        for i, l in enumerate(self.predictor):
            v = F.relu(l(v))
        return v

class Mol_TopExpert_Mulit_DRP(nn.Module):
    ### Define layers in def__init__:
    def __init__(self, hidden_layer_number, hidden_layer_size_cells, hidden_layer_size_drugs, hidden_layer_size_single,input_features_size,
                 global_para, substruct_para, substruct_num, substruct_num1,  global_dim, substruct_dim, emb_dim, device, args,
                 activation_function="relu",
                 add_dropout_cells=False,
                 add_dropout_drugs=False,
                 dropout_rate_cells=0.1,
                 dropout_rate_drugs=0.1,
                 use_single_cell=True,
                 features=['CCLE_chromatin', 'CCLE_copynumber', 'CCLE_expression', 'CCLE_methylation', 'CCLE_miRNA', 'Mordred', 'DrugTax']
                 ):
        super(Mol_TopExpert_Mulit_DRP, self).__init__()
        # Defining network architecture variables
        self.device = device
        self.features = features
        self.hidden_layer_number = hidden_layer_number
        self.hidden_layer_size_cells = hidden_layer_size_cells
        self.hidden_layer_size_drugs = hidden_layer_size_drugs
        self.hidden_layer_size_single = hidden_layer_size_single
        self.add_dropout_cells = add_dropout_cells
        self.add_dropout_drugs = add_dropout_drugs
        self.dropout_rate_cells = dropout_rate_cells
        self.dropout_rate_drugs = dropout_rate_drugs
        self.use_single_cell = use_single_cell
        self.overall_model = nn.ModuleList()
        self.linears = nn.ModuleList()
        self.model_gene = MLP()

        for feature_block in features[5:7]:
            nn_drugs = nn.ModuleList()
            block = nn.Sequential(nn.Linear(input_features_size[feature_block], hidden_layer_size_drugs),
                                  nn.ReLU())
            nn_drugs.append(block)

            for hidden_layer in range(hidden_layer_number - 1):
                if add_dropout_drugs == True:
                    block = nn.Sequential(nn.Dropout(dropout_rate_drugs),
                                          nn.Linear(hidden_layer_size_drugs, hidden_layer_size_drugs),
                                          nn.ReLU())
                else:
                    block = nn.Sequential(nn.Linear(hidden_layer_size_drugs, hidden_layer_size_drugs),
                                          nn.ReLU())
                nn_drugs.append(block)
            self.overall_model.append(nn_drugs)

        self.Mol_substruct_Topexpert = Mol_substruct_Topexpert(global_para=global_para, substruct_para=substruct_para,
        emb_dim=emb_dim, global_dim=global_dim, substruct_dim=substruct_dim, substruct_num=substruct_num, num_tr_scf = args.num_tr_scf, device=device).to(device)


    ### Defining model connections
    def forward(self, inputs, drug_data, drug_data1,  drug_cell_index):
        # drugs_list = []
        # cells_list = []
        # for index in range(len(inputs)):
        #     data = torch.tensor(inputs[index]).to(self.device).float()
        #     for i in range(self.hidden_layer_number):
        #         data = self.overall_model[index][i](data)
        #     if(index<5):
        #         cells_list.append(data)
        #         if(index == 0):
        #             cells_merged = data
        #         else:
        #             cells_merged = torch.cat((cells_merged, data), dim=1)
        #     else:
        #         drugs_list.append(data)
        #         if (index == 5):
        #             drugs_merged = data
        #         else:
        #             drugs_merged = torch.cat((drugs_merged, data), dim=1)
        #

        input = drug_data['drug_cell_emb']
        cells = torch.cat((torch.tensor(input[0]).to(self.device), torch.tensor(input[1]).to(self.device), torch.tensor(input[2]).to(self.device), torch.tensor(input[3]).to(self.device), torch.tensor(input[4]).to(self.device)), dim=1)
        cells_merged = self.model_gene(cells)
        for index in range(2):
            data = torch.tensor(input[index + 5]).to(self.device).float()
            for i in range(self.hidden_layer_number):
                data = self.overall_model[index][i](data)
            if (index == 0):
                drugs_merged = data
            else:
                drugs_merged = torch.cat((drugs_merged, data), dim=1)

        ic50_predicts = self.Mol_substruct_Topexpert(drug_cell_index = drug_cell_index, cells_merged = cells_merged, drugs_merged = drugs_merged, **drug_data)


        return ic50_predicts

