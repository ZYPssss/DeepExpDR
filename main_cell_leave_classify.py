import numpy as np
import pandas as pd
import random

import warnings

from Top_expert_need import MoleculeDataset, random_split
from resources import get_drug_scf_idx, Scf_index, loadDrug2, loadDrug1, load_datasets, loadDrug, load_cells_datasets, set_seed
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch_geometric.loader import DataLoader
import os
from tqdm import tqdm
from rdkit.Chem import BRICS
from rdkit import Chem
warnings.filterwarnings("ignore")
seed = 42
set_seed(seed)

from sklearn.model_selection import KFold
from train_val_classify import train_val_cell_leave_1, train_val_cell_leave, class_train
from Dataload.MyData import  MyDataset, _collate, getDataload
from model.utils import buildPrjSmiles, graph_batch_from_smile
from model.Mol_TopExpert_Mulit_DRP_classify import Mol_TopExpert_Mulit_DRP
from agrs.args import args, args1
from model.Mol_substruct_Topexpert_classify import CLassfiy_gate
import codecs
from subword_nmt.apply_bpe import BPE
from model.Prepare_model import prepare_model
from model.Mol_substruct1 import Mol_substruct1
# Defining whether to use scRNA-seq data or not and, if using, the number of single cells per cell line that will be used
use_sc = False
single_cells = 10
args1.device = torch.device("cuda:" + str(0)) if torch.cuda.is_available() else torch.device("cpu")
device = args1.device
batch_size = 256

# Load datasets
featuring_train, target = load_datasets(mode="train_test", use_sc_data=use_sc)

train_test_index = np.loadtxt('./data/Binary_class/train_test_index.csv', delimiter=',', dtype=int)
valcell_index = np.loadtxt('./data/Binary_class/valcell_index.csv', delimiter=',', dtype=int)
valdrug_index = np.loadtxt('./data/Binary_class/valdrug_index.csv', delimiter=',', dtype=int)

cline_glofeat = np.load('./data/Binary_class/cline_glofeat_cell_cold.npy')
drug_glofeat = np.load('./data/Binary_class/drug_glofeat_cell_cold.npy')
drug_glofeat = torch.tensor(drug_glofeat.copy()).to(device).float()
cline_glofeat = torch.tensor(cline_glofeat.copy()).to(device).float()
# Defining input shapes for implementation in the model
input_features_size = {
    'CCLE_chromatin': featuring_train[0].shape[1],
    'CCLE_copynumber': featuring_train[1].shape[1],
    'CCLE_expression': featuring_train[2].shape[1],
    'CCLE_methylation': featuring_train[3].shape[1],
    'CCLE_miRNA': featuring_train[4].shape[1],
    'Mordred': featuring_train[5].shape[1],
    'DrugTax': featuring_train[6].shape[1]}
if use_sc == True:
    input_features_size['sc0'] = featuring_train[7].shape[1]

all_index = train_test_index
substruct_smiles_list, ddi_mask_H, V_SD, len_max, substructure_index = loadDrug(device)
len_max = 0
substruct_smiles_list1, ddi_mask_H1, substructure_index1 = loadDrug1(device)
set_seed(seed)
ddi_mask_H = torch.from_numpy(ddi_mask_H).to(device)
ddi_mask_H1 = torch.from_numpy(ddi_mask_H1).to(device)
molecule = pd.read_csv('./data/drug_smiles.csv')['Ismiles']
average_projection, smiles_list = buildPrjSmiles(molecule)
average_projection = average_projection.to(device)
molecule_graphs = graph_batch_from_smile(smiles_list)
drug_dataset = MoleculeDataset('./data/drug_smiles', dataset='drug_smiles')
dataset = random_split(drug_dataset, null_value=0, frac_train=1.0, frac_valid=0.0,
                       frac_test=0.0, seed=42)
drug_code = loadDrug2(len_max)
molecule_forward = {'batched_data': molecule_graphs.to(device)}
molecule_para = {
    'num_layer': 3, 'emb_dim': args.dim1, 'graph_pooling': 'mean',
    'drop_ratio': args.dp, 'gnn_type': 'gin', 'virtual_node': False
}
substruct_graphs1 = graph_batch_from_smile(substruct_smiles_list1)
substruct_graphs = V_SD
substruct_forward = {'batched_data': substruct_graphs}
substruct_forward1 = {'batched_data': substruct_graphs1.to(device)}
substruct_para = {
    'num_layer': 1, 'emb_dim': args.dim1, 'graph_pooling': 'mean',
    'drop_ratio': args.dp, 'gnn_type': 'gin', 'virtual_node': False, 'k':2,
    'drug_glofeat': drug_glofeat, 'cline_glofeat': cline_glofeat
}
drug_cell_emb = load_cells_datasets()
drug_data = {
            'substruct_data': substruct_forward,
            'mol_data': molecule_forward,
            'ddi_mask_H': ddi_mask_H,
            'average_projection': average_projection,
            'e': drug_code,
            'drug_cell_emb': drug_cell_emb,
            'substructure_index': substructure_index
}
drug_data1 = {
    'substruct_data': substruct_forward1,
    'mol_data': molecule_forward,
    'ddi_mask_H': ddi_mask_H1,
    'average_projection': average_projection,
    'e': drug_code,
    'drug_cell_emb': drug_cell_emb,
    'substructure_index': substructure_index1
}
scf_tr = Scf_index(dataset, device)
args1.num_tr_scf = scf_tr.num_scf
drug_scf_idx = get_drug_scf_idx(dataset)
class_model = torch.load('./result/drug_classify/classify_model.pth').to(device)

featuring_train, target = load_datasets(mode="train_test", use_sc_data=use_sc)
featuring_validation_cell, target_validation_cell = load_datasets(mode="valcell", use_sc_data=use_sc)
featuring_train = getDataload(featuring_train, all_index, batch_size, use_sc)
featuring_validation_cell = getDataload(featuring_validation_cell, valcell_index, batch_size, use_sc)


model = Mol_TopExpert_Mulit_DRP(hidden_layer_number=11, hidden_layer_size_cells=100, hidden_layer_size_drugs=32,
                                hidden_layer_size_single=76, use_single_cell=use_sc,
                                input_features_size=input_features_size,
                                global_para=molecule_para, substruct_para=substruct_para, emb_dim=args.dim,
                                global_dim=args.dim1,
                                substruct_dim=args.dim1, substruct_num=ddi_mask_H.shape[1],
                                substruct_num1=ddi_mask_H1.shape[1],
                                device=device, args=args1).to(device)
criterion = nn.MSELoss(reduction="none")
opt = torch.optim.Adam(model.parameters(), lr=0.001)
# pre_model.load_state_dict(torch.load('./model/prepare_model/prepare_model.pkl'))
print('-------------------------------train test val---------------------------------------')
train_val_cell_leave(model, class_model, featuring_train, featuring_validation_cell,
                     use_sc, device, 200, criterion, opt, drug_data, drug_data1,
                     scf_tr, drug_scf_idx)
