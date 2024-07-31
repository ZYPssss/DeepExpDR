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
# np.random.seed(42)
# torch.manual_seed(42)
# random.seed(42)
from sklearn.model_selection import KFold
from train_val import train_val_drug_leave, class_train, train_val_drug_leave_1
from Dataload.MyData import  MyDataset, _collate, getDataload
from model.utils import buildPrjSmiles, graph_batch_from_smile
from model.Mol_TopExpert_Mulit_DRP import Mol_TopExpert_Mulit_DRP
from model.Mol_substruct_Topexpert import CLassfiy_gate
from agrs.args import args, args1
import codecs
from subword_nmt.apply_bpe import BPE
from model.Mol_substruct1 import Mol_substruct1
# Defining whether to use scRNA-seq data or not and, if using, the number of single cells per cell line that will be used
use_sc = False
single_cells = 10
args1.device = torch.device("cuda:" + str(1)) if torch.cuda.is_available() else torch.device("cpu")
device = args1.device
seed = 42
set_seed(seed)

def train_class():
    len_max = 70
    molecule = pd.read_csv('./data/drug_smiles.csv')['Ismiles']
    average_projection, smiles_list = buildPrjSmiles(molecule)
    molecule_graphs = graph_batch_from_smile(smiles_list)
    drug_dataset = MoleculeDataset('./data/drug_smiles', dataset='drug_smiles')
    dataset = random_split(drug_dataset, null_value=0, frac_train=1.0, frac_valid=0.0,
                           frac_test=0.0, seed=42)
    molecule_forward = {'batched_data': molecule_graphs.to(device)}
    molecule_para = {
        'num_layer': 3, 'emb_dim': args.dim1, 'graph_pooling': 'mean',
        'drop_ratio': args.dp, 'gnn_type': 'gin', 'virtual_node': False
    }
    drug_data1 = {
        'mol_data': molecule_forward,
    }
    scf_tr = Scf_index(dataset, device)
    args1.num_tr_scf = scf_tr.num_scf
    drug_scf_idx = get_drug_scf_idx(dataset)
    class_model = CLassfiy_gate(global_dim=args.dim1, num_tr_scf=args1.num_tr_scf, global_para=molecule_para, device=device).to(device)
    class_opt = torch.optim.Adam(class_model.parameters(), lr=0.001)
    print('-------------------------------class train---------------------------------------')
    class_train(class_model, device, 900, class_opt, drug_data1, scf_tr, drug_scf_idx)
    torch.save(class_model, './data/classify_model.pth')

train_class()

