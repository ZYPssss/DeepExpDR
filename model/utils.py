from ogb.utils import smiles2graph
from torch_geometric.data import Data
import numpy as np
import torch
from rdkit import Chem


def graph_batch_from_smile(smiles_list):
    edge_idxes, edge_feats, node_feats, lstnode, batch = [], [], [], 0, []
    graphs = [smiles2graph(x) for x in smiles_list]
    for idx, graph in enumerate(graphs):
        edge_idxes.append(graph['edge_index'] + lstnode)
        edge_feats.append(graph['edge_feat'])
        node_feats.append(graph['node_feat'])
        lstnode += graph['num_nodes']
        batch.append(np.ones(graph['num_nodes'], dtype=np.int64) * idx)

    result = {
        'edge_index': np.concatenate(edge_idxes, axis=-1),
        'edge_attr': np.concatenate(edge_feats, axis=0),
        'batch': np.concatenate(batch, axis=0),
        'x': np.concatenate(node_feats, axis=0)
    }

    result = {k: torch.from_numpy(v) for k, v in result.items()}
    result['num_nodes'] = lstnode
    return Data(**result)


def buildPrjSmiles(molecule, device="cpu:0"):

    average_index, smiles_all = [], []

    # print(len(med_voc.items()))  # 131
    smilesList = list(molecule)

    """Create each data with the above defined functions."""
    counter = 0  # counter how many drugs are under that ATC-3
    for smiles in smilesList:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            smiles_all.append(smiles)
            counter += 1
        else:
            print('[SMILES]', smiles)
            print('[Error] Invalid smiles')
    average_index.append(counter)



    """Transform the above each data of numpy
    to pytorch tensor on a device (i.e., CPU or GPU).
    """
    # transform into projection matrix
    n_col = sum(average_index)
    n_row = len(average_index)

    average_projection = np.zeros((n_row, n_col))
    col_counter = 0
    for i, item in enumerate(average_index):
        average_projection[i, col_counter: col_counter + item] = 1 / item
        col_counter += item

    print("Smiles Num:{}".format(len(smiles_all)))
    print("n_col:{}".format(n_col))
    print("n_row:{}".format(n_row))

    return torch.FloatTensor(average_projection), smiles_all
