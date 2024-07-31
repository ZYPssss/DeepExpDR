import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
import numpy as np
import random
use_scq = False
def Chebyshev_Distance(matrix):
    sim = np.zeros((len(matrix), len(matrix)))
    for A in range(len(matrix)):
        for B in range(len(matrix)):
            sim[A][B] = np.linalg.norm(matrix[A]-matrix[B],ord=np.inf)
    return sim

def getDataload(dataset, index, batch_size, use_sc):
    use_scq = use_sc
    dataset = MyDataset(dataset, index, use_sc)
    collate_fn = _collate
    dataset = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn = _collate, num_workers = 0)
    return dataset

def getDataload1(dataset, index, use_sc):
    use_scq = use_sc
    dataset = MyDataset1(dataset, index, use_sc)
    collate_fn1 = _collate1
    dataset = DataLoader(dataset, batch_size=256, shuffle=True, collate_fn = _collate1, num_workers=0)
    return dataset

def _collate(samples):
    cell_chromatin, cell_copynumber, cell_expression, cell_methylation, cell_miRNA, mordred, drugTax, label, index = map(
        list, zip(*samples))
    return cell_chromatin, cell_copynumber, cell_expression, cell_methylation, cell_miRNA, mordred, drugTax, label, index


class MyDataset(Dataset):
    def __init__(self, dataset, index, use_sc):
        super(MyDataset, self).__init__()
        self.dataset = dataset
        self.index = index
        self.use_sc = use_sc
    def __len__(self):
        return len(self.index)
    def __getitem__(self, index):
        return (self.dataset[0][int(self.index[index][1])], self.dataset[1][int(self.index[index][1])],
                self.dataset[2][int(self.index[index][1])], self.dataset[3][int(self.index[index][1])],
                self.dataset[4][int(self.index[index][1])], self.dataset[5][int(self.index[index][0])],
                self.dataset[6][int(self.index[index][0])], self.index[index][2], self.index[index][0:2])


def _collate1(samples):
    if(use_scq==False):
        cell_chromatin, cell_copynumber, cell_expression, cell_methylation, cell_miRNA, mordred, drugTax, label, index = map(list, zip(*samples))
        return cell_chromatin, cell_copynumber, cell_expression, cell_methylation, cell_miRNA, mordred, drugTax, label, index
    else:
        cell_chromatin, cell_copynumber, cell_expression, cell_methylation, cell_miRNA, mordred, drugTax, sc0, label, index  = map(list,zip(*samples))
        return cell_chromatin, cell_copynumber, cell_expression, cell_methylation, cell_miRNA, mordred, drugTax, sc0, label, index

class MyDataset1(Dataset):
    def __init__(self, dataset, index, use_sc):
        super(MyDataset1, self).__init__()
        self.dataset = dataset
        self.index = index
        self.use_sc = use_sc
    def __len__(self):
        return len(self.index)
    def __getitem__(self, index):
        if(self.use_sc == True):
            return (self.dataset[0][int(self.index[index][1])], self.dataset[1][int(self.index[index][1])], self.dataset[2][int(self.index[index][1])], self.dataset[3][int(self.index[index][1])],
                    self.dataset[4][int(self.index[index][1])], self.dataset[5][int(self.index[index][0])], self.dataset[6][int(self.index[index][0])], self.dataset[7][int(self.index[index][1])], self.index[index][2], self.index[index][0:2]
                    )
        else:
            return (self.dataset[0][int(self.index[index][1])], self.dataset[1][int(self.index[index][1])], self.dataset[2][int(self.index[index][1])], self.dataset[3][int(self.index[index][1])],
                    self.dataset[4][int(self.index[index][1])], self.dataset[5][int(self.index[index][0])], self.dataset[6][int(self.index[index][0])], self.index[index][2], self.index[index][0:2]
                    )


