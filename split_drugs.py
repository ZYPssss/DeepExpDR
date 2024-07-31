import torch
import numpy as np
from rdkit.Chem import BRICS
import pandas as pd
import codecs
from subword_nmt.apply_bpe import BPE
import random
import pickle
from rdkit import Chem


def spilt_drug(seed):
    fraction = []
    NDCList = pd.read_csv('./data/drug_smiles.csv')['Ismiles']
    vocab_path = "./ESPF/drug_codes_chembl_freq_1500.txt"
    sub_csv = pd.read_csv("./ESPF/subword_units_map_chembl_freq_1500.csv")
    bpe_codes_drug = codecs.open(vocab_path)
    dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
    idx2word_d = sub_csv['index'].values
    words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))
    len_max = 0
    trmpF_set = []
    for SMILES in NDCList:
            try:
                m = dbpe.process_line(SMILES).split()
                for frac in m:
                    if(frac not in trmpF_set):
                        trmpF_set.append(frac)
            except:
                pass

    fracSet = list(set(trmpF_set))
    random.seed(seed)
    random.shuffle(fracSet)
    with open('./data/substructure_data/drugs_substructure.pkl', 'wb') as f:
        # 使用pickle.dump()将字典对象序列化并保存到文件中
        pickle.dump(fracSet, f)

def split_drug1(seed):
    fraction = []
    NDCList = pd.read_csv('./data/drug_smiles.csv')['Ismiles']
    fracSet = []
    lenmax = 0
    for SMILES in NDCList:
        try:
            m = list(BRICS.BRICSDecompose(Chem.MolFromSmiles(SMILES)))
            m.sort()
            if(lenmax< len(m)):
                lenmax = len(m)
            for frac in m:
                if (frac not in fracSet):
                    fracSet.append(frac)
        except:
            pass

    random.seed(seed)
    random.shuffle(fracSet)
    with open('./data/substructure_data/drugs_substructure1.pkl', 'wb') as f:
        # 使用pickle.dump()将字典对象序列化并保存到文件中
        pickle.dump(fracSet, f)

split_drug1(6)