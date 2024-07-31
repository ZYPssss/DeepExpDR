from collections import defaultdict
from copy import deepcopy

import torch
import codecs
from subword_nmt.apply_bpe import BPE
from tqdm import tqdm
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import random
from rdkit.Chem import BRICS
from rdkit import Chem
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, roc_auc_score
import pickle

DEFAULT_LOCATION = "./"


TRAIN_SPLIT = DEFAULT_LOCATION + "data/splits/train_features.csv"
TEST_SPLIT = DEFAULT_LOCATION + "data/splits/test_features.csv"
CELL_SPLIT = DEFAULT_LOCATION + "data/splits/cell_validation.csv"
DRUG_SPLIT = DEFAULT_LOCATION + "data/splits/drug_validation.csv"

TARGET_LOC = DEFAULT_LOCATION + "data/mycellsGDSC.h5"
LEAVE_CELL_LOC = DEFAULT_LOCATION + "data/splits/leave_out/cell_validation.txt"
LEAVE_DRUG_LOC = DEFAULT_LOCATION + "data/splits/leave_out/drug_validation.txt"

DRUG_SMILES_LOC = DEFAULT_LOCATION + "data/drug_smiles.csv"

SPLITS_LOCATION = DEFAULT_LOCATION + "data/processed/"
ORIGINAL_DATA_LOC = DEFAULT_LOCATION + "data/original/"

RANDOM_STATE = 42

def open_txt(input_file):
	opened_file = open(input_file, "r").readlines()
	return [x.replace("\n", "") for x in opened_file]


def model_evaluation(input_class, input_predictions, subset_type = "test", verbose = False, write_mode = False):
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error, \
                                    r2_score
    import math
    from scipy.stats import pearsonr, spearmanr
    if verbose == True:
        print("Currently evaluating:",subset_type)

    try:
        list_input_class = list(input_class.iloc[:,0])
    except:
        list_input_class = list(input_class)
    list_input_predictions = list(input_predictions)
    try:
        RMSE = math.sqrt(mean_squared_error(input_class, input_predictions))
    except:
        RMSE = 10000
    try:
        MSE = mean_squared_error(input_class, input_predictions)
    except:
        MSE = 10000
    try:
        Pearson, pval_Pearson = pearsonr([float(x) for x in list_input_class], [float(x) for x in list_input_predictions])
    except:
        Pearson = -1.0
    try:
        r2 = r2_score(input_class, input_predictions)
    except:
        r2 = -1.0
    try:
        MAE = mean_absolute_error(input_class, input_predictions)
    except:
        MAE = 10000
    try:
        Spearman, pval_Spearman = spearmanr(list_input_class, list_input_predictions)
    except:
        Spearman = -1.0
    if verbose == True:
        print("RMSE:", round(RMSE, 4), "MSE:" , round(MSE, 4), "Pearson:", round(Pearson, 4), "r^2:", round(r2, 4), "MAE:", round(MAE, 4), "Spearman:", round(Spearman, 4))
        
    if write_mode == True:
        output_file_name = DEFAULT_LOCATION + subset_type + ".csv"
        with open(output_file_name, "w") as output_file:
            output_file.write("Metric,Value\n")
            output_file.write("RMSE," + str(RMSE) + "\n")
            output_file.write("MSE," + str(MSE) + "\n")
            output_file.write("Pearson," + str(Pearson) + "\n")
            output_file.write("r^2," + str(r2) + "\n")
            output_file.write("MAE," + str(MAE) + "\n")
            output_file.write("Spearman," + str(Spearman) + "\n")
    return [RMSE, MSE, Pearson, r2, MAE, Spearman]

def model_evaluation_classify(input_class, input_predictions):

    auc = roc_auc_score(input_class, input_predictions)
    aupr = average_precision_score(input_class, input_predictions)
    score = deepcopy(input_predictions)
    for i in range(len(score)):
        if(input_predictions[i] > 0.5):
            score[i] = 1
        else:
            score[i] = 0
    acc = accuracy_score(input_class, score)
    f1 = f1_score(input_class, score)
    print("AUC:", round(auc, 4), "AUPR:", round(aupr, 4), "ACC:", round(acc, 4),
          "F1:", round(f1, 4))


    return [auc, aupr, acc, f1]


class Scf_index:
    def __init__(self, dataset, device):
        self.device = device

        self.max_scf_idx = None
        self.scfIdx_to_label = None
        self.num_scf = None

        self.get_scf_idx(dataset)

    def get_scf_idx(self, dataset):
        ''''
        scf label: scf 에 속한 sample 수가 많은 순서부터 desending order 로 sorting 해서 label 매김
        self.num_train_scf = train set에 있는 scf 의 종류
        self.
        '''

        scf = defaultdict(int)
        max_scf_idx = 0
        for data in dataset:
            idx = data.scf_idx.item()
            scf[idx] += 1
            if max_scf_idx < idx:
                max_scf_idx = idx
        self.max_scf_idx = max_scf_idx
        scf = sorted(scf.items(), key=lambda x: x[1], reverse=True)

        self.scfIdx_to_label = torch.ones(max_scf_idx + 1).to(torch.long).to(torch.long) * -1
        self.scfIdx_to_label = self.scfIdx_to_label.to(self.device)

        for i, k in enumerate(scf):
            self.scfIdx_to_label[k[0]] = i

        self.num_scf = len(scf)

def get_drug_scf_idx(dataset):
    drug_scf_idx = np.zeros(len(dataset))
    for data in dataset:
        idx = data.scf_idx.item()
        drug_id = data.id.item()
        drug_scf_idx[drug_id] = idx
    return drug_scf_idx


def init_centroid(model, z_s, num_experts):
    z_s_arr = z_s.detach().cpu().numpy()

    num_data = z_s_arr.shape[0]
    if num_data > 35000:
        mask_idx = list(range(num_data))
        random.shuffle(mask_idx)
        z_s_arr = z_s_arr[mask_idx[:35000]]

    kmeans = KMeans(n_clusters=num_experts, random_state=0).fit(z_s_arr)
    centroids = kmeans.cluster_centers_

    model.Mol_substruct_Topexpert.cluster.data = torch.tensor(centroids).to(model.Mol_substruct_Topexpert.cluster.device)


def get_z(model, featuring_train, drug_data, drug_data1, device):
    model.train()
    z_s = []
    for idx, data in enumerate(tqdm(featuring_train, desc='Iteration')):  # tqdm是进度条  返回 enumerate(枚举) 对象。
        # , cell_dict
        model.train()
        input = []
        cell_chromatin, cell_copynumber, cell_expression, cell_methylation, cell_miRNA, mordred, drugTax, label, index, drug_fingers_matrix, fused_network, cell_ic_sim, d1, d2 = data
        cell_chromatin = torch.tensor(cell_chromatin).to(device)
        cell_copynumber = torch.tensor(cell_copynumber).to(device)
        cell_expression = torch.tensor(cell_expression).to(device)
        cell_methylation = torch.tensor(cell_methylation).to(device)
        cell_miRNA = torch.tensor(cell_miRNA).to(device)

        mordred = torch.tensor(mordred).to(device)
        drugTax = torch.tensor(drugTax).to(device)

        label = torch.tensor(label).to(device)
        index = torch.tensor(index).to(device)

        drug_fingers_matrix = torch.tensor(drug_fingers_matrix).to(device)
        fused_network = torch.tensor(fused_network).to(device)
        cell_ic_sim = torch.tensor(cell_ic_sim).to(device)

        d1 = torch.tensor(d1).to(device)
        d2 = torch.tensor(d2).to(device)

        input.append(cell_chromatin)
        input.append(cell_copynumber)
        input.append(cell_expression)
        input.append(cell_methylation)
        input.append(cell_miRNA)
        input.append(mordred)
        input.append(drugTax)
        drugs = [d1, d2]
        cells = torch.cat((cell_chromatin, cell_copynumber, cell_expression, cell_methylation, cell_miRNA), dim=1)
        _, z, _ = model(input, drug_data, drug_data1, index, drug_fingers_matrix, fused_network,
                                    cell_ic_sim, d1, d2)
        z_s.append(z)

    z_s = torch.cat(z_s, dim=0)
    return z_s

def set_seed(seed):

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def drug2emb_encoder(smile, len_max):
    vocab_path = "./ESPF/drug_codes_chembl_freq_1500.txt"
    sub_csv = pd.read_csv("./ESPF/subword_units_map_chembl_freq_1500.csv")

    bpe_codes_drug = codecs.open(vocab_path)
    dbpe = BPE(bpe_codes_drug, merges=-1, separator='')

    idx2word_d = sub_csv['index'].values
    words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

    max_d = len_max
    t1 = dbpe.process_line(smile).split()  # split
    try:
        i1 = np.asarray([words2idx_d[i] for i in t1])  # index
    except:
        i1 = np.array([0])

    l = len(i1)
    if l < max_d:
        i = np.pad(i1, (0, max_d - l), 'constant', constant_values=0)
        input_mask = ([1] * l) + ([0] * (max_d - l))
    else:
        i = i1[:max_d]
        input_mask = [1] * max_d

    return i, np.asarray(input_mask)

def loadDrug(device):
    fraction = []
    NDCList = pd.read_csv('./data/drug_smiles.csv')['Ismiles']
    vocab_path = "./ESPF/drug_codes_chembl_freq_1500.txt"
    sub_csv = pd.read_csv("./ESPF/subword_units_map_chembl_freq_1500.csv")
    bpe_codes_drug = codecs.open(vocab_path)
    dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
    idx2word_d = sub_csv['index'].values
    words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

    with open('./data/substructure_data/drugs_substructure.pkl', 'rb') as f:
        # 使用pickle.load()从文件中读取序列化的对象并还原为原来的Python对象
        fracSet = pickle.load(f)

    i1 = [words2idx_d[i] for i in fracSet]
    i1 = np.asarray(i1)
    l = len(i1)
    input_mask = ([1] * (l-1))
    input_mask.append(0)
    input_mask = np.asarray(input_mask)
    List = []
    List.append(torch.tensor(i1).to(device))
    List.append(torch.tensor(input_mask).to(device))
    ddi_matrix = np.zeros((len(NDCList), len(fracSet)))
    substructure_index = []
    sub_index = []
    mask = []
    for i, SMILES in enumerate(NDCList):
        m = dbpe.process_line(SMILES).split()
        index = []
        mask1 = []
        for frac in m:
            x = fracSet.index(frac)
            ddi_matrix[i, x] += 1
            index.append(x)
            mask1.append(1)
        for k in range(70 - len(index)):
            index.append(0)
            mask1.append(0)

        sub_index.append(index)
        mask.append(mask1)
    sub_index =torch.tensor(sub_index).to(device)
    mask = torch.tensor(mask).to(device)
    substructure_index.append(sub_index)
    substructure_index.append(mask)
    return fracSet, ddi_matrix, List, 50, substructure_index

def loadDrug2(len_max):
    drug_code = []
    e1 = []
    e2 = []
    NDCList = pd.read_csv('./data/drug_smiles.csv')['Ismiles']
    for SMILES in NDCList:
        i1, i2 = drug2emb_encoder(SMILES, len_max)
        e1.append(i1)
        e2.append(i2)
    e1 = torch.tensor(e1)
    e2 = torch.tensor(e2)
    drug_code.append(e1)
    drug_code.append(e2)
    return drug_code

def loadDrug1(device):
    fraction = []
    NDCList = pd.read_csv('./data/drug_smiles.csv')['Ismiles']
    with open('./data/substructure_data/drugs_substructure1.pkl', 'rb') as f:
        # 使用pickle.load()从文件中读取序列化的对象并还原为原来的Python对象
        fracSet = pickle.load(f)
    ddi_matrix = np.zeros((len(NDCList), len(fracSet)))
    substructure_index = []
    sub_index = []
    mask = []
    for i, SMILES in enumerate(NDCList):
        try:
            m = list(BRICS.BRICSDecompose(Chem.MolFromSmiles(SMILES)))
            m.sort()
            index1 = []
            mask1 = []
            for frac in m:
                ddi_matrix[i, fracSet.index(frac)] = 1
                index1.append(fracSet.index(frac))
                mask1.append(1)
            for j in range(20 - len(index1)):
                index1.append(0)
                mask1.append(0)
            sub_index.append(index1)
            mask.append(mask1)
        except:
            pass
    sub_index = torch.tensor(sub_index).to(device)
    mask = torch.tensor(mask).to(device)
    substructure_index.append(sub_index)
    substructure_index.append(mask)


    return fracSet, ddi_matrix, substructure_index


# Load Training datasets
def load_datasets1(mode="train", use_sc_data=True):
    single_cells = 10
    chrom = pd.read_csv(SPLITS_LOCATION + "bulk/chromatin_" + mode + "_features.csv")
    cn = pd.read_csv(SPLITS_LOCATION + "bulk/copynumber_" + mode + "_features.csv")
    expr = pd.read_csv(SPLITS_LOCATION + "bulk/expression_" + mode + "_features.csv")
    meth = pd.read_csv(SPLITS_LOCATION + "bulk/methylation_" + mode + "_features.csv")
    mirna = pd.read_csv(SPLITS_LOCATION + "bulk/mirna_" + mode + "_features.csv")
    mordred = pd.read_csv(SPLITS_LOCATION + "drug/mordred_" + mode + "_features.csv")
    drugtax = pd.read_csv(SPLITS_LOCATION + "drug/drugtax_" + mode + "_features.csv")

    featuring = [chrom, cn, expr, meth, mirna, mordred, drugtax]
    target = featuring[0]["LN_IC50"]

    # CREATE FUNCTION FOR LOAD_SC
    # Load single-cell data
    def load_sc(feature_list, mode=mode):
        cell_line_list = list(feature_list[0]["CELL_LINE_NAME"].unique())
        sc_list = {}
        for i in range(single_cells):
            sc_list['sc_' + str(i)] = []

        if mode == "valcell":
            for sel_cel in cell_line_list:
                df = pd.read_hdf(SPLITS_LOCATION + "single-cell/individual_valid/" + sel_cel + ".h5")
                i = 0
                for index, row in df.iterrows():
                    dam = pd.DataFrame(row).T
                    sc_list['sc_' + str(i)].append(dam)
                    i += 1

        else:
            for sel_cel in cell_line_list:
                df = pd.read_hdf(SPLITS_LOCATION + "single-cell/individual_tt/" + sel_cel + ".h5")
                i = 0
                for index, row in df.iterrows():
                    dam = pd.DataFrame(row).T
                    sc_list['sc_' + str(i)].append(dam)
                    i += 1

        ind_sing = []
        for i in sc_list:
            sing = pd.concat(sc_list[i], ignore_index=True)
            sing = feature_list[0].iloc[:, :4].merge(sing)
            ind_sing.append(sing)

        return ind_sing

    if use_sc_data == True:
        sc_data = load_sc(feature_list=featuring)
        featuring += sc_data

    for i in range(len(featuring)):
        featuring[i] = featuring[i].iloc[:, 4:]
        featuring[i] = np.asarray(featuring[i]).astype('float32')

    return featuring, target

def load_datasets(mode="train", use_sc_data=True):
    single_cells = 10
    chrom = np.loadtxt('./data/bulk/chromatin.csv', dtype=float, delimiter=',')
    cn = np.loadtxt('./data/bulk/copynumber.csv', dtype=float, delimiter=',')
    expr = np.loadtxt('./data/bulk/expression.csv', dtype=float, delimiter=',')
    meth = np.loadtxt('./data/bulk/methylation.csv', dtype=float, delimiter=',')
    mirna = np.loadtxt('./data/bulk/mirna.csv', dtype=float, delimiter=',')
    mordred = np.loadtxt('./data/bulk/mordred.csv', dtype=float, delimiter=',')
    drugtax = np.loadtxt('./data/bulk/drugtax.csv', dtype=float, delimiter=',')

    featuring = [chrom, cn, expr, meth, mirna, mordred, drugtax]
    target = np.loadtxt('./data/bulk/{}_target.csv'.format(mode), dtype=float)

    # for i in range(len(featuring)):
    #     featuring[i] = featuring[i].iloc[:, 4:]
    #     featuring[i] = np.asarray(featuring[i]).astype('float32')
    # target = np.array(target)
    return featuring, target

def load_cells_datasets():
    chrom = np.loadtxt('./data/bulk/chromatin.csv', dtype=float, delimiter=',')
    cn = np.loadtxt('./data/bulk/copynumber.csv', dtype=float, delimiter=',')
    expr = np.loadtxt('./data/bulk/expression.csv', dtype=float, delimiter=',')
    meth = np.loadtxt('./data/bulk/methylation.csv', dtype=float, delimiter=',')
    mirna = np.loadtxt('./data/bulk/mirna.csv', dtype=float, delimiter=',')
    mordred = np.loadtxt('./data/bulk/mordred.csv', dtype=float, delimiter=',')
    drugtax = np.loadtxt('./data/bulk/drugtax.csv', dtype=float, delimiter=',')

    featuring = [chrom, cn, expr, meth, mirna, mordred, drugtax]

    # for i in range(len(featuring)):
    #     featuring[i] = featuring[i].iloc[:, 4:]
    #     featuring[i] = np.asarray(featuring[i]).astype('float32')
    # target = np.array(target)
    return featuring

def set_seed(seed):

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)