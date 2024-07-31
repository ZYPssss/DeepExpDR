import argparse
import torch
parser = argparse.ArgumentParser('DrugParam')
# drug
parser.add_argument('--seed', type=int, default=0, help="Seed for splitting the dataset.")
parser.add_argument('--dataset_dir', type=str, default='./data', help='directory of dataset')
parser.add_argument('--dataset', type=str, default='drug', help='root directory of dataset')
parser.add_argument('--split', type=str, default="scaffold", help="random or scaffold or random_scaffold")
parser.add_argument('--device_no', type=int, default = 1,
                    help='which gpu to use if any (default: 0)')
parser.add_argument('--atten_emb', type=int, default=128, help="Attention embedding.")

# GNN model
parser.add_argument('-i', '-8-input_model_file', type=str, default='',
                    help='filename to read the model (if there is any)')
parser.add_argument('-c', '--ckpt_all', type=str, default='',
                    help='filename to read the model ')
parser.add_argument('--num_layer', type=int, default=5,
                    help='number of GNN message passing layers (default: 5).')
parser.add_argument('--emb_dim', type=int, default=256,
                    help='embedding dimensions (default: 300)')
parser.add_argument('--expert_emb', type=int, default=640,
                    help='expert embedding dimensions (default: 300)')
parser.add_argument('--dropout_ratio', type=float, default=0.6,
                    help='dropout ratio (default: 0.5)')
parser.add_argument('--graph_pooling', type=str, default="mean",
                    help='graph level pooling (sum, mean, max, set2set, attention)')
parser.add_argument('--JK', type=str, default="last",
                    help='how the node features across layers are combined. last, sum, max, concat')
parser.add_argument('--gnn_type', type=str, default="gat")
parser.add_argument('--dropout_ratio1', type=float, default=0.2,
                        help='dropout ratio')
# train
parser.add_argument('--batch_size', type=int, default=1,
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataset loading')

# optimizer
parser.add_argument('--lr', type=float, default=0.00001,
                    help='learning rate (default: 0.001)')
parser.add_argument('--decay', type=float, default=0,
                    help='weight decay (default: 0)')
## loss balance
parser.add_argument('--alpha', type=float, default=0.1, help="balance parameter for clustering")
parser.add_argument('--beta', type=float, default=0.01, help="balance parameter for alignment")

## clustering
parser.add_argument('--min_temp', type=float, default=1, help=" temperature for gumble softmax, annealing")
parser.add_argument('--num_experts', type=int, default = 3)
parser.add_argument('--gate_dim', type=int, default=80, help="gate embedding space dimension, 50 or 300")
parser.add_argument('--layer', type=int, default=3, help='number of GNN layer')
parser.add_argument('--hidden_dim', type=int, default=5, help='hidden dim for cell')
parser.add_argument('--layer_drug', type=int, default=3, help='layer for drug')
parser.add_argument('--dim_drug', type=int, default=128, help='hidden dim for drug')
args1 = parser.parse_args()

parser1 = argparse.ArgumentParser(description='Model')
parser1.add_argument('--epochs', type=int, default=400,
                    metavar='N', help='number of epochs to train')
parser1.add_argument('--lr', type=float, default=0.001,
                    metavar='FLOAT', help='learning rate')
parser1.add_argument('--embed_dim', type=int, default=128,
                    metavar='N', help='embedding dimension')
parser1.add_argument('--weight_decay', type=float, default=0.0005,
                    metavar='FLOAT', help='weight decay')
parser1.add_argument('--droprate', type=float, default=0.3,
                    metavar='FLOAT', help='dropout rate')
parser1.add_argument('--batch_size', type=int, default=128,
                    metavar='N', help='input batch size for training')
parser1.add_argument('--test_batch_size', type=int, default=128,
                    metavar='N', help='input batch size for testing')
parser1.add_argument('--rawpath', type=str, default='data/',
                    metavar='STRING', help='rawpath')
parser1.add_argument('--device', type=str, default='cuda:2',
                    help='device')
parser1.add_argument('--patience', type=int, default=10,
                    help='patience for earlystopping (default: 10)')
parser1.add_argument('--mode', type=str, default='train',
                    help='train or test')
parser1.add_argument('--edge', type=float, default=0.9, help='threshold for cell line graph')
parser1.add_argument('--weight_path', type=str, default='best2',
                    help='filepath for pretrained weights')
parser1.add_argument('--alpha', type=float, default=0.2, help="balance parameter for clustering")
parser1.add_argument('--dim', default=64, type=int, help='model dimension')
parser1.add_argument('--dim1', default=64, type=int, help='model dimension')
parser1.add_argument('--dp', default=0.7, type=float, help='dropout ratio')
parser1.add_argument(
    '--embedding', action='store_true',
    help='use embedding table for substructures' +
         'if it\'s not chosen, the substructure will be encoded by GNN'
)
args = parser1.parse_args()
args1.device = torch.device("cuda:" + str(args1.device_no)) if torch.cuda.is_available() else torch.device("cpu")
args1.num_tasks = 1

