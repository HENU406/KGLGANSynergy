import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops

from utils import evaluate
from parse import parse_method, parser_add_main_args
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, accuracy_score, roc_curve
import copy
import sklearn.metrics as m
from torch.utils.data import DataLoader
from dgllife.utils import EarlyStopping

from load_data import readKGData, readRecData

import warnings

warnings.filterwarnings('ignore')


def eval_classification(labels, logits):
    auc = roc_auc_score(y_true=labels, y_score=logits)
    p, r, t = precision_recall_curve(y_true=labels, probas_pred=logits)
    aupr = m.auc(r, p)
    fpr, tpr, threshold = roc_curve(labels, logits)
    # 利用Youden's index计算阈值
    spc = 1 - fpr
    j_scores = tpr - fpr
    best_youden, youden_thresh, youden_sen, youden_spc = sorted(zip(j_scores, threshold, tpr, spc))[-1]
    predicted_label = copy.deepcopy(logits)
    youden_thresh = round(youden_thresh, 3)
    # print(youden_thresh)

    predicted_label = [1 if i >= youden_thresh else 0 for i in predicted_label]
    p_1 = evaluate.precision(y_true=labels, y_pred=predicted_label)
    r_1 = evaluate.recall(y_true=labels, y_pred=predicted_label)
    acc = accuracy_score(y_true=labels, y_pred=predicted_label)
    f1 = f1_score(y_true=labels, y_pred=predicted_label)
    return p_1, r_1, acc, auc, aupr, f1


def validate_new(valid_set, model):
    model.eval()
    valid_set = torch.LongTensor(valid_set)
    dataloader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            d1 = batch[:, 0]
            d2 = batch[:, 1]
            c = batch[:, 2]
            labels = batch[:, 3]
            logits = model(entity_tensor, edge_tensor, d1, d2, c)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        p, r, acc, auc, aupr, f1 = eval_classification(all_labels, all_logits)
        return p, r, acc, auc, aupr, f1


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


### Parse args ###
parser = argparse.ArgumentParser(description='Training Pipeline for Node Classification')
parser_add_main_args(parser)
args = parser.parse_args()

precision_all = []
recall_all = []
accuracy_all = []
auc_all = []
aupr_all = []
f1_all = []
### Training loop ###
for run in range(args.runs):

    print(args)

    fix_seed(args.seed)

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    entity_tensor, edge_tensor = readKGData(args.data_dir)
    triples = readRecData(args.data_dir)  # [d1, d2, c, s, fold]

    n = entity_tensor.shape[0]  # 实体数
    e = edge_tensor.shape[1]  # 边数

    out_channels = 2  # 分类数
    d = args.n_dim  # 结点维度
    edge_dim = args.lgat_hidden_channels
    print(f"dataset {args.dataset} | num nodes {n} | num edge {e} | num node feats {d} | num classes {out_channels}")

    edge_tensor = to_undirected(edge_tensor)

    edge_tensor, _ = remove_self_loops(edge_tensor)
    edge_tensor, _ = add_self_loops(edge_tensor, num_nodes=n)

    edge_tensor, entity_tensor = edge_tensor.to(device), entity_tensor.to(device)


    comb_tensor = torch.tensor(triples)
    kf = KFold(n_splits=args.split, shuffle=True, random_state=args.seed)

    precision_cv = []
    recall_cv = []
    accuracy_cv = []
    auc_cv = []
    aupr_cv = []
    f1_cv = []

    for split, (train_index, test_index) in enumerate(kf.split(comb_tensor)):

        model = parse_method(args, n, edge_tensor.shape[1], edge_dim, out_channels, d, device)

        criterion = nn.BCELoss()

        if args.method == 'KGLGANSynergy':
            optimizer = torch.optim.Adam([
                {'params': model.params1, 'weight_decay': args.trans_weight_decay},
                {'params': model.params2, 'weight_decay': args.gnn_weight_decay}
            ],
                lr=args.lr)
        else:
            optimizer = torch.optim.Adam(
                model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
        best_val = float('-inf')

        stopper = EarlyStopping(mode='lower', filename='mainsplit-attention-comb-r', patience=args.patience)

        for epoch in range(args.epochs):
            model.train()
            all_loss = 0.0

            precision_batch = []
            recall_batch = []
            accuracy_batch = []
            auc_batch = []
            aupr_batch = []
            f1_batch = []
            for batch in DataLoader(comb_tensor[train_index], batch_size=args.batch_size, shuffle=True):
                d1 = batch[:, 0]  # 第一列
                d2 = batch[:, 1]  # 第二列
                c = batch[:, 2]  # 第三列
                y_true = batch[:, 3]  # 第四列

                y_pred = model(entity_tensor, edge_tensor, d1, d2, c)

                optimizer.zero_grad()

                loss = criterion(y_pred, y_true.float().to(device))

                loss.backward()
                optimizer.step()
                all_loss += loss.item()

            train_loss_avg = all_loss / (len(train_index) // args.batch_size)
            p, r, acc, auc, aupr, f1 = validate_new(comb_tensor[test_index], model)
            print(
                'train {}: Precision {:.4f} | Recall {:.4f} | accuracy {:.4f} | auc {:.4f} | aupr {:.4f} | F1 {:.4f} | loss {:.4f}'.format(
                    epoch, p, r, acc, auc, aupr, f1, train_loss_avg))

            early_stop = stopper.step(train_loss_avg, model)
            if early_stop:
                break
        stopper.load_checkpoint(model)

        p, r, acc, auc, aupr, f1 = validate_new(comb_tensor[test_index], model)
        precision_cv.append(p)
        recall_cv.append(r)
        accuracy_cv.append(acc)
        auc_cv.append(auc)
        aupr_cv.append(aupr)
        f1_cv.append(f1)
        print(
            'test {}: Precision {:.4f} | Recall {:.4f} | accuracy {:.4f} | auc {:.4f} | aupr {:.4f} | F1 {:.4f}'.format(
                split, p, r, acc, auc, aupr, f1))


    precision_cv_mean = np.mean(precision_cv)
    recall_cv_mean = np.mean(recall_cv)
    accuracy_cv_mean = np.mean(accuracy_cv)
    auc_cv_mean = np.mean(auc_cv)
    aupr_cv_mean = np.mean(aupr_cv)
    f1_cv_mean = np.mean(f1_cv)

    precision_all.append(precision_cv_mean)
    recall_all.append(recall_cv_mean)
    accuracy_all.append(accuracy_cv_mean)
    auc_all.append(auc_cv_mean)
    aupr_all.append(aupr_cv_mean)
    f1_all.append(f1_cv_mean)
    print(
        'result {}: Precision {:.4f} | Recall {:.4f} | accuracy {:.4f} | auc {:.4f} | aupr {:.4f} | F1 {:.4f}'.format(
            run, precision_cv_mean, recall_cv_mean, accuracy_cv_mean, auc_cv_mean, aupr_cv_mean, f1_cv_mean))

print('=*' * 50)

for i in range(args.runs):
    print(
        f'result {i}: Precision {precision_all[i]:.4f} | Recall {recall_all[i]:.4f} | '
        f'acc {accuracy_all[i]:.4f} | auc {auc_all[i]:.4f} | aupr {aupr_all[i]:.4f} | F1 {f1_all[i]:.4f}')

print(
    f'final: Precision {np.mean(precision_all):.4f}({np.std(precision_all):.4f}) | '
    f'Recall {np.mean(recall_all):.4f}({np.std(recall_all):.4f}) | '
    f'acc {np.mean(accuracy_all):.4f}({np.std(accuracy_all):.4f}) | '
    f'auc {np.mean(auc_all):.4f}({np.std(auc_all):.4f}) | '
    f'aupr {np.mean(aupr_all):.4f}({np.std(aupr_all):.4f}) | '
    f'F1 {np.mean(f1_all):.4f}({np.std(f1_all):.4f})')
