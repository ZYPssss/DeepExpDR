import numpy as np
import torch
import pandas as pd
import random
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from resources import model_evaluation
from sklearn.metrics import roc_auc_score, average_precision_score


def clf_loss(clf_outs, labels, assign, criterion, num_experts):
    is_valid = labels ** 2 > 0

    is_valid_tensor = torch.unsqueeze(is_valid, -1)
    is_valid_tensor = is_valid_tensor.repeat(1, 1, num_experts)

    ## calculate loss
    labels = torch.unsqueeze(labels, -1)
    labels = labels.repeat(1, 1, num_experts)  # N x tasks x heads
    loss_tensor = criterion(clf_outs, labels)

    #### modify loss based on assign index
    loss_mat = torch.sum(assign * loss_tensor, dim=0)  #
    num_valid_mat = torch.sum(assign * is_valid_tensor.long(), dim=0)  # tasks x head

    return loss_mat, num_valid_mat  # tasks x heads

def model_evaluation1(target_test, predicted_test, verbose=True, subset_type="test", write_mode=True):
    auc = roc_auc_score(target_test, predicted_test)
    aupr = average_precision_score(target_test, predicted_test)
    print('Currently evaluating: {}'.format(subset_type))
    print("AUC:", round(auc, 4), "AUPR:", round(aupr, 4))
    return [auc, aupr]
# train
def train_test(model, class_model, featuring_train, use_sc, device, criterion, opt, drug_data, drug_data1, scf_tr, drug_scf_idx):
    model.train()
    class_model.eval()
    for idx, data in enumerate(tqdm(featuring_train, desc='Iteration')):  # tqdm是进度条  返回 enumerate(枚举) 对象。
        # , cell_dict
        input = []
        drug_scf_idx = torch.tensor(drug_scf_idx).to(device)
        cell_chromatin, cell_copynumber, cell_expression, cell_methylation, cell_miRNA, mordred, drugTax, label, index = data
        cell_chromatin = torch.tensor(cell_chromatin).to(device)
        cell_copynumber = torch.tensor(cell_copynumber).to(device)
        cell_expression = torch.tensor(cell_expression).to(device)
        cell_methylation = torch.tensor(cell_methylation).to(device)
        cell_miRNA = torch.tensor(cell_miRNA).to(device)

        mordred = torch.tensor(mordred).to(device)
        drugTax = torch.tensor(drugTax).to(device)

        label = torch.tensor(label).to(device)
        index = torch.tensor(index).to(device)

        input.append(cell_chromatin)
        input.append(cell_copynumber)
        input.append(cell_expression)
        input.append(cell_methylation)
        input.append(cell_miRNA)
        input.append(mordred)
        input.append(drugTax)

        ic50_predicts = model(input, drug_data, drug_data1, index)
        with torch.no_grad():
            z, q = class_model(drug_data['mol_data'])
            g, q_idx = class_model.assign_head(q)
            drug_id = index[:, 0].long()
            g = g[drug_id]
        num_graph = len(data[0])
        num_experts = class_model.get_num_experts()
        labels = label.view(num_graph, -1).to(torch.float32)
        clf_loss_mat, num_valid_mat = clf_loss(ic50_predicts, labels, g, criterion, num_experts)
        classification_loss = torch.sum(clf_loss_mat / num_valid_mat)

        loss_total = classification_loss
        opt.zero_grad()
        loss_total.backward()
        opt.step()

def train_drug_leave(model, class_model, featuring_train, use_sc, device, criterion, opt, drug_data, drug_data1, scf_tr, drug_scf_idx):
    model.train()
    class_model.eval()
    for idx, data in enumerate(tqdm(featuring_train, desc='Iteration')):  # tqdm是进度条  返回 enumerate(枚举) 对象。
        # , cell_dict
        input = []
        drug_scf_idx = torch.tensor(drug_scf_idx).to(device)
        cell_chromatin, cell_copynumber, cell_expression, cell_methylation, cell_miRNA, mordred, drugTax, label, index = data
        cell_chromatin = torch.tensor(cell_chromatin).to(device)
        cell_copynumber = torch.tensor(cell_copynumber).to(device)
        cell_expression = torch.tensor(cell_expression).to(device)
        cell_methylation = torch.tensor(cell_methylation).to(device)
        cell_miRNA = torch.tensor(cell_miRNA).to(device)

        mordred = torch.tensor(mordred).to(device)
        drugTax = torch.tensor(drugTax).to(device)

        label = torch.tensor(label).to(device)
        index = torch.tensor(index).to(device)

        input.append(cell_chromatin)
        input.append(cell_copynumber)
        input.append(cell_expression)
        input.append(cell_methylation)
        input.append(cell_miRNA)
        input.append(mordred)
        input.append(drugTax)

        ic50_predicts = model(input, drug_data, drug_data1, index)
        with torch.no_grad():
            z, q = class_model(drug_data['mol_data'])
            g, q_idx = class_model.assign_head(q)
            drug_id = index[:, 0].long()
            g = g[drug_id]
        num_graph = len(data[0])
        num_experts = class_model.get_num_experts()
        labels = label.view(num_graph, -1).to(torch.float32)
        clf_loss_mat, num_valid_mat = clf_loss(ic50_predicts, labels, g, criterion, num_experts)
        classification_loss = torch.sum(clf_loss_mat / num_valid_mat)

        loss_total = classification_loss
        opt.zero_grad()
        loss_total.backward()
        opt.step()

def train_cell_leave(model, class_model, featuring_train, use_sc, device, criterion, opt, drug_data, drug_data1, scf_tr, drug_scf_idx):
    model.train()
    class_model.eval()
    for idx, data in enumerate(tqdm(featuring_train, desc='Iteration')):  # tqdm是进度条  返回 enumerate(枚举) 对象。
        # , cell_dict
        input = []
        drug_scf_idx = torch.tensor(drug_scf_idx).to(device)
        cell_chromatin, cell_copynumber, cell_expression, cell_methylation, cell_miRNA, mordred, drugTax, label, index = data
        cell_chromatin = torch.tensor(cell_chromatin).to(device)
        cell_copynumber = torch.tensor(cell_copynumber).to(device)
        cell_expression = torch.tensor(cell_expression).to(device)
        cell_methylation = torch.tensor(cell_methylation).to(device)
        cell_miRNA = torch.tensor(cell_miRNA).to(device)

        mordred = torch.tensor(mordred).to(device)
        drugTax = torch.tensor(drugTax).to(device)

        label = torch.tensor(label).to(device)
        index = torch.tensor(index).to(device)

        input.append(cell_chromatin)
        input.append(cell_copynumber)
        input.append(cell_expression)
        input.append(cell_methylation)
        input.append(cell_miRNA)
        input.append(mordred)
        input.append(drugTax)

        ic50_predicts = model(input, drug_data, drug_data1, index)
        with torch.no_grad():
            z, q = class_model(drug_data['mol_data'])
            g, q_idx = class_model.assign_head(q)
            drug_id = index[:, 0].long()
            g = g[drug_id]
        num_graph = len(data[0])
        num_experts = class_model.get_num_experts()
        labels = label.view(num_graph, -1).to(torch.float32)
        clf_loss_mat, num_valid_mat = clf_loss(ic50_predicts, labels, g, criterion, num_experts)
        classification_loss = torch.sum(clf_loss_mat / num_valid_mat)

        loss_total = classification_loss
        opt.zero_grad()
        loss_total.backward()
        opt.step()

# test
def test(model, class_model, featuring_test, use_sc, device, drug_data, drug_data1):
    model.eval()
    class_model.eval()
    with torch.no_grad():
        y_true = []
        y_pred = []
        y_index = []
        for idx, data in enumerate(tqdm(featuring_test, desc='Iteration')):  # tqdm是进度条  返回 enumerate(枚举) 对象。
            input = []
            cell_chromatin, cell_copynumber, cell_expression, cell_methylation, cell_miRNA, mordred, drugTax, label, index = data
            cell_chromatin = torch.tensor(cell_chromatin).to(device)
            cell_copynumber = torch.tensor(cell_copynumber).to(device)
            cell_expression = torch.tensor(cell_expression).to(device)
            cell_methylation = torch.tensor(cell_methylation).to(device)
            cell_miRNA = torch.tensor(cell_miRNA).to(device)

            mordred = torch.tensor(mordred).to(device)
            drugTax = torch.tensor(drugTax).to(device)

            label = torch.tensor(label).to(device)
            index = torch.tensor(index).to(device)

            input.append(cell_chromatin)
            input.append(cell_copynumber)
            input.append(cell_expression)
            input.append(cell_methylation)
            input.append(cell_miRNA)
            input.append(mordred)
            input.append(drugTax)

            ic50_predicts = model(input, drug_data, drug_data1, index)
            z, q_origin = class_model(drug_data['mol_data'])
            q, q_idx = class_model.assign_head(q_origin)
            drug_id = index[:, 0].long()
            g = q[drug_id]
            # for i in range(len(g)):
            #     max_index = torch.argmax(g[i][0], dim = 0)
            #     for j in range(len(g[i][0])):
            #         if(j == max_index):
            #             g[i][0][j] = 1
            #         else:
            #             g[i][0][j] = 0
            output = torch.sum(ic50_predicts * g, dim=-1).squeeze()

            # total_loss += F.mse_loss(output, label.float().to(device), reduction='sum')
            y_true.append(label)
            y_pred.append(output)
            y_index.append(index)
            # output_drug_cell_embeddings.append(o1)
            # cells_merged.append(o2)
    # Evaluate the performance of the model
    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    y_index = torch.cat(y_index, dim=0)
    target_test = y_true.cpu()
    predicted_test = y_pred.cpu()
    y_index = y_index.cpu()
    # output_drug_cell_embeddings = torch.cat(output_drug_cell_embeddings, dim=0)
    # cells_merged = torch.cat(cells_merged, dim=0)
    # output_drug_cell_embeddings = output_drug_cell_embeddings.cpu()
    # cells_merged = cells_merged.cpu()
    test_result = model_evaluation(target_test, predicted_test, verbose=True, subset_type="test", write_mode=True)
    return test_result, target_test, predicted_test, y_index

def test_cell_leave_out(model, class_model, featuring_validation_cell, use_sc, device, drug_data, drug_data1):
    model.eval()
    class_model.eval()
    with torch.no_grad():
        y_true = []
        y_pred = []
        y_index = []
        output_drug_cell_embeddings = []
        cells_merged = []
        total_loss = 0
        for idx, data in enumerate(tqdm(featuring_validation_cell, desc='Iteration')):  # tqdm是进度条  返回 enumerate(枚举) 对象。
            input = []
            cell_chromatin, cell_copynumber, cell_expression, cell_methylation, cell_miRNA, mordred, drugTax, label, index = data
            cell_chromatin = torch.tensor(cell_chromatin).to(device)
            cell_copynumber = torch.tensor(cell_copynumber).to(device)
            cell_expression = torch.tensor(cell_expression).to(device)
            cell_methylation = torch.tensor(cell_methylation).to(device)
            cell_miRNA = torch.tensor(cell_miRNA).to(device)

            mordred = torch.tensor(mordred).to(device)
            drugTax = torch.tensor(drugTax).to(device)

            label = torch.tensor(label).to(device)
            index = torch.tensor(index).to(device)

            input.append(cell_chromatin)
            input.append(cell_copynumber)
            input.append(cell_expression)
            input.append(cell_methylation)
            input.append(cell_miRNA)
            input.append(mordred)
            input.append(drugTax)

            ic50_predicts = model(input, drug_data, drug_data1, index)
            z, q_origin = class_model(drug_data['mol_data'])
            q, q_idx = class_model.assign_head(q_origin)
            drug_id = index[:, 0].long()
            g = q[drug_id]
            # for i in range(len(g)):
            #     max_index = torch.argmax(g[i][0], dim = 0)
            #     for j in range(len(g[i][0])):
            #         if(j == max_index):
            #             g[i][0][j] = 1
            #         else:
            #             g[i][0][j] = 0
            # output = []
            # for i in range(len(q_idx)):
            #     select_index = q_idx[i]
            #     output.append(ic50_predicts[i][0][select_index])
            # output = torch.tensor(output).to(device)
            output = torch.sum(ic50_predicts * g, dim=-1).squeeze()

            # total_loss += F.mse_loss(output, label.float().to(device), reduction='sum')
            y_true.append(label)
            y_pred.append(output)
            y_index.append(index)
            # output_drug_cell_embeddings.append(o1)
            # cells_merged.append(o2)
    # Evaluate the performance of the model
    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    y_index = torch.cat(y_index, dim=0)
    target_validation_cell = y_true.cpu()
    predicted_val_cells = y_pred.cpu()
    y_index = y_index.cpu()
    # output_drug_cell_embeddings = torch.cat(output_drug_cell_embeddings, dim=0)
    # cells_merged = torch.cat(cells_merged, dim=0)
    # output_drug_cell_embeddings = output_drug_cell_embeddings.cpu()
    # cells_merged = cells_merged.cpu()
    cell_leave_out_result = model_evaluation(target_validation_cell, predicted_val_cells, verbose=True, subset_type="val_cells",
                     write_mode=True)
    return cell_leave_out_result, target_validation_cell, predicted_val_cells, y_index

def test_drug_leave_out(model, class_model, featuring_validation_drug, use_sc, device, drug_data, drug_data1):
    model.eval()
    class_model.eval()
    with torch.no_grad():
        y_true = []
        y_pred = []
        y_index = []
        output_drug_cell_embeddings = []
        cells_merged = []
        total_loss = 0
        for idx, data in enumerate(tqdm(featuring_validation_drug, desc='Iteration')):  # tqdm是进度条  返回 enumerate(枚举) 对象。
            input = []
            cell_chromatin, cell_copynumber, cell_expression, cell_methylation, cell_miRNA, mordred, drugTax, label, index = data
            cell_chromatin = torch.tensor(cell_chromatin).to(device)
            cell_copynumber = torch.tensor(cell_copynumber).to(device)
            cell_expression = torch.tensor(cell_expression).to(device)
            cell_methylation = torch.tensor(cell_methylation).to(device)
            cell_miRNA = torch.tensor(cell_miRNA).to(device)

            mordred = torch.tensor(mordred).to(device)
            drugTax = torch.tensor(drugTax).to(device)

            label = torch.tensor(label).to(device)
            index = torch.tensor(index).to(device)

            input.append(cell_chromatin)
            input.append(cell_copynumber)
            input.append(cell_expression)
            input.append(cell_methylation)
            input.append(cell_miRNA)
            input.append(mordred)
            input.append(drugTax)

            ic50_predicts = model(input, drug_data, drug_data1, index)
            z, q_origin = class_model(drug_data['mol_data'])
            q, q_idx = class_model.assign_head(q_origin)
            drug_id = index[:, 0].long()
            g = q[drug_id]
            # for i in range(len(g)):
            #     max_index = torch.argmax(g[i][0], dim = 0)
            #     for j in range(len(g[i][0])):
            #         if(j == max_index):
            #             g[i][0][j] = 1
            #         else:
            #             g[i][0][j] = 0

            # output = []
            # for i in range(len(q_idx)):
            #     select_index = q_idx[i]
            #     output.append(ic50_predicts[i][0][select_index])
            # output = torch.tensor(output).to(device)
            # q1 = q[index[:, 0].long()]
            output = torch.sum(ic50_predicts * g, dim=-1).squeeze()

            # total_loss += F.mse_loss(output, label.float().to(device), reduction='sum')
            # output_drug_cell_embeddings.append(o1)
            # cells_merged.append(o2)
            y_true.append(label)
            y_pred.append(output)
            y_index.append(index)

    # Evaluate the performance of the model
    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    y_index = torch.cat(y_index, dim=0)
    target_validation_drug = y_true.cpu()
    predicted_val_drugs = y_pred.cpu()
    y_index = y_index.cpu()
    # output_drug_cell_embeddings = torch.cat(output_drug_cell_embeddings, dim=0)
    # cells_merged = torch.cat(cells_merged, dim=0)
    # output_drug_cell_embeddings = output_drug_cell_embeddings.cpu()
    # cells_merged = cells_merged.cpu()
    drug_leave_out_result = model_evaluation(target_validation_drug, predicted_val_drugs, verbose=True, subset_type="val_drugs",write_mode=True)
    return drug_leave_out_result, target_validation_drug, predicted_val_drugs, y_index

def train_val_drug_leave(model, class_model, featuring_train, featuring_validation_drug, use_sc,
              device, epochs, criterion, opt, drug_data, drug_data1, scf_tr, drug_scf_idx):
    best_drug_leave_out_result = [100, 100, -1, -1, 100, -1]
    for epoch in range(epochs):
        # train
        print('epoch:{}'.format(epoch))
        train_drug_leave(model, class_model, featuring_train, use_sc, device, criterion, opt, drug_data, drug_data1, scf_tr, drug_scf_idx)
        # drug-leave-out
        drug_leave_out_result, target_validation_drug, predicted_val_drugs, index = test_drug_leave_out(model, class_model,
                                                                                                 featuring_validation_drug,
                                                                                                 use_sc, device,
                                                                                         drug_data, drug_data1)

        # and (drug_leave_out_result[5] > best_drug_leave_out_result[5])
        if ((drug_leave_out_result[2] > best_drug_leave_out_result[2]) and (
                drug_leave_out_result[3] > best_drug_leave_out_result[3]) and (
                drug_leave_out_result[4] < best_drug_leave_out_result[4])):
            torch.save(model, './result/best_drug_leave_out_model/best_drug_leave_out_model.pth')
            np.savetxt('./result/best_drug_leave_out_model/label.csv',
                       np.array(target_validation_drug))
            np.savetxt('./result/best_drug_leave_out_model/score.csv', np.array(predicted_val_drugs))
            np.savetxt('./result/best_drug_leave_out_model/index.csv', np.array(index), delimiter=',')

            # np.savetxt('./result/best_drug_leave_out_model/output_drug_cell_embeddings({}).csv'.format(flod), np.array(output_drug_cell_embeddings))
            # np.savetxt('./result/best_drug_leave_out_model/cells_merged({}).csv'.format(flod), np.array(cells_merged))
            best_drug_leave_out_result = drug_leave_out_result
            np.savetxt('./result/best_drug_leave_out_model/result.csv',
                       np.array(best_drug_leave_out_result), delimiter=',')

    print('---------------------------Finish------------------------')

    print('best_drug_leave_out_result:')
    print("RMSE:", round(best_drug_leave_out_result[0], 4), "MSE:", round(best_drug_leave_out_result[1], 4), "Pearson:",
          round(best_drug_leave_out_result[2], 4), "r^2:", round(best_drug_leave_out_result[3], 4), "MAE:",
          round(best_drug_leave_out_result[4], 4), "Spearman:", round(best_drug_leave_out_result[5], 4))
    np.savetxt('./result/best_drug_leave_out_model/result.csv', np.array(best_drug_leave_out_result),
               delimiter=',')

def train_val_cell_leave(model, class_model, featuring_train, featuring_validation_cell, use_sc,
              device, epochs, criterion, opt, drug_data, drug_data1,  scf_tr, drug_scf_idx):
    best_cell_leave_out_result = [100, 100, -1, -1, 100, -1]
    for epoch in range(epochs):
        # train
        print('epoch:{}'.format(epoch))
        train_cell_leave(model, class_model, featuring_train, use_sc, device, criterion, opt, drug_data, drug_data1, scf_tr, drug_scf_idx)
        # cell-leave-out
        cell_leave_out_result, target_validation_cell, predicted_val_cells, index = test_cell_leave_out(model, class_model,
                                                                                                 featuring_validation_cell,
                                                                                                 use_sc, device,
                                                                                                 drug_data, drug_data1)
        if ((cell_leave_out_result[2] > best_cell_leave_out_result[2]) and (
                cell_leave_out_result[3] > best_cell_leave_out_result[3]) and (
                cell_leave_out_result[4] < best_cell_leave_out_result[4]) and (
                cell_leave_out_result[5] > best_cell_leave_out_result[5])):
            torch.save(model, './result/best_cell_leave_out_model/best_cell_leave_out_model.pth')
            np.savetxt('./result/best_cell_leave_out_model/label.csv',
                       np.array(target_validation_cell))
            np.savetxt('./result/best_cell_leave_out_model/score.csv', np.array(predicted_val_cells))
            np.savetxt('./result/best_cell_leave_out_model/index.csv', np.array(index), delimiter=',')

            # np.savetxt('./result/best_cell_leave_out_model/output_drug_cell_embeddings({}).csv'.format(flod), np.array(output_drug_cell_embeddings))
            # np.savetxt('./result/best_cell_leave_out_model/cells_merged({}).csv'.format(flod), np.array(cells_merged))
            best_cell_leave_out_result = cell_leave_out_result
            np.savetxt('./result/best_cell_leave_out_model/result.csv',
                       np.array(best_cell_leave_out_result), delimiter=',')

    print('---------------------------Finish------------------------')


    print('best_cell_leave_out_result:')
    print("RMSE:", round(best_cell_leave_out_result[0], 4), "MSE:", round(best_cell_leave_out_result[1], 4), "Pearson:",
          round(best_cell_leave_out_result[2], 4), "r^2:", round(best_cell_leave_out_result[3], 4), "MAE:",
          round(best_cell_leave_out_result[4], 4), "Spearman:", round(best_cell_leave_out_result[5], 4))
    np.savetxt('./result/best_cell_leave_out_model/result.csv', np.array(best_cell_leave_out_result),
               delimiter=',')


def train_val_test(model, class_model, featuring_train, featuring_test, use_sc,
              device, epochs, criterion, opt, drug_data, drug_data1, flod, scf_tr, drug_scf_idx):
    best_test_result = [100, 100, -1, -1, 100, -1]
    for epoch in range(epochs):
        # train
        print('epoch:{}'.format(epoch))
        train_test(model, class_model, featuring_train, use_sc, device, criterion, opt, drug_data, drug_data1, scf_tr, drug_scf_idx)
        # test
        test_result, target_test, predicted_test, index = test(model, class_model, featuring_test, use_sc, device, drug_data, drug_data1)
        if ((test_result[2] > best_test_result[2]) and (test_result[3] > best_test_result[3]) and
                (test_result[4] < best_test_result[4]) and (test_result[5] > best_test_result[5])):
            torch.save(model, './result/best_test_model/best_test_model({}).pth'.format(flod))
            np.savetxt('./result/best_test_model/label({}).csv'.format(flod), np.array(target_test))
            np.savetxt('./result/best_test_model/score({}).csv'.format(flod), np.array(predicted_test))
            np.savetxt('./result/best_test_model/index({}).csv'.format(flod), np.array(index), delimiter=',')

            # np.savetxt('./result/best_test_model/output_drug_cell_embeddings({}).csv'.format(flod), np.array(output_drug_cell_embeddings))
            # np.savetxt('./result/best_test_model/cells_merged({}).csv'.format(flod), np.array(cells_merged))
            best_test_result = test_result
            np.savetxt('./result/best_test_model/result({}).csv'.format(flod), np.array(best_test_result),
                       delimiter=',')

    print('---------------------------Flod:{} Finish------------------------'.format(flod))

    print('best_test_result:')
    print("RMSE:", round(best_test_result[0], 4), "MSE:", round(best_test_result[1], 4), "Pearson:",
          round(best_test_result[2], 4), "r^2:", round(best_test_result[3], 4), "MAE:",
          round(best_test_result[4], 4), "Spearman:", round(best_test_result[5], 4))
    np.savetxt('./result/best_test_model/result({}).csv'.format(flod), np.array(best_test_result), delimiter=',')


def train_val_drug_leave_1(model, class_model, featuring_train, featuring_validation_drug, use_sc,
              device, epochs, criterion, opt, drug_data, drug_data1, scf_tr, drug_scf_idx):
    drug_leave_out_result, target_validation_drug, predicted_val_drugs, index = test_drug_leave_out(model, class_model,
                                                                                                 featuring_validation_drug,
                                                                                                 use_sc, device,
                                                                                                 drug_data, drug_data1)



def train_val_cell_leave_1(model, class_model, featuring_train, featuring_validation_cell, use_sc,
              device, epochs, criterion, opt, drug_data, drug_data1,  scf_tr, drug_scf_idx):
    cell_leave_out_result, target_validation_cell, predicted_val_cells, index = test_cell_leave_out(model, class_model,
                                                                                                 featuring_validation_cell,
                                                                                                 use_sc, device,
                                                                                                 drug_data, drug_data1)


def train_val_test_1(model, class_model, featuring_train, featuring_test, use_sc,
              device, epochs, criterion, opt, drug_data, drug_data1, flod, scf_tr, drug_scf_idx):
    test_result, target_test, predicted_test, index = test(model, class_model, featuring_test, use_sc, device, drug_data, drug_data1)


def class_train(model, device, epochs, opt, drug_data, scf_tr, drug_scf_idx):
    drug_scf_idx = torch.tensor(drug_scf_idx).to(device)
    for epoch in range(epochs):
        print('epoch:{}'.format(epoch))
        model.train()
        z, q = model(drug_data['mol_data'])
        temp_q = q.data
        p = model.target_distribution(temp_q)
        q_idx = torch.argmax(q, dim=-1)
        count = np.zeros((1, len(q[0])))
        for i in range(len(q_idx)):
            j = q_idx[i]
            count[0][j] += 1
        g, q_idx = model.assign_head(q)
        num_graph = len(drug_scf_idx)
        scf_idx = drug_scf_idx.long()
        scf_idx = scf_tr.scfIdx_to_label[scf_idx]
        cluster_loss = F.kl_div(q.log(), p, reduction='sum')
        align_loss = model.alignment_loss(scf_idx, q)
        loss_total = (cluster_loss / num_graph) + align_loss
        print('{},  {},     all:{},   group:{}'.format((cluster_loss / num_graph), align_loss, loss_total, count[0]))
        opt.zero_grad()
        loss_total.backward()
        opt.step()

