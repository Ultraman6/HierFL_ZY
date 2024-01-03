import json
import logging
import math
import numpy as np
import os
import sys
import random

import torch
from torch.utils import data
from tqdm import trange
import math

from options import args_parser


def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum( np.exp(x))
    return ex/sum_ex


def generate_synthetic(dataset_name, args):
    # 从 args 对象中提取参数
    alpha = args.alpha
    beta = args.beta
    iid = args.syn_iid
    NUM_USER = args.num_clients
    dimension = args.dimension if hasattr(args, 'dimension') else 60
    NUM_CLASS = args.num_class if hasattr(args, 'num_class') else 10
    # 生成文件名和路径
    train_path = f"{dataset_name}/train.json"
    test_path = f"{dataset_name}/test.json"
    # 创建必要的目录
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_path), exist_ok=True)

    # 初始化数据结构
    samples_per_user = np.random.lognormal(4, 2, (NUM_USER)).astype(int) + 50
    X = [[] for _ in range(NUM_USER)]
    y = [[] for _ in range(NUM_USER)]
    diagonal = np.zeros(dimension)
    for j in range(dimension):
        diagonal[j] = np.power((j + 1), -1.2)
    cov_x = np.diag(diagonal)

    # 数据生成逻辑
    if iid == -1:  # IID 数据生成
        mean_x = np.zeros((NUM_USER, dimension))
        W = np.random.normal(0, 1, (dimension, NUM_CLASS))
        b = np.random.normal(0, 1, NUM_CLASS)
        for i in range(NUM_USER):
            xx = np.random.multivariate_normal(mean_x[i], cov_x, samples_per_user[i])
            for j in range(samples_per_user[i]):
                tmp = np.dot(xx[j], W) + b
                yy = np.argmax(softmax(tmp))
                X[i].append(xx.tolist())
                y[i].append(yy)
    else:  # 非IID 数据生成
        mean_W = np.random.normal(0, alpha, NUM_USER)
        mean_b = mean_W
        B = np.random.normal(0, beta, NUM_USER)
        if iid == 1:
            W_global = np.random.normal(0, 1, (dimension, NUM_CLASS))
            b_global = np.random.normal(0, 1, NUM_CLASS)
        for i in range(NUM_USER):
            if iid == 1:
                mean_x = np.ones(dimension) * B[i]  # 对于 iid=1, 每个用户的 mean_x 是 B[i] 乘以全1向量
            else:
                mean_x = np.random.normal(B[i], 1, dimension)  # 每个用户有不同的 mean_x
            W = np.random.normal(mean_W[i], 1, (dimension, NUM_CLASS)) if iid == 0 else W_global
            b = np.random.normal(mean_b[i], 1, NUM_CLASS) if iid == 0 else b_global
            xx = np.random.multivariate_normal(mean_x, cov_x, samples_per_user[i])
            for j in range(samples_per_user[i]):
                tmp = np.dot(xx[j], W) + b
                yy = np.argmax(softmax(tmp))
                X[i].append(xx.tolist())
                y[i].append(yy)

    # 创建数据结构
    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}

    # 数据分配与存储
    for i in range(NUM_USER):
        uname = f'f_{i:05d}'
        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)
        num_samples = len(X[i])
        train_len = int(0.9 * num_samples)
        test_len = num_samples - train_len

        train_data['users'].append(uname)
        train_data['user_data'][uname] = {'x': X[i][:train_len], 'y': y[i][:train_len]}
        train_data['num_samples'].append(train_len)
        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': X[i][train_len:], 'y': y[i][train_len:]}
        test_data['num_samples'].append(test_len)

    # 将生成的数据保存到文件
    with open(train_path, 'w') as outfile:
        json.dump(train_data, outfile)
    with open(test_path, 'w') as outfile:
        json.dump(test_data, outfile)

    print(f"New dataset {dataset_name} generated and saved.")

def load_partition_data_federated_synthetic(args):
    dataset_name = f"{args.dataset_root}/synthetic_{'iid' if args.syn_iid == -1 else f'{args.alpha}_{args.beta}'}_{args.num_clients}"
    train_file_path = f"{dataset_name}/train.json"
    test_file_path = f"{dataset_name}/test.json"
    # 检查数据集是否存在，如果不存在则生成
    if not (os.path.exists(train_file_path) and os.path.exists(test_file_path)):
        generate_synthetic(dataset_name, args)
        logging.info("load_partition_data_federated_synthetic_1_1 START")
    # with open(train_file_path, "r") as train_f, open(test_file_path, "r") as test_f:
    #     train_data = json.load(train_f)
    #     test_data = json.load(test_f)
    #
    #     client_ids_train = train_data[args.num_clients]
    #     client_ids_test = test_data[args.num_clients]
    #
    #     full_x_train = torch.from_numpy(np.asarray([])).float()
    #     full_y_train = torch.from_numpy(np.asarray([])).long()
    #     full_x_test = torch.from_numpy(np.asarray([])).float()
    #     full_y_test = torch.from_numpy(np.asarray([])).long()
    #     train_data_local_dict = dict()
    #     test_data_local_dict = dict()
    #
    #     for i in range(len(client_ids_train)):
    #         train_ds = data.TensorDataset(
    #             torch.tensor(train_data["user_data"][client_ids_train[i]]["x"]),
    #             torch.tensor(
    #                 train_data["user_data"][client_ids_train[i]]["y"], dtype=torch.int64
    #             ),
    #         )
    #         test_ds = data.TensorDataset(
    #             torch.tensor(train_data["user_data"][client_ids_test[i]]["x"]),
    #             torch.tensor(
    #                 train_data["user_data"][client_ids_test[i]]["y"], dtype=torch.int64
    #             ),
    #         )
    #         train_dl = data.DataLoader(
    #             dataset=train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False
    #         )
    #         test_dl = data.DataLoader(
    #             dataset=test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False
    #         )
    #         train_data_local_dict[i] = train_dl
    #         test_data_local_dict[i] = test_dl
    #
    #         full_x_train = torch.cat(
    #             (
    #                 full_x_train,
    #                 torch.tensor(train_data["user_data"][client_ids_train[i]]["x"]),
    #             ),
    #             0,
    #         )
    #         full_y_train = torch.cat(
    #             (
    #                 full_y_train,
    #                 torch.tensor(
    #                     train_data["user_data"][client_ids_train[i]]["y"],
    #                     dtype=torch.int64,
    #                 ),
    #             ),
    #             0,
    #         )
    #         full_x_test = torch.cat(
    #             (
    #                 full_x_test,
    #                 torch.tensor(test_data["user_data"][client_ids_test[i]]["x"]),
    #             ),
    #             0,
    #         )
    #         full_y_test = torch.cat(
    #             (
    #                 full_y_test,
    #                 torch.tensor(
    #                     test_data["user_data"][client_ids_test[i]]["y"],
    #                     dtype=torch.int64,
    #                 ),
    #             ),
    #             0,
    #         )
    #
    #     train_ds = data.TensorDataset(full_x_train, full_y_train)
    #     test_ds = data.TensorDataset(full_x_test, full_y_test)
    #     train_data_global = data.DataLoader(
    #         dataset=train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False
    #     )
    #     test_data_global = data.DataLoader(
    #         dataset=test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False
    #     )
    #     train_data_num = len(train_data_global.dataset)
    #     test_data_num = len(test_data_global.dataset)
    #     data_local_num_dict = {
    #         i: len(train_data_local_dict[i].dataset) for i in train_data_local_dict
    #     }
    #
    # return (
    #     train_data_num,
    #     test_data_num,
    #     train_data_global,
    #     test_data_global,
    #     data_local_num_dict,
    #     train_data_local_dict,
    #     test_data_local_dict,
    # )

def main():
    load_partition_data_federated_synthetic(args_parser())
    train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    test_data = {'users': [], 'user_data':{}, 'num_samples':[]}


if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    main()

