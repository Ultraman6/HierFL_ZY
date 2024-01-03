import json

import numpy as np
import torch
from torch.utils import data


def get_dataloader(train_file_path, test_file_path, args):
    with open(train_file_path, "r") as train_f, open(test_file_path, "r") as test_f:
        train_data = json.load(train_f)
        test_data = json.load(test_f)

        client_ids_train = train_data[args.num_clients]
        client_ids_test = test_data[args.num_clients]

        full_x_train = torch.from_numpy(np.asarray([])).float()
        full_y_train = torch.from_numpy(np.asarray([])).long()
        full_x_test = torch.from_numpy(np.asarray([])).float()
        full_y_test = torch.from_numpy(np.asarray([])).long()
        train_data_local_dict = dict()
        test_data_local_dict = dict()

        for i in range(len(client_ids_train)):
            train_ds = data.TensorDataset(
                torch.tensor(train_data["user_data"][client_ids_train[i]]["x"]),
                torch.tensor(
                    train_data["user_data"][client_ids_train[i]]["y"], dtype=torch.int64
                ),
            )
            test_ds = data.TensorDataset(
                torch.tensor(train_data["user_data"][client_ids_test[i]]["x"]),
                torch.tensor(
                    train_data["user_data"][client_ids_test[i]]["y"], dtype=torch.int64
                ),
            )
            train_dl = data.DataLoader(
                dataset=train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False
            )
            test_dl = data.DataLoader(
                dataset=test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False
            )
            train_data_local_dict[i] = train_dl
            test_data_local_dict[i] = test_dl

            full_x_train = torch.cat(
                (
                    full_x_train,
                    torch.tensor(train_data["user_data"][client_ids_train[i]]["x"]),
                ),
                0,
            )
            full_y_train = torch.cat(
                (
                    full_y_train,
                    torch.tensor(
                        train_data["user_data"][client_ids_train[i]]["y"],
                        dtype=torch.int64,
                    ),
                ),
                0,
            )
            full_x_test = torch.cat(
                (
                    full_x_test,
                    torch.tensor(test_data["user_data"][client_ids_test[i]]["x"]),
                ),
                0,
            )
            full_y_test = torch.cat(
                (
                    full_y_test,
                    torch.tensor(
                        test_data["user_data"][client_ids_test[i]]["y"],
                        dtype=torch.int64,
                    ),
                ),
                0,
            )

        train_ds = data.TensorDataset(full_x_train, full_y_train)
        test_ds = data.TensorDataset(full_x_test, full_y_test)
        train_data_global = data.DataLoader(
            dataset=train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False
        )
        test_data_global = data.DataLoader(
            dataset=test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False
        )
        train_data_num = len(train_data_global.dataset)
        test_data_num = len(test_data_global.dataset)
        data_local_num_dict = {
            i: len(train_data_local_dict[i].dataset) for i in train_data_local_dict
        }

    return (
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
    )