from datasets.cifar_mnist import show_distribution, get_mnist, get_cifar10, get_femnist


def get_dataset(dataset_root, dataset, args):
    # trains, train_loaders, tests, test_loaders = {}, {}, {}, {}
    if dataset == 'mnist':
        train_loaders, test_loaders, share_data_edge, v_test_loader = get_mnist(dataset_root, args)
    elif dataset == 'cifar10':
        train_loaders, test_loaders, share_data_edge, v_test_loader = get_cifar10(dataset_root, args)
    elif dataset == 'femnist':
        train_loaders, test_loaders, share_data_edge, v_test_loader = get_femnist(dataset_root, args)
    elif dataset == 'synthetic_0.5_0.5':
        train_loaders, test_loaders, share_data_edge, v_test_loader = get_synthetic(dataset_root, args)
    elif dataset == 'synthetic_0_0':
        train_loaders, test_loaders, share_data_edge, v_test_loader = get_synthetic(dataset_root, args)
    elif dataset == 'synthetic_1_1':
        train_loaders, test_loaders, share_data_edge, v_test_loader = get_synthetic(dataset_root, args)
    elif dataset == 'synthetic_iid':
        train_loaders, test_loaders, share_data_edge, v_test_loader = get_synthetic(dataset_root, args)
    else:
        raise ValueError('Dataset `{}` not found'.format(dataset))
    return train_loaders, test_loaders, share_data_edge, v_test_loader

def get_dataloaders(args):
    """
    :param args:
    :return: A list of trainloaders, a list of testloaders, a concatenated trainloader and a concatenated testloader
    """
    if args.dataset in ['mnist', 'cifar10', "femnist", "synthetic"]: # 新增合成数据集
        train_loaders, test_loaders, share_data_edge, v_test_loader = get_dataset(dataset_root=args.dataset_root,
                                                                                       dataset=args.dataset,
                                                                                       args = args)
    else:
        raise ValueError("This dataset is not implemented yet")
    return train_loaders, test_loaders, share_data_edge, v_test_loader
