from datasets.synthetic.data_loader import get_dataloader
from datasets.synthetic.synthetic_iid.generate_iid import get_save_iid
from options import args_parser


def get_synthetic(args):
    # if args.mode==1:
    #     train, test = get_save_iid()
    # elif args.mode==2:
    #     train, test = get_save_iid()
    # elif args.mode==3:
    #     train, test = get_save_iid()
    # else:
    #     train, test = get_save_iid()
    train, test = get_save_iid()
    dataset_loader = get_dataloader(train, test, args)

if __name__ == "__main__":
    get_synthetic(args_parser())