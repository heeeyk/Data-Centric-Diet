import argparse


class Params:
    def __init__(self):
        self.x = 2

    @staticmethod
    def trainparam():
        parser = argparse.ArgumentParser(description='Training.')
        parser.add_argument('--dataset', type=str, default='msd_pancreas', help='The dataset to be trained')
        parser.add_argument('--gpu', type=str, default='0', help='GPU to be used')
        parser.add_argument('--exp', type=int, default=1, help='index of experiment')
        parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
        parser.add_argument('--lambda_pe', type=float, default=0.0, help='position encoding weight')
        parser.add_argument('--resume', type=str, default=None, help='resume point')
        parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
        parser.add_argument('--dm_path', type=str, default=None, help='save path for data map')
        parser.add_argument('--del_flag', type=bool, default=False, help='flag for delete dataset')
        parser.add_argument('--del_rate', type=int, default=0, help='delete rate of dataset')
        parser.add_argument('--del_mode', type=str, default='hard', help='delete hard or easy learning data in dataset')
        parser.add_argument('--dicefile_load_path', type=str, default=None, help='loadding path of dice file')
        parser.add_argument('--server', type=str, default=0, help='server of running code')
        parser.add_argument('--server_path', type=str, default=0, help='server of running code')
        parser.add_argument('--max_epoch', type=int, default=0, help='delete rate of dataset')
        parser.add_argument('--max_iteration', type=int, default=0, help='delete rate of dataset')
        parser.add_argument('--power', type=int, default=0, help='delete rate of dataset')
        parser.add_argument('--n_class', type=int, default=0, help='delete rate of dataset')
        parser.add_argument('--n_channels', type=int, default=0, help='delete rate of dataset')
        parser.add_argument('--patch_size', type=tuple, default=(128, 128, 128), help='delete rate of dataset')
        parser.add_argument('--patch_nums', type=tuple, default=1, help='delete rate of dataset')
        parser.add_argument('--oversample_foreground_percent', type=float, default=0.3, help='delete rate of dataset')
        args = parser.parse_args([])

        # setting server
        args.server = 'v100'
        if args.server == 'v100':
            args.server_path = '/data1'
        elif args.server == '506':
            args.server_path = '/HDD_data'
        else:
            raise Exception(f"There are not found server {args.server}")

        args.dataset = "msd"
        args.gpu = '1'
        args.exp = 1
        # args.resume = '/data1/HYK/DMDS_save/msd_only/exp0/models/iter_2000'

        args.del_flag = False

        if not args.del_flag:
            args.del_mode = 'whole'
            args.del_rate = 0.0

        args.lr = 3e-3
        args.batch_size = 4
        args.max_epoch = 40000
        args.max_iteration = 20000
        args.power = 0.9
        args.n_class = 2
        args.n_channels = 1
        args.patch_size = (64, 64, 64)
        args.patch_nums = 8
        args.pad = 64
        args.oversample_foreground_percent = 0.6


        return args



