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
        parser.add_argument('--pad', type=int, default=-1, help='delete rate of dataset')
        parser.add_argument('--load_iter', type=int, default=0, help='delete rate of dataset')
        parser.add_argument('--load_epoch', type=int, default=0, help='delete rate of dataset')
        parser.add_argument('--oversample_foreground_percent', type=float, default=0.3, help='delete rate of dataset')
        parser.add_argument('--datamap_epoch', type=int, default=0, help='delete rate of dataset')
        parser.add_argument('--random_seed', type=int, default=1, help='delete rate of dataset')
        parser.add_argument('--del_score', type=str, default='datamap', help='delete rate of dataset')
        parser.add_argument('--single_iteration', type=int, default=40000, help='delete rate of dataset')

        args = parser.parse_args([])

        # setting server
        args.server = 'v100'
        if args.server == 'v100':
            args.server_path = '/data1'
        elif args.server == '506':
            args.server_path = '/HDD_data'
        elif args.server == '3090':
            args.server_path = '/data'
        elif args.server == 'root':
            args.server_path = '/root/autodl-tmp'
        else:
            raise Exception(f"There are not found server {args.server}")

        args.dataset = "msd_word"
        args.exp = 4
        args.gpu = '6'
        args.load_iter = 18000
        args.load_epoch = 0
        args.random_seed = 42

        args.del_flag = True
        args.del_rate = 0.4
        args.del_mode = 'keep ambiguous'
        args.del_score = 'datamap'
        args.datamap_epoch = 170
        # args.single_iteration = 15000
        args.oversample_foreground_percent = 0.4

        if not args.del_flag:
            args.del_mode = 'No_Prunning'
            args.del_score = 'None'
            args.del_rate = 0.0
        if args.del_mode == 'random':
            args.del_score = 'None'

        if args.load_iter != 0:
            args.resume =\
                f'{args.server_path}/HYK/DMDS_save/{args.dataset}/exp{args.exp}-GraNd/' \
                f'{args.del_rate}-{args.del_mode}-{args.del_score}-e{args.datamap_epoch}/models/iter_{args.load_iter}'
            if not args.del_flag:
                args.resume = \
                    f'{args.server_path}/HYK/DMDS_save/{args.dataset}/exp{args.exp}-GraNd/' \
                    f'No_Prunning/models/iter_{args.load_iter}'

        if args.load_epoch != 0:
            args.resume =\
                f'{args.server_path}/HYK/DMDS_save/{args.dataset}/exp4-GraNd/' \
                f'{args.del_rate}-{args.del_mode}-{args.del_score}-e{args.datamap_epoch}/models/epoch_{args.load_epoch}'


        args.lr = 3e-3
        args.batch_size = 4
        args.max_epoch = 40000
        args.max_iteration = 25000
        args.power = 0.9
        args.n_class = 2
        args.n_channels = 1
        args.patch_size = (64, 64, 64)
        args.patch_nums = 8
        args.pad = 64

        print(f'This experiment {args.exp} is for {args.del_mode}-{args.del_score}-{args.del_rate} on {args.dataset},'
              f' running in gpu{args.gpu}')
        print(f'Data Map plot in epoch {args.datamap_epoch}.')

        return args



