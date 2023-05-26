import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from models.vnet import VNet
from datasets.hdf5 import NpzDataset
from datasets.paths import get_paths, get_test_paths
from datasets.transforms import RandomCrop, ToTensor, RandomTranspose, RandomRotate, PositionEncoding, RandomFlip
from tensorboardX import SummaryWriter
from trainer.training_pretrained_vnet import trainbox
from trainer.utils import load_data_test
from datasets.dataset_loading import load_dataset, DataLoader3D
from nnunet.paths import preprocessing_output_dir
from trainer.utils import DataMap
import random
from params import params_msd_nih_for_hard_example_delete as params_file


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True




cpu_num = 4
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)

args = params_file.Params.trainparam()
# 设置随机数种子
setup_seed(args.random_seed)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu



dataset_root = f'{args.server_path}/HYK/DATASET'
snapshot_path = f'{args.server_path}/HYK/DMDS_save/{args.dataset}'
# snapshot_exp = f'{snapshot_path}/{snapshot_prefix}-{args.del_rate}-{args.del_mode}'
if args.del_flag:
    snapshot_exp = f'{snapshot_path}/exp{args.exp}/{args.del_rate}-{args.del_mode}-{args.del_score}-0-e{args.datamap_epoch}'
else:
    snapshot_exp = f'{snapshot_path}/exp{args.exp}/No_Prunning'
args.dm_path = f'{snapshot_exp}/data_maps'

# Create file saveing path for models, data map and dice result.
os.makedirs(snapshot_exp, exist_ok=True)
os.makedirs(args.dm_path, exist_ok=True)
os.makedirs(f'{snapshot_exp}/dice_files', exist_ok=True)
os.makedirs(f'{snapshot_exp}/results', exist_ok=True)
os.makedirs(f'{snapshot_exp}/runs', exist_ok=True)

# set up transforms
train_transforms = [
    # RandomCrop(args.patch_size, args.patch_nums, pad=args.pad, is_binary=True),
    RandomTranspose(),
    RandomRotate()]

if args.lambda_pe > 0:
    train_transforms.append(PositionEncoding(args.lambda_pe))
    n_channels = 4
train_transforms.append(ToTensor())


if __name__ == "__main__":
    # save args params
    argsDict = args.__dict__
    with open(f'{snapshot_exp}/setting.txt', 'w') as f:
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')

    iter_num = 0
    total_epoch = 0
    if args.resume is not None:
        net = torch.load(args.resume, map_location='cuda:0')
        iter_num = int(args.resume.split('_')[-1])
        print('resuming from {} with iter_num {}'.format(args.resume, iter_num))
    else:
        net = VNet(n_channels=args.n_channels, n_classes=args.n_class)
    net.cuda()

    # set dataset

    test_root = f'/{args.server_path}/HYK/DATASET/nnUNet_preprocessed/Task003_Pancreas/msd_nih_to_word/test'

    t = "Task003_Pancreas"
    p_path = os.path.join(preprocessing_output_dir, t, f"{args.server_path}/HYK/DATASET/nnUNet_preprocessed/Task003_Pancreas/stage0_pkl")
    n_path = os.path.join(preprocessing_output_dir, t, "msd_nih_to_word/train")
    all_cases = os.listdir(n_path)
    eval_list, train_list = [], []
    for i in all_cases:
        if args.dataset == "msd_nih" and 'word' in i:
            eval_list.append(i.split('.')[0])
        if args.dataset == "msd_word" and 'nih' in i:
            eval_list.append(i.split('.')[0])
        if args.dataset == "nih_word" and 'msd' in i:
            eval_list.append(i.split('.')[0])
        else:
            train_list.append(i.split('.')[0])
    if args.del_mode == 'random':
        del_list = np.random.choice(train_list, size=int(args.del_rate * len(train_list)), replace=False)
        eval_list.extend(list(del_list))

    dices_root = f'{args.server_path}/HYK/DMDS_save/msd_nih/exp2/No_Prunning/dice_files'
    dice_path = os.listdir(dices_root)
    dice_path.sort()

    if args.del_flag:
        # for i in dice_path:
        #     if f'_e{args.datamap_epoch}_' in i:
        #         x = np.load(f'{dices_root}/{i}', allow_pickle=True).item()
        #         datamap = DataMap(args, save_path=f'{args.server_path}/HYK/DMDS_save/msd_only/exp1/data_maps',
        #                           dice_file=x)
        #         del_list = datamap.data_select(args.del_rate, args.del_mode, args.del_score)
        #         break

        # root = f'{args.server_path}/HYK/DMDS_save/msd_nih/exp2/No_Prunning/dice_files'
        dice_path = os.listdir(dices_root)
        dice_path.sort()
        e = f'_e{args.datamap_epoch}_'
        for i in dice_path:
            if e in i:
                x = np.load(f'{dices_root}/{i}', allow_pickle=True).item()
                break
        datamap1 = DataMap(args, save_path=f'{args.server_path}/HYK/DMDS_save/msd_only/exp1/data_maps', dice_file=x)
        wait_to_del1 = datamap1.data_select(args.del_rate, args.del_mode, args.del_score)
        eval_list.extend(wait_to_del1)

    print(f'we will train {len(all_cases) - len(eval_list)} cases')
    dataset = load_dataset(p_path, n_path, 1000, del_list=eval_list)

    dl = DataLoader3D(
        data=dataset,
        patch_size=args.patch_size,
        final_patch_size=args.patch_size,
        batch_size=args.batch_size,
        oversample_foreground_percent=args.oversample_foreground_percent,
        shuffle=True,
        patch_nums=args.patch_nums,
        transforms=transforms.Compose(train_transforms),

    )

    dice_file = {}
    for i, name in enumerate(list(dataset.keys())):
        dice_file.update({
            f'{name}': {  # 每个样本的名字，我们根据这个来挑选数据
                'idx': i,  # 每个样本对应的id号
                'dice': np.zeros(0, dtype='float32'),  # 每次迭代计算得到的分数
                'corr': np.zeros(0, dtype='float32'),  # 同上
                'conf': 0,  # 置信度
                'var': 0,  # 方差
                'mean_corr': 0,
                'nums': 0,  # 每个样本训练的次数
                'dataset': name.split('_')[0]}  #
            }
        )

    # set record files
    writer = SummaryWriter(f'{snapshot_exp}/runs', comment='test')

    batch_dice = torch.zeros(args.batch_size)
    batch_corr = torch.zeros(args.batch_size)

    all_cases = os.listdir(test_root)
    list.sort(all_cases)
    images, labels = [], []
    eval_cases = 6
    for i in range(eval_cases):
        if i < 3:
            image, label = load_data_test(f'{test_root}/{all_cases[i]}', dataset='msd', convert=True, npz=True)
        else:
            image, label = load_data_test(f'{test_root}/{all_cases[i]}', dataset='nih', convert=True, npz=True)
        images.append(image)
        labels.append(label)
    # all_cases = os.listdir(root_dir)
    # list.sort(all_cases)
    # for i in range(eval_cases):
    #     if i < 3:
    #         root_dir, list_path, test_root, test_list = get_paths('msd')
    #         image, label = load_data_test(f'{root_dir}/{all_cases[i]}', dataset='msd', convert=True, npz=True)
    #     else:
    #         image, label = load_data_test(f'{root_dir}/{all_cases[i]}', dataset='nih', convert=True, npz=True)
    #     images.append(image)
    #     labels.append(label)

    net.train()

    # train
    trainbox(
        net=net,
        dataloader=dl,
        dice_file=dice_file,
        writer=writer,
        iter_num=iter_num,
        snapshot_exp=snapshot_exp,
        args=args,
        eval_images=images, eval_labels=labels, eval_cases=6,
        dataset=dataset,
    )

