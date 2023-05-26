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
from trainer.training_vnet import trainbox
from trainer.utils import load_data_test
from datasets.dataset_loading import load_dataset, DataLoader3D
from params import params_msd_only_for_hard_example_delete as params_file
from nnunet.paths import preprocessing_output_dir

cpu_num = 4
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)

args = params_file.Params.trainparam()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

dataset_root = f'{args.server_path}/HYK/DATASET'
snapshot_path = f'{args.server_path}/HYK/DMDS_save/msd_only'
# snapshot_exp = f'{snapshot_path}/{snapshot_prefix}-{args.del_rate}-{args.del_mode}'
if args.del_flag:
    snapshot_exp = f'{snapshot_path}/exp{args.exp}/{args.del_rate}-{args.del_mode}-{args.load_iter}'
else:
    snapshot_exp = f'{snapshot_path}/exp{args.exp}'
args.dm_path = f'{snapshot_exp}/data_maps'

# Create file saveing path for models, data map and dice result.
os.makedirs(snapshot_exp, exist_ok=True)
os.makedirs(args.dm_path, exist_ok=True)
os.makedirs(f'{snapshot_exp}/dice_files', exist_ok=True)
os.makedirs(f'{snapshot_exp}/results', exist_ok=True)

# set up transforms
train_transforms = [
    RandomCrop(args.patch_size, args.patch_nums, pad=args.pad, is_binary=True),
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
    root_dir, list_path, test_root, test_list = get_paths(args.dataset)
    test_root, test_list = get_test_paths(args.dataset)
    # del_list=np.load("/data1/HYK/DMDS_save/msd_only/exp2/0.4-random-2000.npy", allow_pickle=True)
    dataset = NpzDataset(
        list_file=list_path,
        root_dir=root_dir,
        transform=transforms.Compose(train_transforms),
    )
    dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True)

    dice_file = {}
    name_list = dataset.image_list.copy()
    name_list.sort()
    for i, name in enumerate(name_list):
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


    # t = "Task001_MSDPancreas"
    # p_path = os.path.join(preprocessing_output_dir, t, "nnUNetData_plans_v2.1_stage0")
    # n_path = os.path.join(preprocessing_output_dir, t, "stage0/train")
    # dataset = load_dataset(p_path, n_path, 1000, del_list=None)
    # dl = DataLoader3D(dataset, args.patch_size, args.patch_size, args.batch_size,
    #                   oversample_foreground_percent=args.oversample_foreground_percent,
    #                   shuffle=True, transforms=transforms.Compose(train_transforms))

    # dice_file = {}
    # for i, name in enumerate(list(dataset.keys())):
    #     dice_file.update({
    #         f'{name}': {  # 每个样本的名字，我们根据这个来挑选数据
    #             'idx': i,  # 每个样本对应的id号
    #             'dice': np.zeros(0, dtype='float32'),  # 每次迭代计算得到的分数
    #             'corr': np.zeros(0, dtype='float32'),  # 同上
    #             'conf': 0,  # 置信度
    #             'var': 0,  # 方差
    #             'mean_corr': 0,
    #             'nums': 0,  # 每个样本训练的次数
    #             'dataset': name.split('_')[0]}  #
    #         }
    #     )

    # set record files
    writer = SummaryWriter(f'{snapshot_path}/runs/exp{args.exp}-{args.del_rate}-{args.del_mode}', comment='test')

    batch_dice = torch.zeros(args.batch_size)
    batch_corr = torch.zeros(args.batch_size)

    all_cases = os.listdir(test_root)
    list.sort(all_cases)
    images, labels = [], []
    eval_cases = 6
    for i in range(eval_cases):
        image, label = load_data_test(f'{test_root}/{all_cases[i]}', dataset=args.dataset, convert=True, npz=True)
        images.append(image)
        labels.append(label)
    all_cases = os.listdir(root_dir)
    list.sort(all_cases)
    for i in range(eval_cases):
        image, label = load_data_test(f'{root_dir}/{all_cases[i]}', dataset=args.dataset, convert=True, npz=True)
        images.append(image)
        labels.append(label)

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
        train_transforms=train_transforms
    )

