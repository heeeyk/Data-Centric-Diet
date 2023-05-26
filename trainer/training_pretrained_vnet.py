import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import tqdm
from torch.autograd import Variable
from trainer.metrics import cross_entropy_3d
from trainer.utils import dice, dice_torch, DataMap, recall, load_data_test
import time
from trainer import testbox
from datasets.hdf5 import NpzDataset
from datasets.paths import get_paths
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets.dataset_loading import load_dataset, DataLoader3D
from nnunet.paths import preprocessing_output_dir
from models.vnet import VNet


def trainbox(net,
             dataloader,
             dice_file,
             writer,
             iter_num,
             snapshot_exp,
             args,
             eval_images,
             eval_labels,
             eval_cases=6,
             dataset=None,
             metrics='DataMap'):
    # set optimizer
    lr = args.lr * math.pow(1.0 - (iter_num / args.max_iteration), args.power)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-6)

    # train
    for epoch in tqdm.tqdm(range(1, args.max_epoch)):
        # load data here
        loss_total = 0
        for i_batch, sampled_batch in enumerate(dataloader):
            if sampled_batch:

                volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                case_batch = sampled_batch['keys']

                # volume_batch = volume_batch.reshape((-1,) + volume_batch.shape[2:]).unsqueeze(1)  # dl
                volume_batch = volume_batch.reshape((-1,) + volume_batch.shape[2:])
                label_batch = label_batch.reshape((-1,) + label_batch.shape[2:])
                volume_batch, label_batch = Variable(volume_batch.cuda()), Variable(label_batch.cuda())

                output = net(volume_batch)
                loss = cross_entropy_3d(output, label_batch)
                loss_total += loss.detach().cpu().data.numpy()

                pred = output.argmax(dim=1)
                label_batch = label_batch.squeeze(1)

                batch_dice = torch.zeros(args.batch_size)
                batch_corr = torch.zeros(args.batch_size)
                for i in range(args.batch_size):
                    batch_dice[i] = dice_torch(pred[i*args.patch_nums:(i+1)*args.patch_nums] == 1, label_batch[i*args.patch_nums:(i+1)*args.patch_nums] == 1)
                    batch_corr[i] = recall(pred[i*args.patch_nums:(i+1)*args.patch_nums], label_batch[i*args.patch_nums:(i+1)*args.patch_nums])

                if dice_file is not None:
                    for i, case_name in enumerate(case_batch):
                        dice_file[case_name]['dice'] = np.append(dice_file[case_name]['dice'], np.asarray(batch_dice[i], dtype='float32'))
                        dice_file[case_name]['corr'] = np.append(dice_file[case_name]['corr'], np.asarray(batch_corr[i], dtype='float32'))
                        dice_file[case_name]['nums'] += 1

                writer.add_scalar(
                    'train_loss',
                    loss_total,
                    global_step=iter_num,
                    walltime=time.time(),
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                iter_num = iter_num + 1

            if iter_num >= 20000 and iter_num % 1000 == 0:
                if not os.path.exists(f'{snapshot_exp}/models'):
                    os.makedirs(f'{snapshot_exp}/models')
                torch.save(net, f'{snapshot_exp}/models/iter_{iter_num}')

            if iter_num >= 10000 and iter_num % 1000 == 0:
                test = testbox.TestBox(32, 64, 20, 2, net.eval(), args)
                eval_d = []
                for i in range(eval_cases):
                    _, d = test.test0(eval_images[i], eval_labels[i], p=0.8)
                    eval_d.append(d)
                print('\teval dsc, mean: {:.4f}, max: {:.4f}, min: {:.4f}'.format(np.mean(eval_d), np.max(eval_d), np.min(eval_d)))
                net.train()

            if iter_num >= args.max_iteration:
                break

        np.save(snapshot_exp + f'/dice_files/000patch_dice.npy', dice_file)

        if epoch % 30 == 0:
            if not os.path.exists(f'{snapshot_exp}/models'):
                os.makedirs(f'{snapshot_exp}/models')
            torch.save(net, f'{snapshot_exp}/models/epoch_{epoch}')
        if epoch % 10 == 0:
            if metrics == 'DataMap':
                datamap = DataMap(args, save_path=args.dm_path, dice_file=dice_file)
                datamap.plot()
                datamap.save(epoch, iter=iter_num)
                np.save(snapshot_exp + f'/dice_files/i{iter_num}_e{epoch}_patch_dice.npy', dice_file)

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

        lr = args.lr * math.pow(1.0 - (iter_num / args.max_iteration), args.power)
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-6)

        if iter_num >= args.max_iteration:
            writer.close()
            break
        print('epoch %d : loss : %f' % (epoch, loss_total))
