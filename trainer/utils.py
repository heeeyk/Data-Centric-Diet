import h5py
import torch
import tqdm
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np
import copy
import re
import math
import matplotlib.pyplot as plt
import os
import seaborn as sns
from datasets.utils import label_convert

import asyncio
from aiohttp import TCPConnector, ClientSession

import pyecharts.options as opts
from pyecharts.charts import Scatter3D



def load_data_test(imglabelpath, filename, length_ratio=-1, convert=True, min_length=-1, npz=True):
    data = np.load(imglabelpath)
    if npz:
        image = data['data'][0]
        label = data['data'][1]
    else:
        image = data[0]
        label = data[1]

    # change from dataset to array
    image = np.array(image).astype(np.float32)
    label = np.array(label).astype(dtype=np.uint8)

    if length_ratio > 0:
        expected_length = int(length_ratio * image.shape[0])
        if expected_length < min_length:
            expected_length = min_length
        expected_size = (expected_length, ) + image.shape[1:]
        image = np.reshape(image, (1, 1,) + image.shape).astype(np.float32)
        image = F.interpolate(torch.from_numpy(image), size=expected_size, mode='trilinear', align_corners=False).numpy()[0, 0, :, :, :]

    if convert:
        label = label_convert(filename, label)

    return image, label


def construct_PE(image, weight):
    # image is (H, W, L) shape torch tensor
    # return (1, 1, H, W, L)-shaped image if weight <= 0
    assert image.dim() == 3
    if weight <= 0:
        return image.unsqueeze(0).unsqueeze(0)

    device = image.device
    I1 = np.arange(image.shape[-3]).astype(np.float) / (image.shape[-3] - 1)
    I1 = I1[:, np.newaxis, np.newaxis]
    I1 = np.tile(I1, (1, image.shape[-2], image.shape[-1]))
    I2 = np.arange(image.shape[-2]).astype(np.float) / (image.shape[-2] - 1)
    I2 = I2[np.newaxis, :, np.newaxis]
    I2 = np.tile(I2, (image.shape[-3], 1, image.shape[-1]))
    I3 = np.arange(image.shape[-1]).astype(np.float) / (image.shape[-1] - 1)
    I3 = I3[np.newaxis, np.newaxis, :]
    I3 = np.tile(I3, (image.shape[-3], image.shape[-2], 1))

    position_encoding = np.stack([I1, I2, I3]) * weight         # 4, H, W, L
    position_encoding = torch.from_numpy(position_encoding).unsqueeze(0).to(device)

    return position_encoding


def color_map(N=256, normalized=False):
  def bitget(byteval, idx):
    return ((byteval & (1 << idx)) != 0)

  dtype = 'float32' if normalized else 'uint8'
  cmap = np.zeros((N, 3), dtype=dtype)
  for i in range(N):
    r = g = b = 0
    c = i
    for j in range(8):
      r = r | (bitget(c, 0) << 7 - j)
      g = g | (bitget(c, 1) << 7 - j)
      b = b | (bitget(c, 2) << 7 - j)
      c = c >> 3

    cmap[i] = np.array([r, g, b])

  cmap = cmap / 255 if normalized else cmap
  return cmap


def adjust_opt(optAlg, optimizer, init_lr, iter_num, max_iterations, power=0.9):
    if optAlg == 'sgd':
      lr = init_lr * math.pow(1.0 - (iter_num / max_iterations), power)

      for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def visualize(im, vote_map, label, n_class=9, ratio=1.0):
    im -= im.min()
    im = (im / im.max() * 255).astype(np.uint8)
    cmap = color_map()
    im = im[..., np.newaxis]
    im = im.repeat(3, axis=-1)
    pre_vis = copy.deepcopy(im)

    for c_idx in range(1, n_class):
        im[..., 0][label == c_idx] = cmap[c_idx, 0] * ratio + im[..., 0][label == c_idx] * (1. - ratio)
        im[..., 1][label == c_idx] = cmap[c_idx, 1] * ratio + im[..., 1][label == c_idx] * (1. - ratio)
        im[..., 2][label == c_idx] = cmap[c_idx, 2] * ratio + im[..., 2][label == c_idx] * (1. - ratio)

        pre_vis[..., 0][vote_map == c_idx] = cmap[c_idx, 0] * ratio + pre_vis[..., 0][vote_map == c_idx] * (1. - ratio)
        pre_vis[..., 1][vote_map == c_idx] = cmap[c_idx, 1] * ratio + pre_vis[..., 1][vote_map == c_idx] * (1. - ratio)
        pre_vis[..., 2][vote_map == c_idx] = cmap[c_idx, 2] * ratio + pre_vis[..., 2][vote_map == c_idx] * (1. - ratio)

    vis = np.concatenate((im, pre_vis), axis=2)
    return vis


def vis_one_image(im, label, n_class=9, ratio=0.8):
    im = im.astype(np.float32)
    im -= im.min()
    im = (im / im.max() * 255).astype(np.uint8)
    cmap = color_map()
    im = im[..., np.newaxis]
    im = im.repeat(3, axis=-1)

    for c_idx in range(1, n_class):
        color_idx = c_idx
        if c_idx == 8:
            color_idx = 11
        if c_idx == 6:
            cmap[c_idx, 1] = 255
            cmap[c_idx, 2] = 255
        im[..., 0][label == c_idx] = cmap[color_idx, 0] * ratio + im[..., 0][label == c_idx] * (1. - ratio)
        im[..., 1][label == c_idx] = cmap[color_idx, 1] * ratio + im[..., 1][label == c_idx] * (1. - ratio)
        im[..., 2][label == c_idx] = cmap[color_idx, 2] * ratio + im[..., 2][label == c_idx] * (1. - ratio)

    return im


def vis_sr_images(im, output_map):
  assert im.min() >= 0 and output_map.min() >= 0
  assert im.max() <= 1 and output_map.max() <= 1

  im = (im * 255.).astype(np.uint8)[..., np.newaxis]
  output_map = (output_map * 255.).astype(np.uint8)[..., np.newaxis]
  im = im.repeat(3, axis=-1)
  output_map = output_map.repeat(3, axis=-1)
  vis = np.concatenate((im, output_map), axis=2)
  return vis


def cal_histogram(im, label, cls):
  assert im.min() >= 0 and im.max() <= 1
  im = (im * 255.).astype(np.uint8)

  im = im[label == cls].ravel()
  hist = np.bincount(im, minlength=256)
  return hist


def load_state_dict(net, state_dict, remove='', add=''):
  own_state = net.state_dict()
  for param in own_state.items():
    name = add + param[0].replace(remove, '')
    if name not in state_dict:
      print('{} not in pretrained model'.format(param[0]))
  for name, param in state_dict.items():
    if remove + name.replace(add, '') not in own_state:
      print('skipping {}'.format(name))
      continue
    if isinstance(param, Parameter):
      # backwards compatibility for serialized parameters
      param = param.data
    if param.shape == own_state[remove + name.replace(add, '')].shape:
      own_state[remove + name.replace(add, '')].copy_(param)
    else:
      print('skipping {} because of shape inconsistency'.format(name))


def dice(x, y, eps=1e-7):
  intersect = np.sum(np.sum(np.sum(x * y)))
  y_sum = np.sum(np.sum(np.sum(y)))
  x_sum = np.sum(np.sum(np.sum(x)))
  return 2 * intersect / (x_sum + y_sum + eps)


def dice_torch(x, y, eps=1e-7):
  intersect = torch.sum(torch.sum(torch.sum(x * y)))
  y_sum = torch.sum(torch.sum(torch.sum(y)))
  x_sum = torch.sum(torch.sum(torch.sum(x)))
  return 2 * intersect / (x_sum + y_sum + eps)


def recall(predict, target):  # Sensitivity, Recall, true positive rate
    if torch.is_tensor(predict):
        predict = predict.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    predict = np.atleast_1d(predict.astype(np.bool))
    target = np.atleast_1d(target.astype(np.bool))

    tp = np.count_nonzero(predict & target)
    fn = np.count_nonzero(~predict & target)

    try:
        recall = tp / float(tp + fn)
    except ZeroDivisionError:
        recall = 0.0

    return recall


def false_rate(predict, target):  # Sensitivity, Recall, true positive rate
    if torch.is_tensor(predict):
        predict = predict.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    predict = np.atleast_1d(predict.astype(np.bool))
    target = np.atleast_1d(target.astype(np.bool))

    # tp = np.count_nonzero(predict & target)
    fn = np.count_nonzero(~predict & target)
    fp = np.count_nonzero(predict & ~target)

    try:
        false_rate = float(fp + fn)
    except ZeroDivisionError:
        false_rate = 0.0
    return false_rate



def binary_dice_loss(input, target):
    smooth = 1.

    # apply softmax to input
    input = F.softmax(input, dim=1)
    input = input[:, 1]

    iflat = input.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    loss = 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))
    return loss


def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )
    return l


class DataMap:
    def __init__(self, args, save_path, dice_file=None, load_path=None):

        if load_path is not None:
            dice_file = np.load(load_path, allow_pickle=True).item()
        elif not dice_file:
            raise Exception("please input a dice file!")

        self.dataset = args.dataset
        self.save_path = save_path
        self.dice_file = dice_file
        self.len = len(dice_file)
        self.args = args

    def statistics(self, epoch_slice=False, es=0, ee=0):

        sample_nums = len(self.dice_file)
        total_nums = 0
        confidence, variability, correctness = np.zeros(sample_nums), np.zeros(sample_nums), np.zeros(sample_nums)

        if epoch_slice:
            if ee == -1:
                ee = self.dice_file[0]['nums']
            for i, n in enumerate(self.dice_file):
                confidence[i] = sum(self.dice_file[n]['dice'][es:ee])/(ee-es)
                variability[i] = np.sqrt(
                    sum(np.power(self.dice_file[n]['dice'][es:ee] - confidence[i], 2)) / (ee-es))
                correctness[i] = sum(self.dice_file[n]['corr'])/self.dice_file[n]['nums']
        else:
            for i, n in enumerate(self.dice_file):

                # 全为0的不计入data map
                if not self.dice_file[n]['dice'].any():
                    dice = [0.001, 0.001]
                    nums = 2
                else:
                    dice = self.dice_file[n]['dice'][self.dice_file[n]['dice'] != 0]
                    nums = self.dice_file[n]['dice'][self.dice_file[n]['dice'] != 0].shape[0]

                confidence[i] = sum(dice)/nums
                variability[i] = np.sqrt(sum(np.power(dice - confidence[i], 2)) / nums)
                correctness[i] = sum(self.dice_file[n]['corr'])/nums

                self.dice_file[n]['conf'] = confidence[i]
                self.dice_file[n]['var'] = variability[i]
                self.dice_file[n]['mean_corr'] = correctness[i]

        return confidence, variability, correctness

    def score_compete(self):
        GraNd, EL2N, EL2N_ = np.zeros(self.len), np.zeros(self.len), np.zeros(self.len)
        for i, n in enumerate(self.dice_file):
            # GraNd[i] = sum(self.dice_file[n]['grad'])/self.dice_file[n]['grad'].shape[0]
            # GraNd[i] = sum(self.dice_file[n]['grad'])/self.dice_file[n]['grad'].shape[0]
            GraNd[i] = 0
            EL2N[i] = sum(self.dice_file[n]['EL2N'])/self.dice_file[n]['EL2N'].shape[0]
            EL2N_[i] = sum(self.dice_file[n]['EL2N_']) / self.dice_file[n]['EL2N_'].shape[0]
        return GraNd, EL2N, EL2N_

    def ranking(self, epoch_slice=False, es=0, ee=0):
        score, _, _ = self.statistics(epoch_slice, es, ee)
        rank = np.argsort(score)
        ranked_example = []
        for i in rank:
            ranked_example.append(list(self.dice_file.keys())[i])
        return ranked_example

    def data_select(self, q, mode='hard', score_mode='datamap', epoch_slice=False, es=0, ee=0):
        """
         q ~ [0:1], 数字越大 删除比例越高
         mode = 'hard' or 'easy' or 'random' or 'keep ambiguous' ,删减模式
         score = 'datamap' or 'GraNd' or 'EL2N'
        """
        print(f'score_mode = {score_mode}')
        wait_to_del = []
        if mode == 'hard' or mode == 'random':
            q = int(q * self.len)
        elif mode == 'easy':
            q = int((1-q) * self.len)
        elif mode == 'keep ambiguous':
            q1 = int(0.5*q * self.len)
            q2 = int((1 - 0.5*q) * self.len)

        if score_mode == 'datamap':
            # conf, var, cor = self.statistics()
            score, _, _ = self.statistics(epoch_slice, es, ee)

        else:
            score_3 = self.score_compete()
            q = self.len - q
            if score_mode == 'GraNd':
                score = score_3[0]
            if score_mode == 'EL2N':
                score = score_3[1]
            if score_mode == 'EL2Nx':
                score = score_3[2]
        x = score.copy()
        x.sort()

        if mode == 'random':
            wait_to_del = np.random.choice(list(self.dice_file.keys()), size=q, replace=False)
        else:
            for i, n in enumerate(self.dice_file):
                if score_mode == 'datamap':
                    if mode == 'hard' and score[i] < x[q]:
                        wait_to_del.append(n)
                    if mode == 'easy' and score[i] > x[q]:
                        wait_to_del.append(n)
                    if mode == 'keep ambiguous':
                        if score[i] < x[q1] or score[i] > x[q2]:
                            wait_to_del.append(n)
                else:
                    if mode == 'hard' and score[i] > x[q]:
                        wait_to_del.append(n)
                    if mode == 'easy' and score[i] < x[q]:
                        wait_to_del.append(n)
                    if mode == 'keep ambiguous':
                        # q1 = self.len - q1
                        # q2 = self.len - q2
                        if score[i] < x[q1] or score[i] > x[q2]:
                            wait_to_del.append(n)

        return wait_to_del

    def list_nums(self, del_list):
        """
         返回各数据集删除样本的数量
        """
        msd, nih, synapse, word = 0, 0, 0, 0
        for i in range(len(del_list)):
            if 'pancreas' in  del_list[i]:
                msd += 1
            elif 'word' in del_list[i]:
                word += 1
            elif 'img' in del_list[i]:
                synapse += 1
            elif 'PANCREAS' in del_list[i]:
                nih += 1
        if msd+nih+synapse+word != len(del_list):
            print('something unknown datasets...')
        print(f'{msd} cases msd has been deleted \r\n',
              f'{nih} cases nih has been deleted \r\n',
              f'{synapse} cases synapse has been deleted \r\n',
              f'{word} cases word has been deleted \r\n')


    def map_change(self, epoch_nums):
        sample_nums = len(self.dice_file)
        confidence, variability = np.zeros((epoch_nums, sample_nums)), np.zeros((epoch_nums, sample_nums))
        for e in tqdm.tqdm(range(50, epoch_nums+1)):
            for i in range(sample_nums):
                confidence[e-1, i] = sum(self.dice_file[i]['dice'][e-50:e])/(50)
                variability[e-1, i] = np.sqrt(
                    sum(np.power(self.dice_file[i]['dice'][e-50:e] - confidence[e-1, i], 2)) / (50))
        return confidence, variability


    def plot(self, epoch_slice=False, ee=0, es=0):
        """
         return an image for data map, enter plt.show() to
         show in pycharm, or enter plt.save() to save.
        """
        confidence, variability, correctness = self.statistics(epoch_slice, ee, es)

        # Colors
        cmap = sns.color_palette("coolwarm_r", as_cmap=True)

        # Start Plotting
        fig, (ax1) = plt.subplots(1, 1, figsize=(6.5, 6))
        from matplotlib.patches import Rectangle
        rect1 = Rectangle((0.015, 0.7), 0.06, 0.18, linewidth=2, edgecolor='lightsteelblue', facecolor='none')
        rect2 = Rectangle((0.056, 0.3), 0.1, 0.18, linewidth=2, edgecolor='indianred', facecolor='none')

        # ax1.add_patch(rect1)
        # ax1.add_patch(rect2)
        # ax1.text(0.015, 0.9, 'Easy-to-learn', fontsize=18, color='cornflowerblue')
        # ax1.text(0.056, 0.25, 'Hard-to-learn', fontsize=18, color='indianred')
        p = ax1.scatter(variability, confidence, c=confidence, cmap=cmap, alpha=0.77)

        # Labeling
        ax1.set_xlabel("Variability", fontname="Raleway", fontsize=21)
        ax1.set_ylabel("DAD Score", fontname="Raleway", fontsize=21)

        # Axis Handling
        ax1.set_xlabel("Variability", fontname="Raleway", fontsize=21)
        ax1.set_ylabel("DAD Score", fontname="Raleway", fontsize=21)

        # Styling
        # ax1.spines["right"].set_visible(False)
        # ax1.spines["top"].set_visible(False)
        ax1.xaxis.set_ticks_position("bottom")
        ax1.xaxis.set_ticks(np.arange(0.1, 0.3, 0.05))
        ax1.yaxis.set_ticks_position("left")
        ax1.yaxis.set_ticks(np.arange(0.3, 1.01, 0.1))
        ax1.tick_params(axis='both', which='major', labelsize=16)

        # Colorbar
        # cbar = fig.colorbar(p, ax=ax1, fraction=0.043, pad=0.00, shrink=0.9, aspect=50, orientation="vertical")
        # cbar.solids.set(alpha=1)
        # cbar.ax.set_title("Easy-to-learn", fontsize=21, pad=7)

    def datamap_dynamic(self, epoch_slice=False, ee=0, es=0, ax1=None):
        """
             return an image for data map, enter plt.show() to
             show in pycharm, or enter plt.save() to save.
        """
        confidence, variability, correctness = self.statistics(epoch_slice, ee, es)

        # Colors
        cmap = sns.color_palette("coolwarm_r", as_cmap=True)

        # Start Plotting

        from matplotlib.patches import Rectangle
        p = ax1.scatter(variability, confidence, c=confidence, cmap=cmap, alpha=0.77)

        # Labeling
        # ax1.set_xlabel("Variability", fontname="Raleway", fontsize=21)


        # Styling
        # ax1.spines["right"].set_visible(False)
        # ax1.spines["top"].set_visible(False)
        ax1.xaxis.set_ticks_position("bottom")
        ax1.xaxis.set_ticks(np.arange(0, 0.33, 0.08))
        ax1.yaxis.set_ticks_position("left")
        ax1.yaxis.set_ticks(np.arange(0.1, 1.01, 0.2))
        ax1.tick_params(axis='both', which='major', labelsize=16)
        ax1.spines['right'].set_linewidth(1.5)
        ax1.spines['left'].set_linewidth(1.5)
        ax1.spines['bottom'].set_linewidth(1.5)
        ax1.spines['top'].set_linewidth(1.5)


    def plot_2(self, area1, area2, area3, area4, ax1=None):
        """
        datamap的动态变化
         return an image for data map, enter plt.show() to
         show in pycharm, or enter plt.save() to save.
        """
        confidence, variability, correctness = self.statistics()

        # Start Plotting
        c = list(range(self.len))
        for i, name in enumerate(list(self.dice_file.keys())):
            if name in area1:
                c[i] = 'red'
            elif name in area2:
                c[i] = 'grey'
            elif name in area3:
                c[i] = 'green'
            elif name in area4:
                c[i] = 'blue'
            else:
                c[i] = 'gray'

        p = ax1.scatter(variability, confidence, c=c, alpha=0.77)

        # Labeling
        ax1.set_xlabel("Variability", fontname="Raleway", fontsize=21)
        ax1.set_ylabel("Confidence", fontname="Raleway", fontsize=21)

        # Axis Handling
        ax1.set_xlabel("Variability", fontname="Raleway", fontsize=21)
        ax1.set_ylabel("Confidence", fontname="Raleway", fontsize=21)

        # Styling
        ax1.xaxis.set_ticks_position("bottom")
        ax1.xaxis.set_ticks(np.arange(0, 0.33, 0.08))
        ax1.yaxis.set_ticks_position("left")
        ax1.yaxis.set_ticks(np.arange(0.1, 1.01, 0.2))
        ax1.tick_params(axis='both', which='major', labelsize=16)
        ax1.spines['right'].set_linewidth(1.5)
        ax1.spines['left'].set_linewidth(1.5)
        ax1.spines['bottom'].set_linewidth(1.5)
        ax1.spines['top'].set_linewidth(1.5)
        # Colorbar
        # cbar = fig.colorbar(p, ax=ax1, fraction=0.043, pad=0.00, shrink=0.9, aspect=50, orientation="vertical")
        # cbar.solids.set(alpha=1)
        # cbar.ax.set_title("Correctness", fontsize=21, pad=7)

    def plot_3(self, area1, ax1=None):
        """
        e10 的 hard example 到 e20 有多少保持一致 ?
         return an image for data map, enter plt.show() to
         show in pycharm, or enter plt.save() to save.
        """
        confidence, variability, correctness = self.statistics()

        # Start Plotting
        c = list(range(self.len))
        overlap = 0
        for i, name in enumerate(list(self.dice_file.keys())):
            if name in area1:
                c[i] = 'red'
            # elif name in area2:
            #     c[i] = 'royalblue'
            else:
                c[i] = 'gray'

            # if name in area1 and name in area2:
            #     c[i] = 'seagreen'
            #     overlap += 1
        if len(area1) != 0:
            overlap_ratio = overlap/len(area1)
        else:
            overlap_ratio = 0

        p = ax1.scatter(variability, confidence, c=c, alpha=0.77)

        # # Labeling
        # ax1.set_xlabel("Variability", fontname="Raleway", fontsize=21)
        # ax1.set_ylabel("Confidence", fontname="Raleway", fontsize=21)
        #
        # # Axis Handling
        # ax1.set_xlabel("Variability", fontname="Raleway", fontsize=21)
        # ax1.set_ylabel("Confidence", fontname="Raleway", fontsize=21)

        # Styling
        # ax1.spines["right"].set_visible(False)
        # ax1.spines["top"].set_visible(False)
        ax1.xaxis.set_ticks_position("bottom")
        ax1.xaxis.set_ticks(np.arange(0, 0.33, 0.08))
        ax1.yaxis.set_ticks_position("left")
        ax1.yaxis.set_ticks(np.arange(0.1, 1.01, 0.2))
        ax1.tick_params(axis='both', which='major', labelsize=16)
        ax1.spines['right'].set_linewidth(1.5)
        ax1.spines['left'].set_linewidth(1.5)
        ax1.spines['bottom'].set_linewidth(1.5)
        ax1.spines['top'].set_linewidth(1.5)

        return overlap_ratio

    def plot_4(self, point_list, epoch_slice=False, ee=0, es=0, fl=None):
        """
        e10 的 hard example 到 e20 有多少保持一致 ?
         return an image for data map, enter plt.show() to
         show in pycharm, or enter plt.save() to save.
        """
        confidence, variability, correctness = self.statistics(epoch_slice, ee, es)

        # Start Plotting
        fig, (ax1) = plt.subplots(1, 1, figsize=(4, 8))
        c = list(range(self.len))
        for i in range(len(c)):
            c[i] = 'gray'
        j = 0
        for i, name in enumerate(list(self.dice_file.keys())):
            for noise_label in point_list:
                if noise_label in name:
                    c[i] = 'red'
        if fl:
            x = list(range(len(fl)))
            for i in range(len(fl)):
                x[i] = np.mean(fl[i][ee:es])
            p = ax1.scatter(x, confidence, c=c, alpha=0.77)
        else:
            p = ax1.scatter(variability, confidence, c=c, alpha=0.77)

        # Labeling
        ax1.set_xlabel("Variability", fontname="Raleway", fontsize=21)
        ax1.set_ylabel("Confidence", fontname="Raleway", fontsize=21)

        # Axis Handling
        ax1.set_xlabel("Variability", fontname="Raleway", fontsize=21)
        ax1.set_ylabel("Confidence", fontname="Raleway", fontsize=21)

        # Styling
        # ax1.spines["right"].set_visible(False)
        # ax1.spines["top"].set_visible(False)

        # ax1.xaxis.set_ticks_position("bottom")
        # ax1.xaxis.set_ticks(np.arange(0, 0.51, 0.05))
        # ax1.yaxis.set_ticks_position("left")
        # ax1.yaxis.set_ticks(np.arange(0, 0.04, 0.1))


    def save(self, epoch, iter):
        plt.savefig(f'{self.save_path}/{self.dataset}_e{epoch}_i{iter}.pdf', bbox_inches="tight")
        plt.close()

    def map_3dplot(self, epoch_slice=0, path_list=None):
        if not path_list:
            epoch_nums = self.dice_file[0]['nums']
            confidence, variability, correctness = \
                list(range(int(epoch_nums/epoch_slice)+1)),\
                list(range(int(epoch_nums/epoch_slice)+1)),\
                list(range(int(epoch_nums/epoch_slice)+1))

            for i in range(0, epoch_nums, epoch_slice):
                e = int(i/epoch_slice)
                confidence[e], variability[e], correctness[e] = self.statistics(True, es=i, ee=i+epoch_slice)
        else:
            epoch_nums = len(path_list)
            confidence, variability, correctness = \
                list(range(epoch_nums)),\
                list(range(epoch_nums)),\
                list(range(epoch_nums))

            for e in range(epoch_nums):
                self.dice_file = np.load(path_list[e], allow_pickle=True)
                confidence[e], variability[e], correctness[e] = self.statistics()

        # 构造数据
        data = [
            [
                epoch*5,
                variability[epoch][index],
                confidence[epoch][index],
                correctness[epoch][index],
                0.2,
                index
            ]
            for index in range(len(confidence[0]))
            for epoch in range(len(confidence))
        ]

        # 配置 config
        config_xAxis3D = "epoch"
        config_yAxis3D = "variability"
        config_zAxis3D = "confidence"
        config_color = "fiber"
        config_symbolSize = "vitaminc"

        Scatter3D(init_opts=opts.InitOpts(width="1440px", height="720px")).add(
            series_name="",
            data=data,
            xaxis3d_opts=opts.Axis3DOpts(
                name=config_xAxis3D,
                type_="value",
                # textstyle_opts=opts.TextStyleOpts(color="#fff"),
            ),
            yaxis3d_opts=opts.Axis3DOpts(
                name=config_yAxis3D,
                type_="value",
                # textstyle_opts=opts.TextStyleOpts(color="#fff"),
            ),
            zaxis3d_opts=opts.Axis3DOpts(
                name=config_zAxis3D,
                type_="value",
                # textstyle_opts=opts.TextStyleOpts(color="#fff"),
            ),
            grid3d_opts=opts.Grid3DOpts(width=100, height=100, depth=100),
        ).set_global_opts(
            visualmap_opts=[
                opts.VisualMapOpts(
                    type_="color",
                    is_calculable=True,
                    dimension=3,
                    pos_top="1",
                    max_=0.8,
                    range_color=[
                        "#1710c0",
                        "#0b9df0",
                        "#00fea8",
                        "#f09a09",
                        "#fe0300",
                    ],
                ),
                opts.VisualMapOpts(
                    type_="size",
                    is_calculable=True,
                    dimension=4,
                    pos_bottom="10",
                    max_=0.2,
                    range_size=[1, 2],
                ),
            ]
        ).render("scatter3d.html")


def load_dice_file(root, epoch):
    dice_path = os.listdir(root)
    dice_path.sort()
    for i in dice_path:
        if f'_e{epoch}_' in i:
            x = np.load(f'{root}/{i}', allow_pickle=True).item()
            return x


def vog_compute(k, q, vog_file):
    """
    :param k: VOG checkpoint
    :param q: data delete ratio,(0~1)
    :param path: vog file path
    :return:
    """
    grad_shape = 8 * 64 * 64 * 64
    VOG, wait_to_del = [], []
    for i, n in tqdm.tqdm(enumerate(vog_file)):
        grad = vog_file[n]['grad']
        mean_grad = 0
        VOGp = 0
        for t in range(k):
            mean_grad += grad[grad_shape*t: grad_shape*(t+1)]
        mean_grad /= k
        for t in range(k):
            VOGp += (grad[grad_shape*t: grad_shape*(t+1)] - mean_grad)**2
        VOGp /= np.sqrt(k)
        VOG.append(np.sum(VOGp)/len(VOGp))
    VOG_sort = VOG.copy()
    VOG_sort.sort()
    q1 = int(0.5 * q * len(vog_file))
    q2 = int((1 - 0.5 * q) * len(vog_file))
    for i, n in enumerate(vog_file):
        if VOG[i] < VOG_sort[q1] or VOG[i] > VOG_sort[q2]:
            wait_to_del.append(n)
    return wait_to_del







if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Training.')
    parser.add_argument('--dataset', type=str, default='msd_pancreas', help='The dataset to be trained')
    parser.add_argument('--dm_path', type=str, default=None, help='save path for data map')
    args = parser.parse_args()

    # args.dm_path = '1123asd'
    load_path = '/data/HYK/DMDS_save/msd_nih/exp2-GraNd/No_Prunning/dice_files/i1836_e50_patch_dice.npy'
    d = DataMap(args, save_path=args.dm_path, load_path=load_path)
    # d.plot(epoch_slice=True, ee=10, es=200)
    d.plot()
    plt.show()
    # plt.savefig('/data/HYK/DMDS_save/figure/datamap.png')
