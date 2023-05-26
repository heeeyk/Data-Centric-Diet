
import math
import sys

import torch
import tqdm


import numpy as np
import cc3d
from torch.autograd import Variable
import torch.nn.functional as F
import cv2
from trainer.utils import dice, visualize, construct_PE, vis_one_image
from scipy.ndimage.filters import gaussian_filter
from sys import getsizeof
import monai

def get_gaussian(s, sigma=1.0/8) -> np.ndarray:
    temp = np.zeros(s)
    coords = [i // 2 for i in s]
    sigmas = [i * sigma for i in s]
    temp[tuple(coords)] = 1
    gaussian_map = gaussian_filter(temp, sigmas, 0, mode='constant', cval=0)
    gaussian_map /= np.max(gaussian_map)
    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_map[gaussian_map == 0] = np.min(gaussian_map[gaussian_map != 0])
    return gaussian_map


class TestBox:
    def __init__(self, stride, patch_size, patch_length=0, n_class=2, net=None, args=None, sr=False):
        self.stride = stride
        self.patch_size = patch_size
        self.patch_length = patch_length
        self.n_class = n_class
        self.net = net.eval()
        self.args = args
        self.stride_length = int(stride/4)
        self.sr = sr

    def init(self, image):
        self.padded = False
        w, h, d = image.shape

        if self.sr:
            sx = math.floor((w - self.patch_length) / self.stride_length) + 1
        else:
            sx = math.floor((w - self.patch_size) / self.stride) + 1
        sy = math.floor((h - self.patch_size) / self.stride) + 1
        sz = math.floor((d - self.patch_size) / self.stride) + 1
        if self.sr and self.args.mode != 'direct':
            vote_map = torch.zeros((self.n_class, int(w * 4), h, d), dtype=torch.float).cuda()
        else:
            vote_map = torch.zeros((self.n_class, w, h, d), dtype=torch.float).cuda()
        vote_map0 = torch.zeros((self.n_class, w, h, d), dtype=torch.float).cuda()
        vote_map[0] += 0.1
        vote_map0[0] += 0.1
        image = torch.from_numpy(image.astype(np.float32)).cuda()

        # get gaussian important map
        gaussian_map = get_gaussian(np.array((self.patch_size, self.patch_size, self.patch_size)))
        gaussian_map = torch.tensor(gaussian_map).cuda()
        normalization = torch.zeros((self.n_class, w, h, d), dtype=torch.float).cuda()

        return image, sx, sy, sz, vote_map, gaussian_map, normalization, vote_map0

    def test0(self, image, label, p=0.3):
        image, sx, sy, sz, vote_map, gaussian_map, normalization, vote_map0 = self.init(image)
        w, h, d = image.shape
        with torch.no_grad():
            for x in range(0, sx):
                xs = self.stride * x
                xs_r = xs + self.patch_size
                for y in range(0, sy):
                    ys = self.stride * y
                    ys_r = ys + self.patch_size
                    for z in range(0, sz):
                        zs = self.stride * z
                        zs_r = zs + self.patch_size

                        test_patch = image[xs:xs_r, ys:ys_r, zs:zs_r]
                        test_patch = construct_PE(test_patch, 0.0)
                        output = self.net(test_patch)

                        if type(output) == tuple:
                            output = output[0]
                        output = output.squeeze(0)

                        # gaussian map
                        vote_map[:, xs:xs_r, ys:ys_r, zs:zs_r] += output * gaussian_map
                        normalization[:, xs:xs_r, ys:ys_r, zs:zs_r] += gaussian_map

                        mask = output.argmax(dim=0).float()
                        for i in range(self.n_class):
                            # voting strategy for the over-lapped region
                            vote_map0[i, xs:xs_r, ys:ys_r, zs:zs_r] +=  \
                                (mask == i).float()

                        # vote strategy
                        mask = output.argmax(dim=0).float()
                        for i in range(self.n_class):
                            # voting strategy for the over-lapped region
                            vote_map[i, xs:xs_r, ys:ys_r, zs:zs_r] +=  \
                                (mask == i).float()

            vote_map[..., :xs_r, :ys_r, :zs_r] /= normalization[..., :xs_r, :ys_r, :zs_r]

            vote_map = vote_map.argmax(dim=0)
            vote_map0 = vote_map0.argmax(dim=0)


            if self.padded:
                vote_map = vote_map[:self.origin_size[0], :self.origin_size[1], :self.origin_size[2]]
                image = image[:self.origin_size[0], :self.origin_size[1], :self.origin_size[2]]

            if vote_map.shape != label.shape:
                expected_size = label.shape
                vote_map = vote_map.reshape((1, 1,) + vote_map.shape)
                vote_map = F.interpolate(vote_map.float(), size=expected_size, mode='nearest')[0, 0, :, :, :].to(
                    torch.uint8)
                vote_map0 = vote_map0.reshape((1, 1,) + vote_map0.shape)
                vote_map0 = F.interpolate(vote_map0.float(), size=expected_size, mode='nearest')[0, 0, :, :, :].to(
                    torch.uint8)

            vote_map = vote_map.cpu().data.numpy()
            vote_map0 = vote_map0.cpu().data.numpy()
            image = image.cpu().data.numpy()
            vote_map = remove_background(image, vote_map)
            vote_map0 = remove_background(image, vote_map0)

            # noise = vote_map - vote_map0
            # noise[noise == -1] = 0
            # max_component = get_max_connected_component(vote_map0)
            # noise = post_processing(noise, p=0.4, mcc=max_component)
            # vote_map -= noise

            vote_map = post_processing(vote_map, p=p)
            result_print = ''
            d = 0
            for c in range(1, self.n_class):
                d = dice(vote_map == c, label == c)
                result_print += ' \tdsc {}: {:.4f};'.format(c, d)
            if self.n_class == 2:
                return vote_map, d
            else:
                return vote_map, result_print

    def test_guassian(self, image, label, p=0.3, is_post_processing=True):
        image, sx, sy, sz, vote_map, gaussian_map, normalization, vote_map0 = self.init(image)
        w, h, d = image.shape
        with torch.no_grad():
            for x in range(0, sx):
                xs = self.stride * x
                xs_r = xs + self.patch_size
                for y in range(0, sy):
                    ys = self.stride * y
                    ys_r = ys + self.patch_size
                    for z in range(0, sz):
                        zs = self.stride * z
                        zs_r = zs + self.patch_size

                        test_patch = image[xs:xs_r, ys:ys_r, zs:zs_r]
                        test_patch = construct_PE(test_patch, 0.0)
                        output = self.net(test_patch)

                        if type(output) == tuple:
                            output = output[0]
                        output = output.squeeze(0)

                        # gaussian map
                        vote_map[:, xs:xs_r, ys:ys_r, zs:zs_r] += output * gaussian_map
                        normalization[:, xs:xs_r, ys:ys_r, zs:zs_r] += gaussian_map

            vote_map[..., :xs_r, :ys_r, :zs_r] /= normalization[..., :xs_r, :ys_r, :zs_r]

            vote_map = vote_map.argmax(dim=0)


            if self.padded:
                vote_map = vote_map[:self.origin_size[0], :self.origin_size[1], :self.origin_size[2]]
                image = image[:self.origin_size[0], :self.origin_size[1], :self.origin_size[2]]

            if vote_map.shape != label.shape:
                expected_size = label.shape
                vote_map = vote_map.reshape((1, 1,) + vote_map.shape)
                vote_map = F.interpolate(vote_map.float(), size=expected_size, mode='nearest')[0, 0, :, :, :].to(
                    torch.uint8)

            vote_map = vote_map.cpu().data.numpy()
            image = image.cpu().data.numpy()

            if is_post_processing:
                vote_map = remove_background(image, vote_map)
                vote_map = post_processing(vote_map, p=p)
            else:
                pass

            result_print = ''
            d = 0
            for c in range(1, self.n_class):
                d = dice(vote_map == c, label == c)
                result_print += ' \tdsc {}: {:.4f};'.format(c, d)
            if self.n_class == 2:
                return vote_map, d
            else:
                return vote_map, result_print



    def test_vote(self, image, label, p=0.3):
        image, sx, sy, sz, vote_map, gaussian_map, normalization, vote_map0 = self.init(image)
        w, h, d = image.shape
        with torch.no_grad():
            for x in range(0, sx):
                xs = self.stride * x
                xs_r = xs + self.patch_size
                for y in range(0, sy):
                    ys = self.stride * y
                    ys_r = ys + self.patch_size
                    for z in range(0, sz):
                        zs = self.stride * z
                        zs_r = zs + self.patch_size

                        test_patch = image[xs:xs_r, ys:ys_r, zs:zs_r]
                        test_patch = construct_PE(test_patch, 0.0)
                        output = self.net(test_patch)

                        if type(output) == tuple:
                            output = output[0]
                        output = output.squeeze(0)

                        # vote strategy
                        mask = output.argmax(dim=0).float()
                        for i in range(self.n_class):
                            # voting strategy for the over-lapped region
                            vote_map[i, xs:xs_r, ys:ys_r, zs:zs_r] +=  \
                                (mask == i).float()

            vote_map = vote_map.argmax(dim=0)


            if self.padded:
                vote_map = vote_map[:self.origin_size[0], :self.origin_size[1], :self.origin_size[2]]
                image = image[:self.origin_size[0], :self.origin_size[1], :self.origin_size[2]]

            if vote_map.shape != label.shape:
                expected_size = label.shape
                vote_map = vote_map.reshape((1, 1,) + vote_map.shape)
                vote_map = F.interpolate(vote_map.float(), size=expected_size, mode='nearest')[0, 0, :, :, :].to(
                    torch.uint8)

            vote_map = vote_map.cpu().data.numpy()
            image = image.cpu().data.numpy()
            vote_map = remove_background(image, vote_map)

            vote_map = post_processing(vote_map, p=p)
            result_print = ''
            d = 0
            for c in range(1, self.n_class):
                d = dice(vote_map == c, label == c)
                result_print += ' \tdsc {}: {:.4f};'.format(c, d)
            if self.n_class == 2:
                return vote_map, d
            else:
                return vote_map, result_print

    def test_monai(self, image, label, p):
        image = torch.tensor(image).cuda()
        image = image.unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            vote_map = monai.inferers.sliding_window_inference(
                inputs=image,
                predictor=self.net,
                roi_size=(64, 64, 64),
                sw_batch_size=16,
                overlap=0.5,
                mode='gaussian',
                progress=True,
            )
        vote_map = vote_map.squeeze(0).argmax(dim=0)
        vote_map = vote_map.cpu().data.numpy()
        image = image.cpu().data.numpy().squeeze(0).squeeze(0)
        print(image.shape)
        vote_map = remove_background(image, vote_map)
        vote_map = post_processing(vote_map, p=p)
        result_print = ''
        d = 0
        for c in range(1, self.n_class):
            d = dice(vote_map == c, label == c)
            result_print += ' \tdsc {}: {:.4f};'.format(c, d)
        if self.n_class == 2:
            return vote_map, d
        else:
            return vote_map, result_print





    # def test1(self, image):
    #     image, sx, sy, sz, vote_map = self.init(image)
    #
    #     for x in tqdm.tqdm(range(0, sx)):
    #         xs = self.stride_length * x
    #         for y in range(0, sy):
    #             ys = self.stride * y
    #             start_x = self.stride * x
    #
    #             test_patch = image[xs:xs + self.patch_length, ys:ys + self.patch_size, :]
    #
    #             a = list(range(sz))
    #             for z in range(0, sz):
    #                 zs = self.stride * z
    #                 a[z] = test_patch[..., zs:zs + self.patch_size]
    #
    #             test_patch = torch.stack(a, 0)
    #             test_patch = test_patch.unsqueeze(1)
    #
    #             output, _ = self.net(test_patch, mode=self.args.mode)
    #             pred = output.argmax(dim=1)
    #
    #             pred_list = torch.chunk(pred, sz, 0)
    #             for z in range(0, sz):
    #                 zs = self.stride * z
    #                 for i in range(self.n_class):
    #                     vote_map[i, start_x:start_x + self.patch_size, ys:ys + self.patch_size, zs:zs + self.patch_size] = \
    #                         vote_map[i, start_x:start_x + self.patch_size, ys:ys + self.patch_size, zs:zs + self.patch_size] \
    #                         + (pred_list[z].float() == i).float()
    #     return vote_map
    #
    # def test2(self, image):
    #     image, sx, sy, sz, vote_map = self.init(image)
    #     for x in tqdm.tqdm(range(0, sx)):
    #         xs = self.stride_length * x
    #         for y in range(0, sy):
    #             ys = self.stride * y
    #             start_x = self.stride * x
    #
    #             test_patch = image[xs:xs + self.patch_length, ys:ys + self.patch_size, :]
    #             patch_list = torch.chunk(test_patch, int(512 / self.patch_size), -1)
    #             test_patch = torch.stack(patch_list, 0)
    #             test_patch = test_patch.unsqueeze(1)
    #
    #             output, _ = self.net(test_patch, mode=self.args.mode)
    #             pred = output.argmax(dim=1)
    #             pred_list = torch.chunk(pred, int(512 / self.patch_size), 0)
    #             pred = torch.cat(pred_list, dim=-1).float()
    #
    #             if self.args.method == 'vote':
    #                 # 投票：得分最大的那一类作为mask的输出
    #                 for i in range(self.n_class):
    #                     vote_map[i, start_x:start_x + self.patch_size, ys:ys + self.patch_size] = \
    #                         vote_map[i, start_x:start_x + self.patch_size, ys:ys + self.patch_size] + (pred == i).float()
    #     return vote_map

    # def adapt(self, test_dataloader, checkpoint, lr):
    #     print('---------- reloading model --------------')
    #     self.net.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-6)
    #
    #     self.net.train()
    #     for a_i in tqdm.tqdm(range(self.args.adapt_iter)):
    #         loss = 0
    #         for i_batch, sampled_batch in enumerate(test_dataloader):
    #             volume_batch, u_label_batch = sampled_batch['image'], sampled_batch['u_label']
    #             volume_batch, u_label_batch = volume_batch.cuda(), u_label_batch.cuda()
    #
    #             _, u_list, perm_list = self.net(volume_batch, u_label_batch, is_adapt=True)
    #
    #             loss = unary_loss(u_list, perm_list)
    #             optimizer.zero_grad()
    #             loss.backward()
    #
    #         if (a_i + 1) % 5 == 0:
    #             print('iteration {} | Loss: {:.4f}'.format(a_i + 1, loss.item()))
    #         optimizer.step()
    #
    #     self.net.eval()

    def votemap_visualize(self, image, label, vote_map, save_path):
        if image.shape != label.shape:
            expected_size = label.shape
            image = image.reshape((1, 1,) + image.shape)
            image = F.interpolate(image.float(), size=expected_size, mode='trilinear', align_corners=False)[0, 0, :, :, :]
        if torch.is_tensor(image):
            image = image.numpy()
        vis = visualize(image, vote_map, label, n_class=2, ratio=1.)
        os.makedirs(save_path, exist_ok=True)
        for z_i, im in enumerate(vis):
            cv2.imwrite(os.path.join(save_path, '{}.jpg'.format(z_i)), im)

def range_judge(label_center, r):
    """
    判断label中心到边界的范围是否越界
    """
    if label_center - r < 0:
        left_bound = 0
        right_bound = r*2
    else:
        left_bound = label_center - r
        right_bound = label_center + r
    return left_bound, right_bound


def remove_background(image, vol):
    w, h, d = image.shape
    background = np.quantile(image, 0.4)
    may_be_background = np.mean(image)
    f = 0.25
    vol[image <= background] = 0
    vol[:, :int(f*h)][image[:, :int(f*h)] <= may_be_background] = 0
    vol[:, :, :int(f*d)][image[:, :, :int(f*d)] <= may_be_background] = 0
    vol[:, int((1-f)*h):][image[:, int((1-f)*h):] <= may_be_background] = 0
    vol[:, :, int((1-f)*d):][image[:, :, int((1-f)*d):] <= may_be_background] = 0
    return vol


def get_max_connected_component(vol):
    vol_ = vol.copy()
    vol_[vol_ > 0] = 1
    vol_ = vol_.astype(np.int64)
    vol_cc = cc3d.connected_components(vol_)
    cc_sum = [(i, vol_cc[vol_cc == i].shape[0]) for i in range(vol_cc.max() + 1)]
    cc_sum.sort(key=lambda x: x[1], reverse=True)
    cc_sum.pop(0)  # remove background
    return cc_sum[1][1]


def post_processing(vol, p=0.1, mcc=None):
    """
    :param vol: Segmentation map of model's prediction.
    :param p: (0~1). A proportion of max region. The region which less than max region multiply by p
     will be remove.
    :param mcc: Max connected component of Segmentation map.
    :return: Denoised segmentation map.
    """
    vol_ = vol.copy()
    vol_[vol_ > 0] = 1
    vol_ = vol_.astype(np.int64)
    vol_cc = cc3d.connected_components(vol_)
    cc_sum = [(i, vol_cc[vol_cc == i].shape[0]) for i in range(vol_cc.max() + 1)]
    cc_sum.sort(key=lambda x: x[1], reverse=True)
    cc_sum.pop(0)  # remove background
    if mcc is not None:
        if cc_sum[1][1] < mcc*0.5:
            reduce_cc = [cc_sum[i][0] for i in range(1, len(cc_sum)) if cc_sum[i][1] > cc_sum[0][1] * p]
        else:
            reduce_cc = [9999999]
    else:
        reduce_cc = [cc_sum[i][0] for i in range(1, len(cc_sum)) if cc_sum[i][1] < cc_sum[0][1] * p]
    for i in reduce_cc:
        vol[vol_cc == i] = 0

    return vol


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'

    dataset_root = f'/data/HYK/DATASET'
    snapshot_root = f'/data/HYK/DMDS_save/msd_nih/exp0-GraNd/No_Prunning/models'
    snapshot_path = f'{snapshot_root}/iter_{20000}'

    net = torch.load(snapshot_path, map_location='cuda:0')

    from utils import load_data_test
    from skimage.transform import resize
    import matplotlib.pyplot as plt
    imglabelpath = '/data/HYK/DATASET/nnUNet_preprocessed/Task003_Pancreas/msd_nih_to_word/test/msd_091.npz'
    image, label = load_data_test(imglabelpath, dataset='msd', convert=True)
    test = TestBox(32, 64, 20, 2, net.eval())
    output0, d = test.test0(image, label, p=0.1)
    # output, d = test.test0(image, label, p=0.8)
    test.votemap_visualize(image, label, output0, save_path='/data1/HYK/DMDS_save/test0')

