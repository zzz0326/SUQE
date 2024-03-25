import glob
from sphere import viewport_alignment_2view
import random
import torch
import os.path as op
import numpy as np
import cv2
from torch.utils import data as data
from utils import FileClient, paired_random_crop, augment, totensor, import_yuv
from read_YFromyuv420 import read_y


def _bytes2img(img_bytes):
    img_np = np.frombuffer(img_bytes, np.uint8)
    img = np.expand_dims(cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE), 2)  # (H W 1)
    img = img.astype(np.float32) / 255.
    return img

def cal_7frame_optical_flow(frame_list):
    # 输入七帧 0-2 4-6帧分别与第3帧比较 然后放在自己上面
    frame_main = frame_list[3]
    frame_list.__delitem__(3)
    for i in range(6):
        flow = cv2.calcOpticalFlowFarneback(frame_list[i], frame_main, None, 0.5, 3, 15, 3, 5, 1.1,
                                            cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        # 计算光流的大小
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mag_mean = np.mean(mag)

        # 找到光流较大的区域 将光流较小的区域加上噪声
        mask = np.zeros_like(frame_list[i])
        mask[mag > mag_mean * 3] = 255
        # mask = mask[:, :, np.newaxis]
        noise = add_noise(np.zeros(shape=(mask.shape[0], mask.shape[1], 1)))
        noise_masked = noise * (mask / 255)
        frame_list[i][:, 0:128, :] = frame_list[i][:, 0:128, :] + noise_masked[:, 0:128, :]
        frame_list[i] = frame_list[i].astype(np.uint8)
    frame_list.insert(3, frame_main)
    return frame_list


def add_noise(input_data, trans_type="gauss_noise", rank='cuda'):
    if trans_type == "gauss_noise":
        # noised_data = torch.zeros_like(input_data, dtype=input_data.dtype).to(device)
        #
        # b, c, h, w = input_data.shape
        # noise = np.random.normal(loc=0.0, scale=0.04, size=(b, 1, h, w)) * 0.1
        # noise = torch.from_numpy(noise).to(device)
        # noise = noise.type(input_data.dtype)
        #
        # noised_data = noised_data + input_data
        # noised_data[:, [radius], ...] = noise + input_data[:, [radius], ...]
        # noised_data = noised_data.clamp(0, 1)

        # noised_data = np.random.normal(loc=0.0, scale=0.04, size=input_data.shape)
        noised_data = np.random.normal(loc=0.0, scale=0.04, size=input_data.shape)   # for QP37
        # noised_data = np.random.normal(loc=0.0, scale=0.04, size=input_data.shape) * 0.05  # for QP22
        # noised_data = torch.from_numpy(noised_data).to(rank)
        # noised_data = noised_data.type(input_data.dtype)

        noised_data = noised_data + input_data

        noised_data = np.clip(noised_data, 0, 1)


    return noised_data


class optical_noisy(data.Dataset):
    """MFQEv2 dataset.

    For training data: LMDB is adopted. See create_lmdb for details.

    Return: A dict includes:
        img_lqs: (T, [RGB], H, W)
        img_gt: ([RGB], H, W)
        key: str
    """

    def __init__(self, opts_dict, radius, rank):
        super().__init__()

        self.opts_dict = opts_dict
        self.rank = rank
        # dataset paths
        self.gt_root = self.opts_dict['gt_path']
        # self.gt_root = op.join(
        #     'data/MFQEv2/',
        #     self.opts_dict['gt_path']
        # )
        self.lq_root = self.opts_dict['lq_path']
        # self.lq_root = op.join(
        #     'data/MFQEv2/',
        #     self.opts_dict['lq_path']
        # )

        # extract keys from meta_info.txt
        self.meta_info_path = op.join(
            self.gt_root,
            self.opts_dict['meta_info_fp']
        )
        with open(self.meta_info_path, 'r') as fin:
            self.keys = [line.split(' ')[0] for line in fin]

        # define file client
        self.file_client = None
        self.io_opts_dict = dict()  # FileClient needs
        self.io_opts_dict['type'] = 'lmdb'
        self.io_opts_dict['db_paths'] = [
            self.lq_root,
            self.gt_root
        ]
        self.io_opts_dict['client_keys'] = ['lq', 'gt']

        # generate neighboring frame indexes
        # indices of input images
        # radius | nfs | input index
        # 0      | 1   | 4, 4, 4  # special case, for image enhancement
        # 1      | 3   | 3, 4, 5
        # 2      | 5   | 2, 3, 4, 5, 6
        # 3      | 7   | 1, 2, 3, 4, 5, 6, 7
        # no more! septuplet sequences!
        if radius == 0:
            self.neighbor_list = [4, 4, 4]  # always the im4.png
        else:
            nfs = 2 * radius + 1
            self.neighbor_list = [i + (9 - nfs) // 2 for i in range(nfs)]

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_opts_dict.pop('type'), **self.io_opts_dict
            )
        # random reverse
        if self.opts_dict['random_reverse'] and random.random() < 0.5:
            self.neighbor_list.reverse()

        # ==========
        # get frames
        # ==========

        # get the GT frame (im4.png)
        gt_size = self.opts_dict['gt_size']
        key = self.keys[index]
        clip, seq, _ = key.split('/')  # key example: 00001/0001/im1.png

        # load Enlarger view
        # clip:视频数量 seq:七帧为一个单位 将一个视频分成多个seq
        # video_order = int(clip) - 1
        # frame_order = (int(seq) - 1) * 7 + 3
        #
        # yuv_path = '/data/360Enhance/video_sequence/Enlarger/0_' + str(video_order) + '_' + str(frame_order) + '.yuv'
        # Enlarger_view = read_y(yuv_path)

        img_gt_path = key
        # img_bytes = self.file_client.get(img_gt_path, 'gt')
        # img_gt = _bytes2img(img_bytes)  # (H W 1)

        # get the neighboring LQ frames
        img_lqs = []
        for neighbor in self.neighbor_list:
            img_lq_path = f'{clip}/{seq}/im{neighbor}.png'
            # print(img_lq_path)
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = _bytes2img(img_bytes)  # (H W 1)
            # 600, 540, 1
            img_lqs.append(img_lq)

        # img_lqs.append(Enlarger_view)

        # cat the enlarger data with lq_data



        # ==========
        # data augmentation
        # ==========

        # randomly crop
        img_lqs_noise, img_lqs = paired_random_crop(
            img_lqs, img_lqs, gt_size, img_gt_path
        )

        # flip, rotate

        img_results = augment(
            img_lqs, self.opts_dict['use_flip'], self.opts_dict['use_rot']
        )


        img_lqs_noise = cal_7frame_optical_flow(img_results)
        img_lqs_noise = totensor(img_lqs_noise)
        img_lqs_noise = torch.stack(img_lqs_noise[0:7], dim=0)
        # to tensor
        img_results = totensor(img_results)
        img_lqs = torch.stack(img_results[0:7], dim=0)

        # add noisy




        return {
            'lqs_noise': img_lqs_noise[:, :, :, 0: 256],
            'lqs': img_lqs[:, :, :, 128: 384],
            # 'view_one': img_lqs[:, :, :, 0: 256],  # (T [RGB] H W)
            # 'view_two': img_lqs[:, :, :, 128: 384],  # ([RGB] H W)
        }

    def __len__(self):
        return len(self.keys)