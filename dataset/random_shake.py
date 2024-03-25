import glob
from sphere import viewport_alignment_2view
import random
import torch
import os.path as op
import numpy as np
import cv2
from torch.utils import data as data
from utils import FileClient, paired_random_crop, augment, totensor, import_yuv, shake_random_crop
from read_YFromyuv420 import read_y


def _bytes2img(img_bytes):
    img_np = np.frombuffer(img_bytes, np.uint8)
    img = np.expand_dims(cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE), 2)  # (H W 1)
    img = img.astype(np.float32) / 255.
    return img


class Shake_shake(data.Dataset):
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

        # get the neighboring LQ frames
        # 还是随机剪裁，确定裁剪的位置（x，y），第一个序列都在（x，y）上，第二个序列除第4帧以外在（x，y）上都进行随机抖动。
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
        # img_gt, img_lqs = paired_random_crop(
        #     img_gt, img_lqs, gt_size, img_gt_path
        # )

        unstable, stable = shake_random_crop(
            img_lqs, gt_size
        )
        # print(len(unstable), len(stable))

        # flip, rotate
        for w in range(len(stable)):
            unstable.append(stable[w])  # gt joint augmentation with lq
        unstable = augment(
            unstable, self.opts_dict['use_flip'], self.opts_dict['use_rot']
        )

        # to tensor
        unstable = totensor(unstable)

        stable = torch.stack(unstable[int(len(unstable) / 2):int(len(unstable))], dim=0)
        # print(len(unstable))
        unstable = torch.stack(unstable[0:int(len(unstable)/2)], dim=0)


        return {
            'unstable': unstable,
            'stable': stable
            # 'view_one': img_lqs[:, :, :, 0: 256],  # (T [RGB] H W)
            # 'view_two': img_lqs[:, :, :, 128: 384],  # ([RGB] H W)
        }

    def __len__(self):
        return len(self.keys)


