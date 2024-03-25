from CalWsPsnrWsSsim import ws_psnr_clip, ws_ssim, clip_weight, pan_ws_psnr, ws_psnr, cal_WS_and_Nor
from equilib import equi2equi
import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
import utils
import tqdm
import sys
from collections import OrderedDict
from net_stdf import MFVQE
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# img = torch.zeros((3, 1440, 2880))
# total = 2880 * 1440
#
# # 每个36度 增强五次
# # 288个像素
#
# list_k = [72, -72, 36, -36, 0]
# start = 0
# for x in range(len(list_k)):
#     k = list_k[x]
#     pitch = k / 180.
#     rot = {
#         "roll": 0 * np.pi,  #
#         "pitch": pitch * np.pi,  # vertical
#         "yaw": 0 * np.pi,  # horizontal
#     }
#     img = equi2equi(
#         src=img,
#         height=1440,
#         width=2880,
#         # mode="bilinear",
#         rots=rot
#     )
#     # img = torch.squeeze(img, 0)
#     # img = img.permute(1, 2, 0).cpu().numpy() * 255
#
#     img[:, 576:864, :] = 1
#
#     # enhance the img
#     # 1 means enhancement
#
#
#
#     rot_back = {
#         "roll": 0 * np.pi,  #
#         "pitch": pitch * np.pi * -1,  # vertical
#         "yaw": 0 * np.pi,  # horizontal
#     }
#
#     img = equi2equi(
#         src=img,
#         height=1440,
#         width=2880,
#         # mode="bilinear",
#         rots=rot_back
#     )
#     img2 = img.permute(1, 2, 0).cpu().numpy() * 255
#     count = np.sum(img2 == 255)/3
#     now = count / total
#     print(now, now-start)
#     start = now
#     cv2.imwrite('test+' + str(k) + '.png', img2)
# # img = torch.squeeze(img, 0)
# img = img.permute(1, 2, 0).cpu().numpy() * 255
# img = cv2.convertScaleAbs(img)
#
# # img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV_I420)
#
# cv2.imwrite('aaaaa.png', img)

test_list = []
test_path = '/data1/360Enhance/Aug_train/raw_video/test/'
for file in os.listdir(test_path):
    if file.endswith(".yuv"):
        test_list.append(file.split('/')[-1].split('_')[0])

list_k = [0, 36, -36, 72, -72]
transform = transforms.Compose([
    transforms.ToTensor()
])

weight = cv2.imread('/data1/360Enhance/VQA_dataset/Group8/Reference/aaaaa.png', 0)


def rotate_erp(input_path, output_path, width, height):
    video_name = input_path.split('/')[-1].split('_')[0]
    size = width * height * 3 // 2

    with open(input_path, "rb") as f:
        all_data = np.frombuffer(f.read(300 * size), dtype=np.uint8)
        for i in range(300):
            x = all_data[size * i:size * (i + 1)].reshape((height * 3 // 2, width))

            x = cv2.cvtColor(x, cv2.COLOR_YUV2RGB_I420)
            x = transform(x)
            x = x.to('cuda')
            # x = torch.unsqueeze(x, 0)

            for w in range(len(list_k)):
                k = list_k[w]
                pitch = k / 180.
                rot = {
                    "roll": 0 * np.pi,  #
                    "pitch": pitch * np.pi,  # vertical
                    "yaw": 0 * np.pi,  # horizontal
                }
                img = equi2equi(
                    src=x,
                    height=1440,
                    width=2880,
                    # mode="bilinear",
                    rots=rot
                )
                img = torch.squeeze(img, 0)
                img = img.permute(1, 2, 0).cpu().numpy() * 255
                img = img[576:864, :, :]
                img = cv2.convertScaleAbs(img)

                img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV_I420)

                with open(output_path + video_name + '_' + str(k) + ".yuv", "ab") as f:
                    f.write(img_yuv.tobytes())


# rotate_erp('/data1/360Enhance/VQA_dataset/Group8/Reference/G8AlpsParagliding_3840x1920_fps25.yuv', '/data1/360Enhance/VQA_dataset/Group8/Reference/PanEnh/', 3840, 1920)

ckp_path = '/home/zouzizhuang/stdf/exp/clip_37/ckp_250000.pt'
# ckp_path = '/home/zouzizhuang/stdf/exp/clip_27/ckp_300000.pt'
# output_file = open('/data1/360Enhance/VQA_dataset/Group8/Reference/output_stable.txt', 'w')
# sys.stdout = output_file

# ==========
# Load pre-trained model
# ==========
opts_dict = {
    'radius': 3,
    'stdf': {
        'in_nc': 1,
        'out_nc': 64,
        'nf': 32,
        'nb': 3,
        'base_ks': 3,
        'deform_ks': 3,
    },
    'qenet': {
        'in_nc': 64,
        'out_nc': 1,
        'nf': 48,
        'nb': 8,
        'base_ks': 3,
    },
}

t_total_p = 0
t_total_wp = 0
t_total_ws = 0
t_total_s = 0

model = MFVQE(opts_dict=opts_dict)
msg = f'loading model {ckp_path}...'
print(msg)
checkpoint = torch.load(ckp_path)
if 'module.' in list(checkpoint['state_dict'].keys())[0]:  # multi-gpu training
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        name = k[7:]  # remove module
        new_state_dict[name] = v
    # new_state_dict['ffnet.deform_conv.weight'] = new_state_dict['ffnet.deform_conv.weight'][0:63, :, :, :]
    # new_state_dict['ffnet.deform_conv.bias'] = new_state_dict['ffnet.deform_conv.bias'][0:63]
    # new_state_dict['qenet.in_conv.0.weight'] = new_state_dict['qenet.in_conv.0.weight'][:, 0:63, :, :]
    model.load_state_dict(new_state_dict)
else:  # single-gpu training
    model.load_state_dict(checkpoint['state_dict'])

msg = f'> model {ckp_path} loaded.'
print(msg)
model = model.cuda()
model.eval()

for k in range(len(test_list)):

    raw_yuv_path = '/data1/360Enhance/test_data/raw/' + test_list[k] + '_'
    lq_yuv_path = '/data1/360Enhance/test_data/qp37/' + test_list[k] + '_'

    h = 288
    w = 2880
    nfs = 300

    list_degree = [0, 36, -36, 72, -72]

    raw_y = []
    for i in range(5):
        degree = list_degree[i]

        en_t = utils.import_yuv(
            seq_path=raw_yuv_path + str(int(degree)) + '.yuv', h=288, w=2880, tot_frm=nfs, start_frm=0, only_y=True
        )
        en_t = en_t.astype(np.float32) / 255.
        raw_y.append(en_t)

    lq_y = []
    for i in range(5):
        degree = list_degree[i]

        en_t = utils.import_yuv(
            seq_path=lq_yuv_path + str(int(degree)) + '.yuv', h=288, w=2880, tot_frm=nfs, start_frm=0, only_y=True
        )
        en_t = en_t.astype(np.float32) / 255.
        lq_y.append(en_t)

    # msg = '> yuv loaded.'
    # print(msg)

    # ==========
    # Define criterion
    # ==========
    criterion = utils.PSNR()
    unit = 'dB'

    # ==========
    # Test
    # ==========
    # pbar = tqdm(total=nfs, ncols=80)
    ori_psnr_counter = utils.Counter()
    enh_psnr_counter = utils.Counter()
    total_wp = 0
    total_ws = 0
    total_p = 0
    total_s = 0
    total = 0
    for idx in range(50, 70):
        # load lq

        # pan_y = torch.from_numpy(lq_w_y[idx]).unsqueeze(0).cuda()

        # rot = {
        #     "roll": 0 * np.pi,  #
        #     "pitch": 0 * np.pi,  # vertical
        #     "yaw": 0 * np.pi,  # horizontal
        # }
        # pan_y = equi2equi(
        #     src=pan_y,
        #     height=1440,
        #     width=2880,
        #     # mode="bilinear",
        #     rots=rot
        # )

        input_data_1 = []
        input_data_2 = []
        input_data_3 = []
        input_data_4 = []
        input_data_5 = []

        idx_list = list(range(idx - 3, idx + 4))
        idx_list = np.clip(idx_list, 0, nfs - 1)
        for idx_ in idx_list:
            input_data_1.append(lq_y[0][idx_])
            input_data_2.append(lq_y[1][idx_])
            input_data_3.append(lq_y[2][idx_])
            input_data_4.append(lq_y[3][idx_])
            input_data_5.append(lq_y[4][idx_])
        input_data_1 = torch.from_numpy(np.array(input_data_1))
        input_data_1 = torch.unsqueeze(input_data_1, 0).cuda()

        input_data_2 = torch.from_numpy(np.array(input_data_2))
        input_data_2 = torch.unsqueeze(input_data_2, 0).cuda()

        input_data_3 = torch.from_numpy(np.array(input_data_3))
        input_data_3 = torch.unsqueeze(input_data_3, 0).cuda()

        input_data_4 = torch.from_numpy(np.array(input_data_4))
        input_data_4 = torch.unsqueeze(input_data_4, 0).cuda()

        input_data_5 = torch.from_numpy(np.array(input_data_5))
        input_data_5 = torch.unsqueeze(input_data_5, 0).cuda()

        enhanced_frm = torch.zeros((5, 1, 1, 288, 2880), dtype=torch.float32).cuda()
        # print(input_data_1[:, :, :, 288 * 0:288 * 1].shape)
        # enhance

        for z in range(10):
            enhanced_frm[0, :, :, :, 288 * z:288 * (z + 1)] = model(input_data_1[:, :, :, 288 * z:288 * (z + 1)])
            enhanced_frm[1, :, :, :, 288 * z:288 * (z + 1)] = model(input_data_2[:, :, :, 288 * z:288 * (z + 1)])
            enhanced_frm[2, :, :, :, 288 * z:288 * (z + 1)] = model(input_data_3[:, :, :, 288 * z:288 * (z + 1)])
            enhanced_frm[3, :, :, :, 288 * z:288 * (z + 1)] = model(input_data_4[:, :, :, 288 * z:288 * (z + 1)])
            enhanced_frm[4, :, :, :, 288 * z:288 * (z + 1)] = model(input_data_5[:, :, :, 288 * z:288 * (z + 1)])
        del input_data_1
        del input_data_2
        del input_data_3
        del input_data_4
        del input_data_5
        # for z in range(5,10):
        #     enhanced_frm[0, :, :, :, 288 * z:288 * (z + 1)] = input_data_1[:, 3, :, 288 * z:288 * (z + 1)]
        #     enhanced_frm[1, :, :, :, 288 * z:288 * (z + 1)] = input_data_2[:, 3, :, 288 * z:288 * (z + 1)]
        #     enhanced_frm[2, :, :, :, 288 * z:288 * (z + 1)] = input_data_3[:, 3, :, 288 * z:288 * (z + 1)]
        #     enhanced_frm[3, :, :, :, 288 * z:288 * (z + 1)] = input_data_4[:, 3, :, 288 * z:288 * (z + 1)]
        #     enhanced_frm[4, :, :, :, 288 * z:288 * (z + 1)] = input_data_5[:, 3, :, 288 * z:288 * (z + 1)]

        # rotation

        en_list = []
        org_list = []
        raw_list = []
        for w in range(5):
            # enh_map = torch.zeros((288, 2880), dtype=torch.float32)

            # k = list_k[w]
            # pitch = k / 180.
            # rot = {
            #     "roll": 0 * np.pi,  #
            #     "pitch": pitch * np.pi,  # vertical
            #     "yaw": 0 * np.pi,  # horizontal
            # }

            # enh_map = equi2equi(
            #     src=enh_map,
            #     height=1440,
            #     width=2880,
            #     # mode="bilinear",
            #     rots=rot
            # )




            # ws-psnr
            pitch = list_degree[w] / 180.
            rot_back = {
                "roll": 0 * np.pi,  #
                "pitch": pitch * np.pi * -1,  # vertical
                "yaw": 0 * np.pi,  # horizontal
            }


            en_list.append(enhanced_frm[w].squeeze(0).cpu().detach().numpy().squeeze(0))
            # en_list.append(enh_map)
            org_list.append(lq_y[w][idx])
            raw_list.append(raw_y[w][idx])
        ws_psnr0, psnr0, wsssim0, ssim0 = cal_WS_and_Nor(en_list, raw_list)
        ws_psnr1, psnr1, wsssim1, ssim1 = cal_WS_and_Nor(org_list, raw_list)
        ws_psnr = ws_psnr0 - ws_psnr1
        psnr = psnr0 - psnr1
        wsssim = wsssim0 - wsssim1
        ssim = ssim0 - ssim1
        # print(ws_psnr, psnr, wsssim, ssim)
        total_ws += wsssim
        total_p += psnr
        total_wp += ws_psnr
        total_s += ssim
        # total += pan_ws_psnr(en_list, raw_list) - pan_ws_psnr(org_list, raw_list)
        del enhanced_frm
        del raw_list
        del org_list
        del en_list

        #
        # lq_save = lq_y[w][i] * 255
        # lq_save = lq_save.astype(np.uint8)
        # enh_map = enh_map.detach().numpy() * 255
        # enh_map = enh_map.astype(np.uint8)

        # with open("/home/zouzizhuang/stdf/EnhResult" + '_' + str(idx) + '_' + str(w) + ".yuv", "wb") as f:
    print(test_list[k], total_wp / 20., total_p / 20., total_ws / 20., total_s / 20.)
    t_total_wp += total_wp
    t_total_p += total_p
    t_total_ws += total_ws
    t_total_s += total_s
print(t_total_wp / (len(test_list) * 20.), t_total_p / (len(test_list) * 20.), t_total_ws / (len(test_list) * 20.),
      t_total_s / (len(test_list) * 20.))
    #     f.write(enh_map.tobytes())
    # for t in range(1440):
    #     for v in range(2880):
    #         if weight[t, v] != 255:
    #             enh_map[0, t, v] = pan_y[0, t, v]
    # 转换以后有损失

    # img2 = img.permute(1, 2, 0).cpu().numpy() * 255

    # eval
    #     gt_frm = torch.from_numpy(raw_y[idx]).cuda()
    #     batch_ori = criterion(input_data[0, 3, ...], gt_frm)
    #     batch_perf = criterion(enhanced_frm[0, 0, ...], gt_frm)
    #     ori_psnr_counter.accum(volume=batch_ori)
    #     enh_psnr_counter.accum(volume=batch_perf)
    #
    #     # display
    #     pbar.set_description(
    #         "[{:.3f}] {:s} -> [{:.3f}] {:s}"
    #             .format(batch_ori, unit, batch_perf, unit)
    #     )
    #     pbar.update()
    #
    # pbar.close()
