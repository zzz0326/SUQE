from equilib import equi2equi
import torch
import numpy as np
import torchvision.transforms as transforms
import utils
import tqdm
import sys
from collections import OrderedDict
from net_stdf import MFVQE
import cv2
import os
import glob

transform = transforms.Compose([
    transforms.ToTensor()
])

train_list = ['G10BoatInPark', 'G10BodybuildingWorkout', 'G10DrivingInCountry', 'G10XiaoGuang', 'G10PandaBaseChengdu',
              'G1AbandonedKingdom', 'G1Aerial', 'G1BajaCalifonia', 'G1BikingToWork', 'G1LateShow', 'G2AstonVillaGoal',
              'G2CougarsTreats', 'G2ForgottenBook', 'G2FormationPace', 'G2VictoriaFalls', 'G3BackcountrySkiing',
              'G3GetYoGurl', 'G3Sailing49er', 'G3SkyrimHelgen', 'G4BikingInSaalbach', 'G4HachaWaterfall',
              'G4WingsuitFlight', 'G5AngelFalls', 'G5BearSwimming', 'G5Neighborhood', 'G5ResistMarch',
              'G5SubwayConstruction', 'G6AngelFallsClimbing', 'G6GTRDriving', 'G6YeahBoy', 'G6ManhattanNight',
              'G6YosemiteGliding', 'G7AuroraQuartzLake', 'G7DragonCastleAttatck', 'G7PressConference', 'G7Shooting',
              'G7UcaimaWaterfall', 'G8ANewEmpire', 'G8DivingWithSharks', 'G8Pagoda', 'G8Salon', 'G8YourMan',
              'G9DivingWithJellyfish', 'G9DrivingInCity', 'G9FootballMatch', 'G9BloomingAppleOrchards']

ext_list = ['G10PandaBaseChengdu', 'G9BloomingAppleOrchards', 'G6ManhattanNight']


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

            list_k = [72, -72, 36, -36, 0]
            for w in range(len(list_k)):
                zzz = list_k[w]
                pitch = zzz / 180.
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

                with open(output_path + video_name + ".yuv", "ab") as f:
                    f.write(img_yuv.tobytes())



# # for raw training clip
# raw_path = '/data1/360Enhance/VQA_dataset/Group'
#
# for i in range(10):
#     now_path = raw_path + str(i + 1) + '/' + 'Reference/'
#     for file in os.listdir(now_path):
#         if file.endswith(".yuv"):
#             if file.split('_')[0] in ext_list:
#                 file_path = now_path + file
#                 out_path = '/data1/360Enhance/train_data/raw/'
#                 weight = int(file.split('_')[1].split('x')[0])
#                 height = int(file.split('_')[1].split('x')[1])
#                 rotate_erp(file_path, out_path, weight, height)
#             else:
#                 pass


# directory = "/data1/360Enhance/VQA_dataset/"
#
# file_pattern = os.path.join(directory, "*.mp4")
# files = glob.glob(file_pattern)
#
# size_list = []
# for i in range(len(files)):
#     size_list.append(files[i].split('/')[-1].split('_')[0]+'_'+files[i].split('_')[3])


# folder_path = "/data/360Enhance/VQA_dataset/"
#
# # 获取文件夹中所有文件的名称
# files = os.listdir(folder_path)
#
# # 遍历文件并修改文件名
# for file in files:
#     # 获取文件名和文件扩展名
#     file_name, file_ext = os.path.splitext(file)
#
#     # 用下划线替换空格
#     name = file_name.split('_')[0]
#     qp = file_name.split('_')[1]
#
#     for i in range(len(size_list)):
#         if size_list[i].split('_')[0] == name:
#             size = size_list[i].split('_')[1]
#     new_file_name = name + '_' + size + '_' + qp
#     # 生成新文件名（包括文件扩展名）
#     new_file = new_file_name + file_ext
#
#     # print(new_file)
#     # 生成原文件路径和新文件路径
#     old_file_path = os.path.join(folder_path, file)
#     new_file_path = os.path.join(folder_path, new_file)
#
#     # 重命名文件
#     os.rename(old_file_path, new_file_path)

# path = '/data1/360Enhance/Aug_train/raw_video/test/'
# for file in os.listdir(path):
#     if file.endswith(".yuv"):
#         if file.split('_')[0] not in train_list:
#             file_path = path + file
#             # qp = file.split('_')[2].split('.')[0]
#             out_path = '/data1/360Enhance/test_data/raw/'
#             weight = int(file.split('_')[1].split('x')[0])
#             height = int(file.split('_')[1].split('x')[1])
#             # print(file_path, out_path + qp + '/', weight, height)
#             rotate_erp(file_path, out_path, weight, height)
#         else:
#             pass

out_path = '/data1/360Enhance/test_data/'
path = '/data/360Enhance/VQA_dataset/'
for file in os.listdir(path):
    if file.endswith("qp22.yuv") or file.endswith("qp32.yuv"):
        if file.split('_')[0] not in train_list:
            file_path = path + file
            qp = file.split('_')[2].split('.')[0]
            weight = int(file.split('_')[1].split('x')[0])
            height = int(file.split('_')[1].split('x')[1])
            # print(file_path, out_path + qp + '/', weight, height)
            rotate_erp(file_path, out_path + qp + '/', weight, height)
        else:
            pass



# def rotate_center_erp(input_path, output_path, width, height):
#     video_name = input_path.split('/')[-1].split('_')[0]
#     size = width * height * 3 // 2
#
#     with open(input_path, "rb") as f:
#         all_data = np.frombuffer(f.read(300 * size), dtype=np.uint8)
#         for i in range(300):
#             x = all_data[size * i:size * (i + 1)].reshape((height * 3 // 2, width))
#
#             x = cv2.cvtColor(x, cv2.COLOR_YUV2RGB_I420)
#             x = transform(x)
#             x = x.to('cuda')
#             # x = torch.unsqueeze(x, 0)
#
#             # for w in range(-45, 50, 5):
#             w = -25
#             pitch = w / 180.
#             rot = {
#                 "roll": 0 * np.pi,  #
#                 "pitch": pitch * np.pi,  # vertical
#                 "yaw": 0 * np.pi,  # horizontal
#             }
#             img = equi2equi(
#                 src=x,
#                 height=1440,
#                 width=2880,
#                 # mode="bilinear",
#                 rots=rot
#             )
#             img = torch.squeeze(img, 0)
#             img = img.permute(1, 2, 0).cpu().numpy() * 255
#             img = img[576:864, :, :]
#             img = cv2.convertScaleAbs(img)
#
#             img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV_I420)
#
#             with open(output_path + video_name + '_' + str(w) + ".yuv", "ab") as f:
#                 f.write(img_yuv.tobytes())



# rotate_center_erp('/data/360Enhance/VQA_dataset/G7UcaimaWaterfall_3840x1920_qp27.yuv', '/data/360Enhance/train_set/com/qp27/', width=3840, height=1920)