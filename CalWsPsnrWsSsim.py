import numpy as np
import math
import scipy.ndimage
from numpy.ma.core import exp
from scipy.constants.constants import pi
from equilib import equi2equi
import h5py

with h5py.File('matrix_weight.h5', 'r') as f:
    weight_0 = f['weight0'][:]
    weight_36 = f['weight36'][:]
    weight_72 = f['weight72'][:]

weight0 = np.zeros((288 * 5, 2880))

for z in range(5):
    if z == 0:
        weight0[z * 288:(z + 1) * 288, :] = weight_0
        # image2[z * 288:(z + 1) * 288, :] = np.multiply(org[z], weight_0_1)
    if z == 1 or z == 2:
        weight0[z * 288:(z + 1) * 288, :] = weight_36
        # image2[z * 288:(z + 1) * 288, :] = np.multiply(org[z], weight_36_1)
    if z == 3 or z == 4:
        weight0[z * 288:(z + 1) * 288, :] = weight_72
        # image2[z * 288:(z + 1) * 288, :] = np.multiply(org[z], weight_72_1)
weight1 = weight0.copy()
weight1[weight1 > 0] = 1


def genERP(i, j, N):
    val = math.pi / N
    # w_map[i+j*w] = cos ((j - (h/2) + 0.5) * PI/h)
    w = math.cos((j - (N / 2) + 0.5) * val)
    return w


def compute_map_ws(img):
    # 计算权重
    """calculate weights for the sphere, the function provide weighting map for a given video
        :img    the input original video
    """
    equ = np.zeros((img.shape[0], img.shape[1]))

    # j是y轴上的坐标
    for j in range(0, equ.shape[0]):
        for i in range(0, equ.shape[1]):
            equ[j, i] = genERP(i, j, equ.shape[0])

    return equ


def getGlobalWSMSEValue_clip(mx, my, mw):
    val = np.sum(np.multiply((mx - my) ** 2, mw))
    den = val / np.sum(mw)

    return den


def getGlobalWSMSEValue(mx, my):
    mw = compute_map_ws(mx)
    val = np.sum(np.multiply((mx - my) ** 2, mw))
    den = val / np.sum(mw)

    return den


def pan_ws_mse(clip, org, weight):
    # order of the clip 72, -72, 36, -36, 0
    # 计算总体系数
    coe = np.sum(weight)
    # print(coe)

    m = np.zeros((1440, 2880), dtype=np.float32)

    # m_raw = np.zeros((1440, 2880), dtype=np.float32)
    # for i in range(len(org)):
    #     t = org[i]
    #     m_raw[t != 0] = t[t != 0]

    val = 0
    # 边缘的先进来，后面覆盖前面的值就可以了
    for z in range(5):
        val += np.sum(np.multiply((clip[z] - org[z]) ** 2, weight[z * 288:(z + 1) * 288, :]))

    # m[m == 0] = inp[m == 0]
    # val += np.sum(np.multiply((m - m_raw) ** 2, mw))
    # print(val, coe)
    den = val / coe
    # den = val / 3385632.0
    return den


def pan_mse(clip, org, weight):
    # order of the clip 72, -72, 36, -36, 0
    # 计算总体系数
    coe = np.sum(weight)
    # print(coe)
    val = 0

    for z in range(5):
        val += np.sum(np.multiply((clip[z] - org[z]) ** 2, weight[z * 288:(z + 1) * 288, :]))


    # m[m == 0] = inp[m == 0]
    # val += np.sum(np.multiply((m - m_raw) ** 2, mw))
    # print(val, coe)
    den = val / coe

    return den




def cal_WS_and_Nor(clip, org):

    ws_mse = pan_ws_mse(clip, org, weight0)
    mse = pan_mse(clip, org, weight1)

    try:
        ws_psnr = 20 * math.log10(255.0 / math.sqrt(ws_mse))
    except ZeroDivisionError:
        ws_psnr = np.inf

    try:
        psnr = 20 * math.log10(255.0 / math.sqrt(mse))
    except ZeroDivisionError:
        psnr = np.inf
    # print("WS-PSNR ", ws_psnr)

    # 新建一个矩阵，把五个都放进去，计算均值和方差（乘上0或1的系数后），得到map_ssim后
    # 再新建一个ws的矩阵就可以了
    image1 = np.zeros((288 * 5, 2880))
    image2 = np.zeros((288 * 5, 2880))


    # print(ws_psnr, psnr)
    for z in range(5):
        image1[z * 288:(z + 1) * 288, :] = clip[z]
        image2[z * 288:(z + 1) * 288, :] = org[z]

    image1 = np.multiply(image1, weight1)
    image2 = np.multiply(image2, weight1)

    map_ssim, index = compute_ssim(image1, image2)
    # ws = estws(map_ssim)
    ssim = np.sum(map_ssim) / weight1.sum()
    wsssim = np.sum(map_ssim * weight0) / weight0.sum()

    return ws_psnr, psnr, wsssim, ssim


def pan_ws_psnr(clip, org):
    ws_mse = pan_ws_mse(clip, org)
    try:
        ws_psnr = 20 * math.log10(255.0 / math.sqrt(ws_mse))
    except ZeroDivisionError:
        ws_psnr = np.inf
    # print("WS-PSNR ", ws_psnr)

    return ws_psnr


def ws_psnr_clip(image1, image2, weight):
    ws_mse = getGlobalWSMSEValue_clip(image1, image2, weight)
    # second estimate the ws_psnr

    try:
        ws_psnr = 20 * math.log10(255.0 / math.sqrt(ws_mse))
    except ZeroDivisionError:
        ws_psnr = np.inf
    # print("WS-PSNR ", ws_psnr)

    return ws_psnr


# img[h,w,c], image1 and image2 should have same resolution
def ws_psnr(image1, image2):
    ws_mse = getGlobalWSMSEValue(image1, image2)
    # second estimate the ws_psnr

    try:
        ws_psnr = 20 * math.log10(255.0 / math.sqrt(ws_mse))
    except ZeroDivisionError:
        ws_psnr = np.inf
    # print("WS-PSNR ", ws_psnr)

    return ws_psnr


def get_weight(height, width):
    equ = np.zeros((height, width))

    # j是y轴上的坐标
    for j in range(0, equ.shape[0]):
        for i in range(0, equ.shape[1]):
            equ[j, i] = genERP(i, j, equ.shape[0])

    return equ


def clip_weight(degree, center, height, width):
    pan = np.zeros((1, height, width))
    pan[:, int(height / 2) - center:int(height / 2) + center, :] = 1
    # degree = degree / 180.
    rot_back_1 = {
        "roll": 0 * np.pi,  #
        "pitch": degree * np.pi * -1,  # vertical
        "yaw": 0 * np.pi,  # horizontal
    }

    pan = equi2equi(
        src=pan,
        height=1440,
        width=2880,
        # mode="bilinear",
        rots=rot_back_1
    )

    # print(np.sum(pan1) - np.sum(pan))

    weight = get_weight(height, width)
    weight = np.expand_dims(weight, 0)
    pan = np.multiply(pan, weight).squeeze()

    return pan


import cv2

myfloat = np.float64


def generate_ws(i, j, M, N):
    res = np.cos((i + 0.5 - N / 2) * np.pi / N)
    return res


def estws(map_ssim):
    N, M = map_ssim.shape
    ws_map = np.zeros_like(map_ssim)

    for i in range(N):
        for j in range(M):
            ws_map[i][j] = generate_ws(i, j, M, N)

    return ws_map
    # cv2.imwrite("ws_map.png",ws_map*255)
    # import pdb; pdb.set_trace()


def compute_ssim(img_mat_1, img_mat_2):
    # Variables for Gaussian kernel definition
    gaussian_kernel_sigma = 1.5
    gaussian_kernel_width = 11
    gaussian_kernel = np.zeros((gaussian_kernel_width, gaussian_kernel_width))

    # Fill Gaussian kernel
    for i in range(gaussian_kernel_width):
        for j in range(gaussian_kernel_width):
            gaussian_kernel[i, j] = \
                (1 / (2 * pi * (gaussian_kernel_sigma ** 2))) * \
                exp(-(((i - 5) ** 2) + ((j - 5) ** 2)) / (2 * (gaussian_kernel_sigma ** 2)))

    # Convert image matrices to double precision (like in the Matlab version)
    img_mat_1 = img_mat_1.astype(np.float)
    img_mat_2 = img_mat_2.astype(np.float)

    # Squares of input matrices
    img_mat_1_sq = img_mat_1 ** 2
    img_mat_2_sq = img_mat_2 ** 2
    img_mat_12 = img_mat_1 * img_mat_2

    # Means obtained by Gaussian filtering of inputs
    img_mat_mu_1 = scipy.ndimage.filters.convolve(img_mat_1, gaussian_kernel)
    img_mat_mu_2 = scipy.ndimage.filters.convolve(img_mat_2, gaussian_kernel)

    # Squares of means
    img_mat_mu_1_sq = img_mat_mu_1 ** 2
    img_mat_mu_2_sq = img_mat_mu_2 ** 2
    img_mat_mu_12 = img_mat_mu_1 * img_mat_mu_2

    # Variances obtained by Gaussian filtering of inputs' squares
    img_mat_sigma_1_sq = scipy.ndimage.filters.convolve(img_mat_1_sq, gaussian_kernel)
    img_mat_sigma_2_sq = scipy.ndimage.filters.convolve(img_mat_2_sq, gaussian_kernel)

    # Covariance
    img_mat_sigma_12 = scipy.ndimage.filters.convolve(img_mat_12, gaussian_kernel)

    # Centered squares of variances
    img_mat_sigma_1_sq = img_mat_sigma_1_sq - img_mat_mu_1_sq
    img_mat_sigma_2_sq = img_mat_sigma_2_sq - img_mat_mu_2_sq
    img_mat_sigma_12 = img_mat_sigma_12 - img_mat_mu_12;

    # c1/c2 constants
    # First use: manual fitting
    c_1 = 6.5025
    c_2 = 58.5225

    # Second use: change k1,k2 & c1,c2 depend on L (width of color map)
    l = 255
    k_1 = 0.01
    c_1 = (k_1 * l) ** 2
    k_2 = 0.03
    c_2 = (k_2 * l) ** 2

    # Numerator of SSIM
    num_ssim = (2 * img_mat_mu_12 + c_1) * (2 * img_mat_sigma_12 + c_2)
    # Denominator of SSIM
    den_ssim = (img_mat_mu_1_sq + img_mat_mu_2_sq + c_1) * \
               (img_mat_sigma_1_sq + img_mat_sigma_2_sq + c_2)
    # SSIM
    ssim_map = num_ssim / den_ssim
    index = np.average(ssim_map)

    # print index

    return ssim_map, index


def ws_ssim(image1, image2):
    map_ssim, MSSIM = compute_ssim(image1, image2)
    ws = estws(map_ssim)
    wsssim = np.sum(map_ssim * ws) / ws.sum()
    # print(wsssim)
    print("WS-SSIM ", wsssim)

    return wsssim
