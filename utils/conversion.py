import random
import torch
import skimage.color as skc
import numpy as np
import cv2


def img2float32(img):
    """Convert the type and range of the input image into np.float32 and [0, 1].

    Args:
        img (img in ndarray):
            1. np.uint8 type (of course with range [0, 255]).
            2. np.float32 type, with unknown range.

    Return:
        img (ndarray): The converted image with type of np.float32 and 
        range of [0, 1].
    """
    img_type = img.dtype
    assert img_type in (np.uint8, np.float32), (
        f'The image type should be np.float32 or np.uint8, but got {img_type}')

    if img_type == np.uint8:  # the range must be [0, 255]
        img = img.astype(np.float32)
        img /= 255.
    else:  # np.float32, may excess the range [0, 1]
        img = img.clip(0, 1)

    return img


def ndarray2img(ndarray):
    """Convert the type and range of the input ndarray into np.uint8 and 
    [0, 255].

    Args:
        ndarray (ndarray):
            1. np.uint8 type (of course with range [0, 255]).
            2. np.float32 type with unknown range.

    Return:
        img (img in ndarray): The converted image with type of np.uint8 and 
        range of [0, 255].
    
    
    对float32类型分情况讨论: 
        1. 如果最大值超过阈值, 则视为较黑的图像, 直接clip处理；
        2. 否则, 视为[0, 1]图像处理后的结果, 乘以255.再clip.
    
    不能直接astype, 该操作会删除小数, 不精确. 应先round, 再clip, 再转换格式.
    
    image -> img2float32 -> ndarray2img 应能准确还原.
    """
    data_type = ndarray.dtype
    assert data_type in (np.uint8, np.float32), (
        f'The data type should be np.float32 or np.uint8, but got {data_type}')

    if data_type == np.float32:
        detection_threshold = 2
        if (ndarray < detection_threshold).all():  # just excess [0, 1] slightly
            ndarray *= 255.
        else:  # almost a black picture
            pass
        img = ndarray.round()  # first round. directly astype will cut decimals
        img = img.clip(0, 255)  # or, -1 -> 255, -2 -> 254!
        img = img.astype(np.uint8)
    else:
        img = ndarray

    return img


def rgb2ycbcr(rgb_img):
    """RGB to YCbCr color space conversion.

    Args:
        rgb_img (img in ndarray): (..., 3) format.

    Return:
        ycbcr_img (img in ndarray): (..., 3) format.

    Error:
        rgb_img is not in (..., 3) format.

    Input image, not float array!

    Y is between 16 and 235.
    
    YCbCr image has the same dimensions as input RGB image.
    
    This function produces the same results as Matlab's `rgb2ycbcr` function.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    It differs from a similar function in cv2.cvtColor: `RGB <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.
    """
    ycbcr_img = skc.rgb2ycbcr(rgb_img)
    return ycbcr_img


def ycbcr2rgb(ycbcr_img):
    """YCbCr to RGB color space conversion.

    Args:
        ycbcr_img (img in ndarray): (..., 3) format.

    Return:
        rgb_img (img in ndarray): (..., 3) format.

    Error:
        ycbcr_img is not in (..., 3) format.

    Input image, not float array!

    Y is between 16 and 235.
    
    YCbCr image has the same dimensions as input RGB image.
    
    This function produces the same results as Matlab's `ycbcr2rgb` function.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    It differs from a similar function in cv2.cvtColor: `RGB <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.
    """
    rgb_img = skc.ycbcr2rgb(ycbcr_img)
    return rgb_img


def rgb2gray(rgb_img):
    """Compute luminance of an RGB image.

    Args:
        rgb_img (img in ndarray): (..., 3) format.

    Return:
        gray_img (single channel img in array)

    Error:
        rgb_img is not in (..., 3) format.

    Input image, not float array!

    alpha通道会被忽略.
    """
    gray_img = skc.rgb2gray(rgb_img)
    return gray_img


def gray2rgb(gray_img):
    """Create an RGB representation of a gray-level image.

    Args:
        gray_img (img in ndarray): (..., 1) or (... , ) format.

    Return:
        rgb_img (img in ndarray)
    
    Input image, not float array!

    其实还有一个alpha通道参数, 但不常用. 参见: 
    https://scikit-image.org/docs/dev/api/skimage.color.html#skimage.color.gray2rgb
    """
    rgb_img = skc.gray2rgb(gray_img, alpha=None)
    return rgb_img


def bgr2rgb(img):
    code = getattr(cv2, 'COLOR_BGR2RGB')
    img = cv2.cvtColor(img, code)
    return img


def rgb2bgr(img):
    code = getattr(cv2, 'COLOR_RGB2BGR')
    img = cv2.cvtColor(img, code)
    return img


# ==========
# Data augmentation
# ==========

def shake_random_crop(img_lqs, gt_patch_size, scale=1):
    """Paired random crop.

    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        gt_patch_size (int): GT patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    h_lq, w_lq, _ = img_lqs[0].shape

    lq_patch_size = gt_patch_size // scale

    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). ')

    # randomly choose top and left coordinates for lq patch
    top = random.randint(10, h_lq - lq_patch_size - 10)
    left = random.randint(10, w_lq - lq_patch_size - 10)

    # crop lq patch
    stable = [
        v[top:top + lq_patch_size, left:left + lq_patch_size, ...]
        for v in img_lqs
    ]
    # down = h_lq - top - lq_patch_size
    # right = w_lq - left - lq_patch_size

    # unstable = []
    # for w in range(len(img_lqs)):
    #     move_h = 3 - w
    #     move_w = 3 - w
    #     unstable.append(
    #         img_lqs[w][top + move_h:top + lq_patch_size + move_h, left + move_w:left + lq_patch_size + move_w, ...])

    unstable = []
    for w in range(len(img_lqs)):
        if w == int(len(img_lqs) / 2):
            unstable.append(img_lqs[w][top:top + lq_patch_size, left:left + lq_patch_size, ...])
        else:
            seed = random.randint(0, 3)
            if seed == 0:
                move_h = - random.randint(0, 2)
                move_w = - random.randint(0, 2)
            if seed == 1:
                move_h = - random.randint(0, 2)
                move_w = random.randint(0, 2)
            if seed == 2:
                move_h = random.randint(0, 2)
                move_w = - random.randint(0, 2)
            if seed == 3:
                move_h = random.randint(0, 2)
                move_w = random.randint(0, 2)
            unstable.append(
                img_lqs[w][top + move_h:top + lq_patch_size + move_h, left + move_w:left + lq_patch_size + move_w, ...])

    # unstable = [
    #     v[top:top + lq_patch_size, left:left + lq_patch_size, ...]
    #     for v in img_lqs
    # ]

    # crop corresponding gt patch

    if len(img_lqs) == 1:
        stable = img_lqs[0]
    return unstable, stable


def paired_random_crop(img_gts, img_lqs, gt_patch_size, gt_path, scale=1):
    """Paired random crop.

    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        gt_patch_size (int): GT patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    h_lq, w_lq, _ = img_lqs[0].shape
    h_gt, w_gt, _ = img_gts[0].shape
    lq_patch_size = gt_patch_size // scale

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(
            f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
            f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). '
                         f'Please remove {gt_path}.')

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    img_lqs = [
        v[top:top + lq_patch_size, left:left + lq_patch_size, ...]
        for v in img_lqs
    ]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    img_gts = [
        v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...]
        for v in img_gts
    ]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    return img_gts, img_lqs

def paired_random_crop_two(img_gts, img_lqs_1, img_lqs_2, gt_patch_size, gt_path, scale=1):
    """Paired random crop.

    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        gt_patch_size (int): GT patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs_1, list):
        img_lqs_1 = [img_lqs_1]

    h_lq, w_lq, _ = img_lqs_1[0].shape
    h_gt, w_gt, _ = img_gts[0].shape
    lq_patch_size = gt_patch_size // scale

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(
            f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
            f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). '
                         f'Please remove {gt_path}.')

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    img_lqs_1 = [
        v[top:top + lq_patch_size, left:left + lq_patch_size, ...]
        for v in img_lqs_1
    ]
    img_lqs_2 = [
        v[top:top + lq_patch_size, left:left + lq_patch_size, ...]
        for v in img_lqs_2
    ]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    img_gts = [
        v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...]
        for v in img_gts
    ]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs_1) == 1:
        img_lqs_1 = img_lqs_1[0]
    if len(img_lqs_2) == 1:
        img_lqs_2 = img_lqs_2[0]
    return img_gts, img_lqs_1, img_lqs_2


def augment(imgs, hflip=True, rotation=True, flows=None):
    """Augment: horizontal flips or rotate (0, 90, 180, 270 degrees).

    Use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray]: Image list to be augmented.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation or not. Default: True.
        flows (list[ndarray]: Flow list to be augmented.
            Dimension is (h, w, 2). Default: None.

    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.
    """
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _imflip_(img, direction='horizontal'):
        """Inplace flip an image horizontally or vertically.

        Args:
            img (ndarray): Image to be flipped.
            direction (str): The flip direction, either "horizontal" or
                "vertical" or "diagonal".

        Returns:
            ndarray: The flipped image (inplace).
        """
        assert direction in ['horizontal', 'vertical', 'diagonal']
        if direction == 'horizontal':
            return cv2.flip(img, 1, img)
        elif direction == 'vertical':
            return cv2.flip(img, 0, img)
        else:
            return cv2.flip(img, -1, img)

    def _augment(img):
        if hflip:
            _imflip_(img, 'horizontal')
        if vflip:
            _imflip_(img, 'vertical')
        if rot90:  # for (H W 3) image, H <-> W
            img = img.transpose(1, 0, 2)
        return img

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    def _augment_flow(flow):
        if hflip:
            _imflip_(flow, 'horizontal')
            flow[:, :, 0] *= -1
        if vflip:
            _imflip_(flow, 'vertical')
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        return imgs


def bgr2yuv420(imgs):
    def _totensor(img):
        # 获取输入图像的大小和通道数
        h, w, c = img.shape

        # BGR 到 YUV 的转换矩阵
        yuv_coef = np.array([[0.299, 0.587, 0.114], [-0.14713, -0.28886, 0.436], [0.615, -0.51498, -0.10001]])

        # 将 BGR 图像转换为 YUV 图像
        img_yuv = np.dot(img, yuv_coef.T)

        # 对 YUV 图像进行裁剪，使其大小为偶数
        h_half = h // 2 * 2
        w_half = w // 2 * 2
        img_yuv = img_yuv[:h_half, :w_half, :]

        # 将 YUV 图像分成三个通道
        y = img_yuv[:, :, 0]
        y = np.expand_dims(y, axis=2)

        y = torch.from_numpy(y.transpose(2, 0, 1))
        y = y.float()
        return y

    if isinstance(imgs, list):
        return [_totensor(img) for img in imgs]
    else:
        return _totensor(imgs)


def totensor(imgs, opt_bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        opt_bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, opt_bgr2rgb, float32):
        # print(img.shape)
        if img.shape[2] == 3 and opt_bgr2rgb:
            img = bgr2rgb(img)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, opt_bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, opt_bgr2rgb, float32)
