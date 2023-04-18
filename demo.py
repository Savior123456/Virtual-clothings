import os
import numpy as np
import torch

from PIL import Image as PILImage
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

import networks
from utils.transforms import BGR2RGB_transform
from utils.transforms import transform_parsing
import cv2

from collections import OrderedDict


def get_palette(num_cls):

    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


def multi_scale_testing(model, batch_input_im, crop_size=[473, 473], flip=False, multi_scales=[1]):
    flipped_idx = (15, 14, 17, 16, 19, 18)
    if len(batch_input_im.shape) > 4:
        batch_input_im = batch_input_im.squeeze()
    if len(batch_input_im.shape) == 3:
        batch_input_im = batch_input_im.unsqueeze(0)

    interp = torch.nn.Upsample(
        size=crop_size, mode='bilinear', align_corners=True)
    ms_outputs = []
    for s in multi_scales:
        interp_im = torch.nn.Upsample(
            scale_factor=s, mode='bilinear', align_corners=True)
        scaled_im = interp_im(batch_input_im)
        parsing_output = model(scaled_im)
        parsing_output = parsing_output[0][-1]
        output = parsing_output[0]
        if flip:
            flipped_output = parsing_output[1]
            flipped_output[14:20, :, :] = flipped_output[flipped_idx, :, :]
            output += flipped_output.flip(dims=[-1])
            output *= 0.5
        output = interp(output.unsqueeze(0))
        ms_outputs.append(output[0])
    ms_fused_parsing_output = torch.stack(ms_outputs)
    ms_fused_parsing_output = ms_fused_parsing_output.mean(0)
    ms_fused_parsing_output = ms_fused_parsing_output.permute(1, 2, 0)  # HWC
    parsing = torch.argmax(ms_fused_parsing_output, dim=2)
    parsing = parsing.data.cpu().numpy()
    ms_fused_parsing_output = ms_fused_parsing_output.data.cpu().numpy()
    return parsing, ms_fused_parsing_output


def _box2cs(box):
    x, y, w, h = box[:4]
    return _xywh2cs(x, y, w, h)


def _xywh2cs(x, y, w, h, aspect_ratio=1):
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5
    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w * 1.0, h * 1.0], dtype=np.float32)

    return center, scale


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale

    src_w = scale_tmp[0]
    dst_w = output_size[1]
    dst_h = output_size[0]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, (dst_w-1) * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [(dst_w-1) * 0.5, (dst_h-1) * 0.5]
    dst[1, :] = np.array([(dst_w-1) * 0.5, (dst_h-1) * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def main():
    cudnn.benchmark = True
    cudnn.enabled = True

    input_size = [473, 473]

    model = networks.init_model('resnet101', num_classes=20, pretrained=None)

    IMAGE_MEAN = model.mean
    IMAGE_STD = model.std
    INPUT_SPACE = model.input_space
    print('image mean: {}'.format(IMAGE_MEAN))
    print('image std: {}'.format(IMAGE_STD))
    print('input space:{}'.format(INPUT_SPACE))
    if INPUT_SPACE == 'BGR':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN,
                                 std=IMAGE_STD),

        ])
    if INPUT_SPACE == 'RGB':
        transform = transforms.Compose([
            transforms.ToTensor(),
            BGR2RGB_transform(),
            transforms.Normalize(mean=IMAGE_MEAN,
                                 std=IMAGE_STD),
        ])

    state_dict = torch.load(
        'F:\github\self-correction\exp-schp-201908261155-lip.pth')['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    # model.cuda()
    model.eval()

    sp_results_dir = os.path.join('./', 'sp_results')
    if not os.path.exists(sp_results_dir):
        os.makedirs(sp_results_dir)

    palette = get_palette(20)
    im_path = './image.jpg'
    im = cv2.imread(im_path, cv2.IMREAD_COLOR)
    h, w, _ = im.shape
    # Get person center and scale
    person_center, s = _box2cs([0, 0, w - 1, h - 1])
    r = 0
    trans = get_affine_transform(person_center, s, r, [473, 473])
    input = cv2.warpAffine(
        im,
        trans,
        (473, 473),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0))
    image = transform(input)
    image = image.squeeze()
    # img = img.cuda()
    parsing, logits = multi_scale_testing(
        model, image, crop_size=input_size)
    parsing_result = transform_parsing(
        parsing, person_center, s, w, h, input_size)
    parsing_result_path = os.path.join(sp_results_dir, 'test' + '.png')
    output_im = PILImage.fromarray(
        np.asarray(parsing_result, dtype=np.uint8))
    output_im.putpalette(palette)
    output_im.save(parsing_result_path)


if __name__ == '__main__':
    main()
