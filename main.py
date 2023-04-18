from body import Body
from hand import Hand
from UIFunctions import *
from utils.capnums import Camera
import time
import json
import torch
import sys
import os
from models import SegGenerator, ALIASGenerator, GMM
from torch import nn
import torchgeometry as tgm
from torch.nn import functional as F
import argparse
from PIL import Image
from torchvision import transforms
from PIL import ImageDraw
import math
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu
from PySide6.QtGui import QImage, QPixmap, QColor
from PySide6.QtCore import QTimer, QThread, Signal, QObject, QPoint, Qt
from ui.CustomMessageBox import MessageBox
from ui.home import Ui_MainWindow
from src import util
from src.model import bodypose_model, bodypose_25_model
from utils.transforms import BGR2RGB_transform
from utils.transforms import transform_parsing
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

#       res_picture  用于展示模型训练出来的结果
#       pre_video用于展示选中的人物照片
#       res_video用于展示选中衣服的照片

def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride) # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad

# transfer caffe model to pytorch which will match the layer name
def transfer(model, model_weights):
    transfered_model_weights = {}
    for weights_name in model.state_dict().keys():
        transfered_model_weights[weights_name] = model_weights['.'.join(weights_name.split('.')[1:])]
    return transfered_model_weights

# draw the body keypoint and lims
def draw_bodypose(canvas, candidate, subset):
    stickwidth = 4
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)
    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    # plt.imsave("preview.jpg", canvas[:, :, [2, 1, 0]])
    # plt.imshow(canvas[:, :, [2, 1, 0]])
    return canvas

def draw_handpose(canvas, all_hand_peaks, show_number=False):
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
    fig = Figure(figsize=plt.figaspect(canvas))

    fig.subplots_adjust(0, 0, 1, 1)
    fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
    bg = FigureCanvas(fig)
    ax = fig.subplots()
    ax.axis('off')
    ax.imshow(canvas)

    width, height = ax.figure.get_size_inches() * ax.figure.get_dpi()

    for peaks in all_hand_peaks:
        for ie, e in enumerate(edges):
            if np.sum(np.all(peaks[e], axis=1)==0)==0:
                x1, y1 = peaks[e[0]]
                x2, y2 = peaks[e[1]]
                ax.plot([x1, x2], [y1, y2], color=matplotlib.colors.hsv_to_rgb([ie/float(len(edges)), 1.0, 1.0]))

        for i, keyponit in enumerate(peaks):
            x, y = keyponit
            ax.plot(x, y, 'r.')
            if show_number:
                ax.text(x, y, str(i))
    bg.draw()
    canvas = np.fromstring(bg.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    return canvas

# image drawed by opencv is not good.
def draw_handpose_by_opencv(canvas, peaks, show_number=False):
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
    # cv2.rectangle(canvas, (x, y), (x+w, y+w), (0, 255, 0), 2, lineType=cv2.LINE_AA)
    # cv2.putText(canvas, 'left' if is_left else 'right', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    for ie, e in enumerate(edges):
        if np.sum(np.all(peaks[e], axis=1)==0)==0:
            x1, y1 = peaks[e[0]]
            x2, y2 = peaks[e[1]]
            cv2.line(canvas, (x1, y1), (x2, y2), matplotlib.colors.hsv_to_rgb([ie/float(len(edges)), 1.0, 1.0])*255, thickness=2)

    for i, keyponit in enumerate(peaks):
        x, y = keyponit
        cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)
        if show_number:
            cv2.putText(canvas, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), lineType=cv2.LINE_AA)
    return canvas

# detect hand according to body pose keypoints
# please refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/src/openpose/hand/handDetector.cpp
def handDetect(candidate, subset, oriImg):
    # right hand: wrist 4, elbow 3, shoulder 2
    # left hand: wrist 7, elbow 6, shoulder 5
    ratioWristElbow = 0.33
    detect_result = []
    image_height, image_width = oriImg.shape[0:2]
    for person in subset.astype(int):
        # if any of three not detected
        has_left = np.sum(person[[5, 6, 7]] == -1) == 0
        has_right = np.sum(person[[2, 3, 4]] == -1) == 0
        if not (has_left or has_right):
            continue
        hands = []
        #left hand
        if has_left:
            left_shoulder_index, left_elbow_index, left_wrist_index = person[[5, 6, 7]]
            x1, y1 = candidate[left_shoulder_index][:2]
            x2, y2 = candidate[left_elbow_index][:2]
            x3, y3 = candidate[left_wrist_index][:2]
            hands.append([x1, y1, x2, y2, x3, y3, True])
        # right hand
        if has_right:
            right_shoulder_index, right_elbow_index, right_wrist_index = person[[2, 3, 4]]
            x1, y1 = candidate[right_shoulder_index][:2]
            x2, y2 = candidate[right_elbow_index][:2]
            x3, y3 = candidate[right_wrist_index][:2]
            hands.append([x1, y1, x2, y2, x3, y3, False])

        for x1, y1, x2, y2, x3, y3, is_left in hands:
            # pos_hand = pos_wrist + ratio * (pos_wrist - pos_elbox) = (1 + ratio) * pos_wrist - ratio * pos_elbox
            # handRectangle.x = posePtr[wrist*3] + ratioWristElbow * (posePtr[wrist*3] - posePtr[elbow*3]);
            # handRectangle.y = posePtr[wrist*3+1] + ratioWristElbow * (posePtr[wrist*3+1] - posePtr[elbow*3+1]);
            # const auto distanceWristElbow = getDistance(poseKeypoints, person, wrist, elbow);
            # const auto distanceElbowShoulder = getDistance(poseKeypoints, person, elbow, shoulder);
            # handRectangle.width = 1.5f * fastMax(distanceWristElbow, 0.9f * distanceElbowShoulder);
            x = x3 + ratioWristElbow * (x3 - x2)
            y = y3 + ratioWristElbow * (y3 - y2)
            distanceWristElbow = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
            distanceElbowShoulder = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            width = 1.5 * max(distanceWristElbow, 0.9 * distanceElbowShoulder)
            # x-y refers to the center --> offset to topLeft point
            # handRectangle.x -= handRectangle.width / 2.f;
            # handRectangle.y -= handRectangle.height / 2.f;
            x -= width / 2
            y -= width / 2  # width = height
            # overflow the image
            if x < 0: x = 0
            if y < 0: y = 0
            width1 = width
            width2 = width
            if x + width > image_width: width1 = image_width - x
            if y + width > image_height: width2 = image_height - y
            width = min(width1, width2)
            # the max hand box value is 20 pixels
            if width >= 20:
                detect_result.append([int(x), int(y), int(width), is_left])

    '''
    return value: [[x, y, w, True if left hand else False]].
    width=height since the network require squared input.
    x, y is the coordinate of top left 
    '''
    return detect_result

# get max index of 2d array
def npmax(array):
    arrayindex = array.argmax(1)
    arrayvalue = array.max(1)
    i = arrayvalue.argmax()
    j = arrayindex[i]
    return i, j

def save_output(pred, d_dir, image_size):
    predict = pred
    predict = predict.squeeze()
    # predict = predict.cpu().data.numpy()
    im = Image.fromarray(predict).convert('RGB')
    imo = im.resize(image_size, resample=Image.BILINEAR)
    imo.save(d_dir + '.png')

def save_images(img_tensor, img_name, save_dir):
    tensor = (img_tensor.clone()+1)*0.5 * 255
    tensor = tensor.cpu().clamp(0,255)

    try:
        array = tensor.numpy().astype('uint8')
    except:
        array = tensor.detach().numpy().astype('uint8')

    if array.shape[0] == 1:
        array = array.squeeze(0)
    elif array.shape[0] == 3:
        array = array.swapaxes(0, 1).swapaxes(1, 2)

    im = Image.fromarray(array)
    im.save(os.path.join(save_dir, img_name), format='JPEG')
    return im

# make this suitable for video and image
class BasePredictor(QObject):
    main_pre_img = Signal(np.ndarray)  # raw image signal
    main_res_img = Signal(np.ndarray)  # test result signal
    main_status_msg = Signal(
        str
    )  # Detecting/pausing/stopping/testing complete/error reporting signal
    # main_fps = Signal(str)  # fps
    # main_labels = Signal(dict)  # Detected target results (number of each category)
    # main_progress = Signal(int)  # Completeness

    def __init__(self, model):
        QObject.__init__(self)
        # GUI args
        self.used_model_name = "Virtual Try On"  # The detection model name to use
        self.new_model_name = None  # Models that change in real time
        self.source = ""  # input source
        self.stop_dtc = False  # Terminate
        self.continue_dtc = True  # pause
        self.save_res = False  # Save test results
        self.speed_thres = 10  # delay, ms
        self.labels_dict = {}  # return a dictionary of results
        self.progress_value = 0  # progress bar

        self.img_name = "xxx"



        # Usable if setup is done
        # model.eval()
        self.model = model
        self.imgsz = None
        self.device = None
        self.dataset = None
        self.vid_path, self.vid_writer = None, None
        self.data_path = None
        self.source_type = None
        self.batch = None

    def preprocess(self):
        raise NotImplementedError

    def postprocess(self, img):
        raise NotImplementedError

    def write_results(self, result):
        raise NotImplementedError

    @torch.no_grad()
    def run(self):
        raise NotImplementedError

def gen_noise(shape):
    noise = np.zeros(shape, dtype=np.uint8)
    ### noise
    noise = cv2.randn(noise, 0, 255)
    noise = np.asarray(noise / 255, dtype=np.uint8)
    noise = torch.tensor(noise, dtype=torch.float32)
    return noise

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_height', type=int, default=1024)
    parser.add_argument('--load_width', type=int, default=768)

    # common
    parser.add_argument('--semantic_nc', type=int, default=13, help='# of human-parsing map classes')
    parser.add_argument('--init_type', choices=['normal', 'xavier', 'xavier_uniform', 'kaiming', 'orthogonal', 'none'], default='xavier')
    parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')

    # for GMM
    parser.add_argument('--grid_size', type=int, default=5)

    # for ALIASGenerator
    parser.add_argument('--norm_G', type=str, default='spectralaliasinstance')
    parser.add_argument('--ngf', type=int, default=64, help='# of generator filters in the first conv layer')
    parser.add_argument('--num_upsampling_layers', choices=['normal', 'more', 'most'], default='most',
                        help='If \'more\', add upsampling layer between the two middle resnet blocks. '
                             'If \'most\', also add one more (upsampling + resnet) layer at the end of the generator.')

    opt = parser.parse_args()
    return opt



class Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class VirtualClothingPredictor(BasePredictor):
    time_msg = Signal(str)
    def __init__(self):
        super().__init__(None)
        self.used_model_name = ""  # The detection model name to use
        self.new_model_name = ""  # Models that change in real time
        self.source = ""  # input source
        self.cloth = ""
        self.pose = ""
        # self.parse_img = "F:\\github\GUI\\imges\\image_parse.png"
        self.parse_img = ""
        self.open_json = "F:\\github\\test\\openpose-json\\08909_00_keypoints.json"
        self.save_res = False  # Save test results
        self.save_dir = "./results/"
        self.speed_thres = 10  # delay, ms
        self.labels_dict = {}  # return a dictionary of results

        # Usable if setup is done
        self.imgsz = (512, 512)
        self.device = None
        self.dataset = None
        self.vid_path, self.vid_writer = None, None
        self.annotator = None
        self.data_path = None
        self.source_type = None
        self.batch = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_fp16 = False
        self.pre_time = None
        self.infer_time = None
        self.post_time = '/'
        self.checkpoint_dir = './checkpoints'
        self.checkpoints = ['seg_final.pth','gmm_final.pth','alias_final.pth']




        self.opt = get_opt()

        # self.seg = SegGenerator(input_nc=13 + 8, output_nc=13)
        # self.gmm = GMM(inputA_nc=7, inputB_nc=3)
        # self.alias = ALIASGenerator(input_nc=9)

        self.seg = SegGenerator(self.opt, input_nc=self.opt.semantic_nc + 8, output_nc=self.opt.semantic_nc)
        self.gmm = GMM(self.opt, inputA_nc=7, inputB_nc=3)
        self.opt.semantic_nc = 7
        self.alias = ALIASGenerator(self.opt, input_nc=9)
        self.opt.semantic_nc = 13

        self.load_checkpoint(self.seg, os.path.join(self.checkpoint_dir, self.checkpoints[0]))
        self.load_checkpoint(self.gmm, os.path.join(self.checkpoint_dir, self.checkpoints[1]))
        self.load_checkpoint(self.alias, os.path.join(self.checkpoint_dir, self.checkpoints[2]))

        self.seg.cuda().eval()
        self.gmm.cuda().eval()
        self.alias.cuda().eval()
        self.up = nn.Upsample(size=(1024, 768), mode='bilinear')
        self.gauss = tgm.image.GaussianBlur((15, 15), (3, 3))
        self.gauss.cuda()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def load_checkpoint(self, model, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            raise ValueError("'{}' is not a valid checkpoint path".format(checkpoint_path))
        model.load_state_dict(torch.load(checkpoint_path))
    def get_parse_agnostic(self, parse, pose_data):
        parse_array = np.array(parse)
        parse_upper = ((parse_array == 5).astype(np.float32) +
                       (parse_array == 6).astype(np.float32) +
                       (parse_array == 7).astype(np.float32))
        parse_neck = (parse_array == 10).astype(np.float32)

        r = 10
        agnostic = parse.copy()

        # mask arms
        for parse_id, pose_ids in [(14, [2, 5, 6, 7]), (15, [5, 2, 3, 4])]:
            mask_arm = Image.new('L', (self.opt.load_width, self.opt.load_height), 'black')
            mask_arm_draw = ImageDraw.Draw(mask_arm)
            i_prev = pose_ids[0]
            for i in pose_ids[1:]:
                if (pose_data[i_prev, 0] == 0.0 and pose_data[i_prev, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                    continue
                mask_arm_draw.line([tuple(pose_data[j]) for j in [i_prev, i]], 'white', width=r*10)
                pointx, pointy = pose_data[i]
                radius = r*4 if i == pose_ids[-1] else r*15
                mask_arm_draw.ellipse((pointx-radius, pointy-radius, pointx+radius, pointy+radius), 'white', 'white')
                i_prev = i
            parse_arm = (np.array(mask_arm) / 255) * (parse_array == parse_id).astype(np.float32)
            agnostic.paste(0, None, Image.fromarray(np.uint8(parse_arm * 255), 'L'))

        # mask torso & neck
        agnostic.paste(0, None, Image.fromarray(np.uint8(parse_upper * 255), 'L'))
        agnostic.paste(0, None, Image.fromarray(np.uint8(parse_neck * 255), 'L'))

        return agnostic

    def get_img_agnostic(self, img, parse, pose_data):
        parse_array = np.array(parse)
        parse_head = ((parse_array == 4).astype(np.float32) +
                      (parse_array == 13).astype(np.float32))
        parse_lower = ((parse_array == 9).astype(np.float32) +
                       (parse_array == 12).astype(np.float32) +
                       (parse_array == 16).astype(np.float32) +
                       (parse_array == 17).astype(np.float32) +
                       (parse_array == 18).astype(np.float32) +
                       (parse_array == 19).astype(np.float32))

        r = 20
        agnostic = img.copy()
        agnostic_draw = ImageDraw.Draw(agnostic)

        length_a = np.linalg.norm(pose_data[5] - pose_data[2])
        length_b = np.linalg.norm(pose_data[12] - pose_data[9])
        point = (pose_data[9] + pose_data[12]) / 2
        pose_data[9] = point + (pose_data[9] - point) / length_b * length_a
        pose_data[12] = point + (pose_data[12] - point) / length_b * length_a

        # mask arms
        agnostic_draw.line([tuple(pose_data[i]) for i in [2, 5]], 'gray', width=r*10)
        for i in [2, 5]:
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')
        for i in [3, 4, 6, 7]:
            if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                continue
            agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'gray', width=r*10)
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')

        # mask torso
        for i in [9, 12]:
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx-r*3, pointy-r*6, pointx+r*3, pointy+r*6), 'gray', 'gray')
        agnostic_draw.line([tuple(pose_data[i]) for i in [2, 9]], 'gray', width=r*6)
        agnostic_draw.line([tuple(pose_data[i]) for i in [5, 12]], 'gray', width=r*6)
        agnostic_draw.line([tuple(pose_data[i]) for i in [9, 12]], 'gray', width=r*12)
        agnostic_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 12, 9]], 'gray', 'gray')

        # mask neck
        pointx, pointy = pose_data[1]
        agnostic_draw.rectangle((pointx-r*7, pointy-r*7, pointx+r*7, pointy+r*7), 'gray', 'gray')
        agnostic.paste(img, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
        agnostic.paste(img, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))

        return agnostic

    def preprocess(self):
        st = time.time()
        c = Image.open(self.cloth).convert('RGB')
        c = transforms.Resize(self.opt.load_width, interpolation=2)(c)
        # cm = Image.open(self.cloth_mask)
        # cm = transforms.Resize(self.opt.load_width, interpolation=0)(cm)
        cm = np.array(c).mean(2)
        cm = np.where(cm < 240, 255, 0)[None] / 255

        cm = torch.Tensor(cm)
        c = self.transform(c)  # [-1,1]
        # cm_array = np.array(cm)
        # cm_array = (cm_array >= 128).astype(np.float32)
        # cm = torch.from_numpy(cm_array)  # [0,1]

        # body_estimation = Body('models/body_pose_model.pth')
        # hand_estimation = Hand('models/hand_pose_model.pth')
        #
        # test_image = self.source
        # oriImg = cv2.imread(test_image)  # B,G,R order
        # candidate, subset = body_estimation(oriImg)
        # canvas = np.zeros_like(oriImg)
        # canvas = draw_bodypose(canvas, candidate, subset)
        # # detect hand
        # hands_list = handDetect(candidate, subset, oriImg)
        #
        # all_hand_peaks = []
        # for x, y, w, is_left in hands_list:
        #     # cv2.rectangle(canvas, (x, y), (x+w, y+w), (0, 255, 0), 2, lineType=cv2.LINE_AA)
        #     # cv2.putText(canvas, 'left' if is_left else 'right', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        #
        #     # if is_left:
        #     # plt.imshow(oriImg[y:y+w, x:x+w, :][:, :, [2, 1, 0]])
        #     # plt.show()
        #     peaks = hand_estimation(oriImg[y:y + w, x:x + w, :])
        #     peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0] + x)
        #     peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1] + y)
        #     # else:
        #     #     peaks = hand_estimation(cv2.flip(oriImg[y:y+w, x:x+w, :], 1))
        #     #     peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], w-peaks[:, 0]-1+x)
        #     #     peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
        #     #     print(peaks)
        #     all_hand_peaks.append(peaks)
        #
        # canvas = draw_handpose(canvas, all_hand_peaks)
        #
        # plt.imshow(canvas[:, :, [2, 1, 0]])
        # plt.axis('off')
        # plt.show()

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

        sp_results_dir = "F:\github\GUI\sp_results"

        # if not os.path.exists(sp_results_dir):
        #     os.makedirs(sp_results_dir)

        palette = get_palette(20)
        im_path = self.source
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
        parsing_result_path = os.path.join(sp_results_dir, 'test03' + '.png')
        output_im = PILImage.fromarray(
            np.asarray(parsing_result, dtype=np.uint8))
        output_im.putpalette(palette)
        output_im.save(parsing_result_path)
        self.parse_img = parsing_result_path

        model_type = 'body25'  # 'coco'  #
        model_path = './models/pose_iter_584000.caffemodel.pt'
        body_estimation = Body(model_path, model_type)
        hand_estimation = Hand('models/hand_pose_model.pth')

        test_image_path = self.source
        oriImg = cv2.imread(test_image_path)  # B,G,R order
        candidate, subset = body_estimation(oriImg)
        canvas = np.zeros_like(oriImg)
        canvas,array_keypoints = util.draw_bodypose(canvas, candidate, subset, model_type)
        # detect hand
        hands_list = util.handDetect(candidate, subset, oriImg)

        all_hand_peaks = []
        for x, y, w, is_left in hands_list:
            # cv2.rectangle(canvas, (x, y), (x+w, y+w), (0, 255, 0), 2, lineType=cv2.LINE_AA)
            # cv2.putText(canvas, 'left' if is_left else 'right', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # if is_left:
            # plt.imshow(oriImg[y:y+w, x:x+w, :][:, :, [2, 1, 0]])
            # plt.show()
            peaks = hand_estimation(oriImg[y:y + w, x:x + w, :])
            peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0] + x)
            peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1] + y)
            # else:
            #     peaks = hand_estimation(cv2.flip(oriImg[y:y+w, x:x+w, :], 1))
            #     peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], w-peaks[:, 0]-1+x)
            #     peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
            #     print(peaks)
            all_hand_peaks.append(peaks)


        canvas = util.draw_handpose(canvas, all_hand_peaks)
        # canvas = util.draw_bodypose(canvas, candidate, subset, model_type)
        result_name = 'result_pose.jpg'
        cv2.imwrite(result_name, canvas)
        img_pose = Image.open(result_name)
        # img_pose = Image.fromarray(canvas)
        pic_new = img_pose.resize((768, 1024), Image.ANTIALIAS)
        pic_new.save(result_name)
        # load pose image
        pose_rgb = Image.open(result_name)
        # print(q.size)
        # pose_rgb = Image.open(canvas)
        # pose_rgb = Image.fromarray(canvas)
        # print(pose_rgb.size)
        # print(pose_rgb.shape)
        # pose_rgb = canvas
        pose_rgb = transforms.Resize((self.opt.load_height,self.opt.load_width), interpolation=2)(pose_rgb)
        pose_rgb = self.transform(pose_rgb)  # [-1,1]

        # plt.imshow(pose_rgb[:, :, [2, 1, 0]])
        # plt.axis('off')
        # plt.show()

        # [349.091,147.821,0.922318,400.138,317.836,0.770555,292.373,329.363,
        # 0.741053,303.693,513.417,0.217837,0,0,0,513.438,303.726,0.68234,
        # 527.623,547.373,0.808538,394.255,720.279,0.750887,351.821,711.821,
        # 0.523726,272.552,697.645,0.468322,266.87,995.149,0.224299,0,0,0,428.392,
        # 723.103,0.503569,473.774,1000.79,0.230145,0,0,0,326.226,125.155,
        # 0.960687,374.579,113.822,0.882049,306.453,147.828,0.331686,425.595,122.235,
        # 0.884945,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

        # pose_data = np.array([349.091,147.821,0.922318,400.138,317.836,0.770555,292.373,329.363,0.741053,303.693,513.417,0.217837,0,0,0,513.438,303.726,0.68234,527.623,547.373,0.808538,394.255,720.279,0.750887,351.821,711.821,0.523726,272.552,697.645,0.468322,266.87,995.149,0.224299,0,0,0,428.392,723.103,0.503569,473.774,1000.79,0.230145,0,0,0,326.226,125.155,0.960687,374.579,113.822,0.882049,306.453,147.828,0.331686,425.595,122.235,0.884945,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        # pose_data = pose_data.reshape((-1, 3))[:, :2]
        pose_data = np.array(array_keypoints)
        pose_data = pose_data.reshape((-1, 3))[:, :2]
        # print(pose_data)
        # # with open(self.open_json, 'r') as f:
        # #     pose_label = json.load(f)
        # #     pose_data = pose_label['people'][0]['pose_keypoints_2d']
        # #     pose_data = np.array(pose_data)
        # #     pose_data = pose_data.reshape((-1, 3))[:, :2]
        # print(pose_data)
        # load parsing image
        parse = Image.open(self.parse_img)
        parse = transforms.Resize(self.opt.load_width, interpolation=0)(parse)
        parse_agnostic = self.get_parse_agnostic(parse, pose_data)
        parse_agnostic = torch.from_numpy(np.array(parse_agnostic)[None]).long()

        labels = {
            0: ['background', [0, 10]],
            1: ['hair', [1, 2]],
            2: ['face', [4, 13]],
            3: ['upper', [5, 6, 7]],
            4: ['bottom', [9, 12]],
            5: ['left_arm', [14]],
            6: ['right_arm', [15]],
            7: ['left_leg', [16]],
            8: ['right_leg', [17]],
            9: ['left_shoe', [18]],
            10: ['right_shoe', [19]],
            11: ['socks', [8]],
            12: ['noise', [3, 11]],
        }
        parse_agnostic_map = torch.zeros(20, self.opt.load_height, self.opt.load_width, dtype=torch.float)
        parse_agnostic_map.scatter_(0, parse_agnostic, 1.0)
        new_parse_agnostic_map = torch.zeros(self.opt.semantic_nc, self.opt.load_height, self.opt.load_width, dtype=torch.float)
        for i in range(len(labels)):
            for label in labels[i][1]:
                new_parse_agnostic_map[i] += parse_agnostic_map[label]

        # load person image
        img = Image.open(self.source)
        img = transforms.Resize(self.opt.load_width, interpolation=2)(img)
        img_agnostic = self.get_img_agnostic(img, parse, pose_data)
        img = self.transform(img)
        img_agnostic = self.transform(img_agnostic)  # [-1,1]

        result = {
            'img': img,
            'img_agnostic': img_agnostic,
            'parse_agnostic': new_parse_agnostic_map,
            'pose': pose_rgb,
            'cloth': c,
            'cloth_mask': cm,
        }
        self.pre_time = time.time() - st
        return result

    def postprocess(self, img):
        return img

    def write_results(self, result):
        if torch.cuda.is_available():
            result = result.cpu()
        img_np = result.numpy()
        file_name = self.source.split(os.sep)[-1]
        save_path = os.path.join(self.save_dir, file_name)
        cv2.imwrite(save_path, img_np)

    @torch.no_grad()
    def run(self):
        if not self.save_dir:
            self.save_dir = os.sep.join(self.source.split(os.sep)[:-1])
        os.makedirs(self.save_dir, exist_ok=True)




        inputs = self.preprocess()
        st = time.time()
        img_agnostic = inputs['img_agnostic'].cuda()[None]
        parse_agnostic = inputs['parse_agnostic'].cuda()[None]
        pose = inputs['pose'].cuda()[None]
        c = inputs['cloth'].cuda()[None]
        cm = inputs['cloth_mask'].cuda()[None]



        # Part 1. Segmentation generation
        parse_agnostic_down = F.interpolate(parse_agnostic, size=(256, 192), mode='bilinear')
        pose_down = F.interpolate(pose, size=(256, 192), mode='bilinear')
        print(cm.shape)
        c_masked_down = F.interpolate(c * cm, size=(256, 192), mode='bilinear')
        cm_down = F.interpolate(cm, size=(256, 192), mode='bilinear')
        seg_input = torch.cat(
            (cm_down, c_masked_down, parse_agnostic_down, pose_down, gen_noise(cm_down.size()).cuda()), dim=1)

        parse_pred_down = self.seg(seg_input)
        parse_pred = self.gauss(self.up(parse_pred_down))
        parse_pred = parse_pred.argmax(dim=1)[:, None]

        parse_old = torch.zeros(parse_pred.size(0), 13, 1024, 768, dtype=torch.float).cuda()
        parse_old.scatter_(1, parse_pred, 1.0)

        labels = {
            0: ['background', [0]],
            1: ['paste', [2, 4, 7, 8, 9, 10, 11]],
            2: ['upper', [3]],
            3: ['hair', [1]],
            4: ['left_arm', [5]],
            5: ['right_arm', [6]],
            6: ['noise', [12]]
        }
        parse = torch.zeros(parse_pred.size(0), 7, 1024, 768, dtype=torch.float).cuda()
        for j in range(len(labels)):
            for label in labels[j][1]:
                parse[:, j] += parse_old[:, label]

        # Part 2. Clothes Deformation
        agnostic_gmm = F.interpolate(img_agnostic, size=(256, 192), mode='nearest')
        parse_cloth_gmm = F.interpolate(parse[:, 2:3], size=(256, 192), mode='nearest')
        pose_gmm = F.interpolate(pose, size=(256, 192), mode='nearest')
        c_gmm = F.interpolate(c, size=(256, 192), mode='nearest')
        gmm_input = torch.cat((parse_cloth_gmm, pose_gmm, agnostic_gmm), dim=1)

        _, warped_grid = self.gmm(gmm_input, c_gmm)
        warped_c = F.grid_sample(c, warped_grid, padding_mode='border')
        warped_cm = F.grid_sample(cm, warped_grid, padding_mode='border')

        # Part 3. Try-on synthesis
        misalign_mask = parse[:, 2:3] - warped_cm
        misalign_mask[misalign_mask < 0.0] = 0.0
        parse_div = torch.cat((parse, misalign_mask), dim=1)
        parse_div[:, 2:3] -= misalign_mask

        output = self.alias(torch.cat((img_agnostic, pose, warped_c), dim=1), parse, parse_div, misalign_mask)
        im = save_images(output[0],'xxx.jpg', self.save_dir)


        self.infer_time = time.time() - st
        return  im

        # if self.save_res:
        #     self.write_results(out)


class MainWindow(QMainWindow, Ui_MainWindow):
    begin_sgl = (
        Signal()
    )  # The main window sends an execution signal to the yolo instance

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        # basic interface
        self.setupUi(self)
        self.setAttribute(Qt.WA_TranslucentBackground)  # rounded transparent
        self.setWindowFlags(
            Qt.FramelessWindowHint
        )  # Set window flag: hide window borders
        UIFuncitons.uiDefinitions(self)
        # Show module shadows
        UIFuncitons.shadow_style(self, self.Class_QF, QColor(162, 129, 247))
        UIFuncitons.shadow_style(self, self.Target_QF, QColor(251, 157, 139))
        UIFuncitons.shadow_style(self, self.Fps_QF, QColor(170, 128, 213))
        UIFuncitons.shadow_style(self, self.Model_QF, QColor(64, 186, 193))

        # read model folder
        self.pt_list = os.listdir("./models")
        self.pt_list = [file for file in self.pt_list if file.endswith(".pt")]
        self.pt_list.sort(
            key=lambda x: os.path.getsize("./models/" + x)
        )  # sort by file size
        self.model_box.clear()
        self.model_box.addItems(self.pt_list)
        # self.Qtimer_ModelBox = QTimer(
        #     self
        # )  # Timer: Monitor model file changes every 2 seconds
        # self.Qtimer_ModelBox.timeout.connect(self.ModelBoxRefre)
        # self.Qtimer_ModelBox.start(2000)

        # thread
        self.predictor = VirtualClothingPredictor()  # Create a Yolo instance
        self.select_model = self.model_box.currentText()  # default model
        self.predictor.new_model_name = "./models/%s" % self.select_model
        self.thread = QThread()  # Create yolo thread
        self.predictor.main_pre_img.connect(
            lambda x: self.show_image(x, self.pre_video)
        )
        self.predictor.main_res_img.connect(
            lambda x: self.show_image(x, self.res_video)
        )
        self.predictor.main_status_msg.connect(lambda x: self.show_status(x))
        # self.predictor.main_fps.connect(lambda x: self.fps_label.setText(x))
        # self.yolo_predict.yolo2main_labels.connect(self.show_labels)
        # self.predictor.main_class_num.connect(
        #     lambda x: self.Class_num.setText(str(x))
        # )
        # self.predictor.main_target_num.connect(
        #     lambda x: self.Target_num.setText(str(x))
        # )
        # self.predictor.main_progress.connect(lambda x: self.progress_bar.setValue(x))
        # self.begin_sgl.connect(self.run_button_run)
        self.pre_time = self.predictor.pre_time

        self.end_time = self.predictor.post_time

        self.infer_time = self.predictor.infer_time


        self.Target_num.setText(self.infer_time)

        self.fps_label.setText(self.end_time)

        self.Class_num.setText(self.pre_time)

        # self.Model_name.setText(self.predictor.used_model_name)

        self.predictor.moveToThread(self.thread)


        # Model parameters
        self.model_box.currentTextChanged.connect(self.change_model)
        # self.iou_spinbox.valueChanged.connect(
        #     lambda x: self.change_val(x, "iou_spinbox")
        # )  # iou box
        # self.iou_slider.valueChanged.connect(
        #     lambda x: self.change_val(x, "iou_slider")
        # )  # iou scroll bar
        # self.conf_spinbox.valueChanged.connect(
        #     lambda x: self.change_val(x, "conf_spinbox")
        # )  # conf box
        # self.conf_slider.valueChanged.connect(
        #     lambda x: self.change_val(x, "conf_slider")
        # )  # conf scroll bar
        # self.speed_spinbox.valueChanged.connect(
        #     lambda x: self.change_val(x, "speed_spinbox")
        # )  # speed box
        # self.speed_slider.valueChanged.connect(
        #     lambda x: self.change_val(x, "speed_slider")
        # )  # speed scroll bar

        # Prompt window initialization
        # self.Class_num.setText("--")
        # self.Target_num.setText("--")
        # self.fps_label.setText("--")
        # self.Model_name.setText("Virtual Try On")


        # Select detection source
        self.src_file_button.clicked.connect(self.select_image)  # select local file
        self.src_cloth_button.clicked.connect(self.select_cloth_image)  # select local file
        # self.src_cam_button.clicked.connect(self.show_status("The function has not yet been implemented."))#chose_cam
        # self.src_rtsp_button.clicked.connect(self.show_status("The function has not yet been implemented."))#chose_rtsp

        # start testing button
        # self.run_button.clicked.connect(self.run_or_continue)  # pause/start
        self.stop_button.clicked.connect(self.stop)  # termination

        # Other function buttons
        self.save_res_button.toggled.connect(self.is_save_res)  # save image option
        #self.save_txt_button
        #self.save_txt_button.toggled.connect(self.is_save_txt)  # Save label option
        self.ToggleBotton.clicked.connect(
            lambda: UIFuncitons.toggleMenu(self, True)
        )  # left navigation button
        self.settings_button.clicked.connect(
            lambda: UIFuncitons.settingBox(self, True)
        )  # top right settings button


        #绑定按钮选择系统图片的事件
        self.selcect_person_pic_button.clicked.connect(self.select_image)
        self.selcect_close_pic_button.clicked.connect(self.select_cloth_image)

        self.run_button.clicked.connect(self.run_button_run)

        # pixmap1 = QPixmap(self.predictor.source)
        # self.pre_video.setPixmap(pixmap1.scaled(500, 400, Qt.KeepAspectRatio))
        #
        # pixmap = QPixmap(self.predictor.cloth)
        # self.res_video.setPixmap(pixmap.scaled(500, 400, Qt.KeepAspectRatio))
        
        # initialization
        self.load_config()


    
    def run_button_run(self):
        print("这里加入需要run的代码")
        #file_name = QFileDialog.getOpenFileName(self,"open file dialog","F:\github\GUI\img","Images (*.png *.xpm *.jpg *.bmp *.gif *.webp)")  
        im = self.predictor.run()

        self.Model_name.setText("Virtual Try On")
        self.pre_time = self.predictor.pre_time

        self.end_time = self.predictor.post_time

        self.infer_time = self.predictor.infer_time

        self.Target_num.setText(f'{self.infer_time:.2f}')

        # self.fps_label.setText(f'{self.end_time:.2f}')

        self.Class_num.setText(f'{self.pre_time:.2f}')

        pixmap = QPixmap(os.path.join(self.predictor.save_dir, "xxx.jpg"))
        self.res_picture.setPixmap(pixmap.scaled(self.res_picture.width(), self.res_picture.height(), Qt.KeepAspectRatio))

        
        


    
    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(Home, '选择图片', '', 'Images (*.png *.xpm *.jpg *.bmp *.gif *.webp)')
        if file_path:
            print(file_path)
            self.predictor.source = file_path
            pixmap = QPixmap(file_path)
            self.pre_video.setPixmap(pixmap.scaled(500,400, Qt.KeepAspectRatio))

            #pre_video.setPixmap(pixmap.scaled(self.pre_video.width(), self.pre_video.height(), Qt.KeepAspectRatio))

    def select_cloth_image(self):
        file_path, _ = QFileDialog.getOpenFileName(Home, '选择图片', '', 'Images (*.png *.xpm *.jpg *.bmp *.gif *.webp)')
        if file_path:
            print(file_path)
            self.predictor.cloth = file_path
            pixmap = QPixmap(file_path)
            self.res_video.setPixmap(pixmap.scaled(500,400, Qt.KeepAspectRatio))


    # The main window displays the original image and detection results
    @staticmethod
    def show_image(img_src, label):
        try:
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            # keep the original data ratio
            if iw / w > ih / h:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(
                frame.data,
                frame.shape[1],
                frame.shape[0],
                frame.shape[2] * frame.shape[1],
                QImage.Format_RGB888,
            )
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))

    # Control start/pause
    # def run_or_continue(self):
    #     if self.predictor.source == "":
    #         self.show_status(
    #             "Please select a video source before starting detection..."
    #         )
    #         self.run_button.setChecked(False)
    #     else:
    #         self.predictor.stop_dtc = False
    #         if self.run_button.isChecked():
    #             self.run_button.setChecked(True)  # start button
    #             # self.save_txt_button.setEnabled(
    #             #     False
    #             # )  # It is forbidden to check and save after starting the detection
    #             # self.save_res_button.setEnabled(False)
    #             self.show_status("Detecting...")
    #             self.predictor.continue_dtc = True  # Control whether Yolo is paused
    #             if not self.thread.isRunning():
    #                 self.thread.start()
    #                 self.begin_sgl.emit()
    #
    #         else:
    #             self.predictor.continue_dtc = False
    #             self.show_status("Pause...")
    #             self.run_button.setChecked(False)  # start button

    # bottom status bar information
    def show_status(self, msg):
        self.status_bar.setText(msg)
        if msg == "Detection completed" or msg == "检测完成":
            self.save_res_button.setEnabled(True)
            self.save_txt_button.setEnabled(True)
            self.run_button.setChecked(False)
            self.progress_bar.setValue(0)
            if self.thread.isRunning():
                self.thread.quit()  # end process
        elif msg == "Detection terminated!" or msg == "检测终止":
            self.save_res_button.setEnabled(True)
            self.save_txt_button.setEnabled(True)
            self.run_button.setChecked(False)
            self.progress_bar.setValue(0)
            if self.thread.isRunning():
                self.thread.quit()  # end process
            self.pre_video.clear()  # clear image display
            self.res_video.clear()
            self.Class_num.setText("--")
            self.Target_num.setText("--")
            self.fps_label.setText("--")

    # select local file
    def open_src_file(self):
        config_file = "config/fold.json"
        config = json.load(open(config_file, "r", encoding="utf-8"))
        open_fold = config["open_fold"]
        if not os.path.exists(open_fold):
            open_fold = os.getcwd()
        name, _ = QFileDialog.getOpenFileName(
            self,
            "image",
            open_fold,
            "Pic File(*.jpg *.png)",
        )
        if name:
            self.predictor.source = name
            self.show_status("Load File：{}".format(os.path.basename(name)))
            config["open_fold"] = os.path.dirname(name)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, "w", encoding="utf-8") as f:
                f.write(config_json)
            name, _ = QFileDialog.getOpenFileName(
                self,
                "image",
                open_fold,
                "Pic File(*.jpg *.png)",
            )
            if name:
                self.predictor.cloth = name
            else:
                self.predictor.source = ""
            self.stop()

    # Select camera source----  have one bug
    def chose_cam(self):
        try:
            self.stop()
            MessageBox(
                self.close_button,
                title="Note",
                text="loading camera...",
                time=2000,
                auto=True,
            ).exec()
            # get the number of local cameras
            _, cams = Camera().get_cam_num()
            popMenu = QMenu()
            popMenu.setFixedWidth(self.src_cam_button.width())
            popMenu.setStyleSheet(
                """
                                            QMenu {
                                            font-size: 16px;
                                            font-family: "Microsoft YaHei UI";
                                            font-weight: light;
                                            color:white;
                                            padding-left: 5px;
                                            padding-right: 5px;
                                            padding-top: 4px;
                                            padding-bottom: 4px;
                                            border-style: solid;
                                            border-width: 0px;
                                            border-color: rgba(255, 255, 255, 255);
                                            border-radius: 3px;
                                            background-color: rgba(200, 200, 200,50);}
                                            """
            )

            for cam in cams:
                exec("action_%s = QAction('%s')" % (cam, cam))
                exec("popMenu.addAction(action_%s)" % cam)

            x = self.src_cam_button.mapToGlobal(self.src_cam_button.pos()).x()
            y = self.src_cam_button.mapToGlobal(self.src_cam_button.pos()).y()
            y = y + self.src_cam_button.frameGeometry().height()
            pos = QPoint(x, y)
            action = popMenu.exec(pos)
            if action:
                self.predictor.source = action.text()
                self.show_status("Loading camera：{}".format(action.text()))

        except Exception as e:
            self.show_status("%s" % e)

    # # select network source
    # def chose_rtsp(self):
    #     self.rtsp_window = Window()
    #     config_file = "config/ip.json"
    #     if not os.path.exists(config_file):
    #         ip = "rtsp://admin:admin888@192.168.1.2:555"
    #         new_config = {"ip": ip}
    #         new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
    #         with open(config_file, "w", encoding="utf-8") as f:
    #             f.write(new_json)
    #     else:
    #         config = json.load(open(config_file, "r", encoding="utf-8"))
    #         ip = config["ip"]
    #     self.rtsp_window.rtspEdit.setText(ip)
    #     self.rtsp_window.show()
    #     self.rtsp_window.rtspButton.clicked.connect(
    #         lambda: self.load_rtsp(self.rtsp_window.rtspEdit.text())
    #     )

    # load network sources
    def load_rtsp(self, ip):
        try:
            self.stop()
            MessageBox(
                self.close_button, title="提示", text="加载 rtsp...", time=1000, auto=True
            ).exec()
            self.predictor.source = ip
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open("config/ip.json", "w", encoding="utf-8") as f:
                f.write(new_json)
            self.show_status("Loading rtsp：{}".format(ip))
            self.rtsp_window.close()
        except Exception as e:
            self.show_status("%s" % e)

    # Save test result button--picture/video
    def is_save_res(self):
        if self.save_res_button.checkState() == Qt.CheckState.Unchecked:
            self.show_status("NOTE: Run image results are not saved.")
            self.predictor.save_res = False
        elif self.save_res_button.checkState() == Qt.CheckState.Checked:
            self.show_status("NOTE: Run image results will be saved.")
            self.predictor.save_res = True
            """ file_dir = QFileDialog.getExistingDirectory(self,"选择一个目录","./",QFileDialog.ShowDirsOnly)
            file_name = file_dir + "/result.jpg"
            self.predictor.save_dir = file_name
            print(file_name) """  #选择一个系统目录，并且添加预测文件的保存路径

    # Save test result button -- label (txt)
    def is_save_txt(self):
        if self.save_txt_button.checkState() == Qt.CheckState.Unchecked:
            self.show_status("NOTE: Labels results are not saved.")
            self.predictor.save_txt = False
        elif self.save_txt_button.checkState() == Qt.CheckState.Checked:
            self.show_status("NOTE: Labels results will be saved.")
            self.predictor.save_txt = True

    # Configuration initialization  ~~~wait to change~~~
    def load_config(self):
        config_file = "config/setting.json"
        if not os.path.exists(config_file):
            iou = 0.26
            conf = 0.33
            rate = 10
            save_res = 0
            save_txt = 0
            new_config = {
                "iou": iou,
                "conf": conf,
                "rate": rate,
                "save_res": save_res,
                "save_txt": save_txt,
            }
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, "w", encoding="utf-8") as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, "r", encoding="utf-8"))
            if len(config) != 5:
                iou = 0.26
                conf = 0.33
                rate = 10
                save_res = 0
                save_txt = 0
            else:
                iou = config["iou"]
                conf = config["conf"]
                rate = config["rate"]
                save_res = config["save_res"]
                save_txt = config["save_txt"]
        self.save_res_button.setCheckState(Qt.CheckState(save_res))
        self.predictor.save_res = False if save_res == 0 else True
        # self.save_txt_button.setCheckState(Qt.CheckState(save_txt))
        # self.predictor.save_txt = False if save_txt == 0 else True
        self.run_button.setChecked(False)
        self.show_status("Welcome~")



#现在需要在stop页面里面写
    # Terminate button and associated state
    def stop(self):
        if self.thread.isRunning():
            self.thread.quit()  # end thread
        self.predictor.stop_dtc = True
        self.run_button.setChecked(False)  # start key recovery
        self.save_res_button.setEnabled(True)  # Ability to use the save button
        # self.save_txt_button.setEnabled(True)  # Ability to use the save button
        #self.pre_video.clear()  # clear image display
        #self.res_video.clear()  # clear image display
        # self.progress_bar.setValue(0)
        self.Class_num.setText("--")
        self.Target_num.setText("--")
        self.fps_label.setText("--")

    # Change detection parameters
    # def change_val(self, x, flag):
        # if flag == "iou_spinbox":
        #     self.iou_slider.setValue(
        #         int(x * 100)
        #     )  # The box value changes, changing the slider
        # elif flag == "iou_slider":
        #     self.iou_spinbox.setValue(
        #         x / 100
        #     )  # The slider value changes, changing the box
        #     self.show_status("IOU Threshold: %s" % str(x / 100))
        #     self.predictor.iou_thres = x / 100
        # elif flag == "conf_spinbox":
        #     self.conf_slider.setValue(int(x * 100))
        # elif flag == "conf_slider":
        #     self.conf_spinbox.setValue(x / 100)
        #     self.show_status("Conf Threshold: %s" % str(x / 100))
        #     self.predictor.conf_thres = x / 100
        # elif flag == "speed_spinbox":
        #     self.speed_slider.setValue(x)
        # elif flag == "speed_slider":
        #     self.speed_spinbox.setValue(x)
        #     self.show_status("Delay: %s ms" % str(x))
        #     self.predictor.speed_thres = x  # ms

    # change model
    def change_model(self, x):
        self.select_model = self.model_box.currentText()
        self.predictor.new_model_name = "./models/%s" % self.select_model
        self.show_status("Change Model：%s" % self.select_model)
        self.Model_name.setText(self.select_model)

    # Cycle monitoring model file changes
    def ModelBoxRefre(self):
        pt_list = os.listdir("./models")
        pt_list = [file for file in pt_list if file.endswith(".pt")]
        pt_list.sort(key=lambda x: os.path.getsize("./models/" + x))
        # It must be sorted before comparing, otherwise the list will be refreshed all the time
        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.model_box.clear()
            self.model_box.addItems(self.pt_list)

    # Get the mouse position (used to hold down the title bar and drag the window)
    def mousePressEvent(self, event):
        p = event.globalPosition()
        globalPos = p.toPoint()
        self.dragPos = globalPos

    # Optimize the adjustment when dragging the bottom and right edges of the window size
    def resizeEvent(self, event):
        # Update Size Grips
        UIFuncitons.resize_grips(self)


    def closeEvent(self, event):
        config_file = "config/setting.json"
        config = dict()
        config["iou"] = self.iou_spinbox.value()
        config["conf"] = self.conf_spinbox.value()
        config["rate"] = self.speed_spinbox.value()
        config["save_res"] = (
            0 if self.save_res_button.checkState() == Qt.Unchecked else 2
        )
        config["save_txt"] = (
            0 if self.save_txt_button.checkState() == Qt.Unchecked else 2
        )
        config_json = json.dumps(config, ensure_ascii=False, indent=2)
        with open(config_file, "w", encoding="utf-8") as f:
            f.write(config_json)
        # Exit the process before closing
        if self.thread.isRunning():
            self.predictor.stop_dtc = True
            self.thread.quit()
            MessageBox(
                self.close_button,
                title="Note",
                text="Exiting, please wait...",
                time=3000,
                auto=True,
            ).exec()
            sys.exit(0)
        else:
            sys.exit(0)
 
    """  def select_image():
        file_path, _ = QFileDialog.getOpenFileName(Home, '选择图片', '', 'Images (*.png *.xpm *.jpg *.bmp *.gif)')
        if file_path:
            pixmap = QPixmap(file_path)
            self.pre_video.setPixmap(pixmap.scaled(self.pre_video.width(), self.pre_video.height(), Qt.KeepAspectRatio)) """  
    



if __name__ == "__main__":
    app = QApplication(sys.argv)
    Home = MainWindow()
    Home.show()
    sys.exit(app.exec())

