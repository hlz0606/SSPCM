
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os
import logging
import contextlib
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from torch.autograd import Function
from collections import OrderedDict
import numpy as np
import random
import cv2
from utils.transforms import get_affine_transform, cvt_MToTheta
from core.inference import get_max_preds_tensor, get_final_preds
from core.loss import JointsMSELoss

from .pose_hrnet import PoseHighResolutionNet
from utils.utils import get_optimizer


from torch import randperm


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def mask_joint(image, joints, MASK_JOINT_NUM=4):
    ## N,J,2 joints
    N, J = joints.shape[:2]
    _, _, width, height = image.shape
    re_joints = joints[:, :, :2] + torch.randn((N, J, 2)).cuda() * 10
    re_joints = re_joints.int()
    size = torch.randint(10, 20, (N, J, 2)).int().cuda()

    x0 = re_joints[:, :, 0] - size[:, :, 0]
    y0 = re_joints[:, :, 1] - size[:, :, 1]

    x1 = re_joints[:, :, 0] + size[:, :, 0]
    y1 = re_joints[:, :, 1] + size[:, :, 1]

    torch.clamp_(x0, 0, width)
    torch.clamp_(x1, 0, width)
    torch.clamp_(y0, 0, height)
    torch.clamp_(y1, 0, height)

    for i in range(N):
        # num = np.random.randint(MASK_JOINT_NUM)
        # ind = np.random.choice(J, num)
        ind = np.random.choice(J, MASK_JOINT_NUM)

        for j in ind:
            image[i, :, y0[i, j]:y1[i, j], x0[i, j]:x1[i, j]] = 0
    return image





class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64):
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1):
        super(Bottleneck, self).__init__()

        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=groups, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(width, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck_CAFFE(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_CAFFE, self).__init__()
        # add stride to conv1x1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PoseResNet(nn.Module):

    def __init__(self, block, layers, cfg, groups=1, width_per_group=64, **kwargs):
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS
        self.cfg = cfg
        self.groups = groups
        self.base_width = width_per_group

        super(PoseResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            extra.NUM_DECONV_LAYERS,
            extra.NUM_DECONV_FILTERS,
            extra.NUM_DECONV_KERNELS,
        )

        self.final_layer_1 = nn.Conv2d(
            in_channels=extra.NUM_DECONV_FILTERS[-1],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x, attack=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        fea = self.layer4(x)   ### torch.Size([32, 512, 8, 6])

        ################
        if attack!=None:
            fea = attack(fea)
        ##################################

        x = self.deconv_layers(fea)
        ht_1 = self.final_layer_1(x)

        return fea, ht_1

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

            logger.info('=> init final conv weights from normal distribution')
            for m in self.final_layer_1.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)

            logger.info('=> loading pretrained model {}'.format(pretrained))
            checkpoint = torch.load(pretrained, map_location='cpu')
            if isinstance(checkpoint, OrderedDict):
                state_dict = checkpoint
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict_old = checkpoint['state_dict']
                state_dict = OrderedDict()
                # delete 'module.' because it is saved from DataParallel module
                for key in state_dict_old.keys():
                    if key.startswith('module.'):
                        state_dict[key[7:]] = state_dict_old[key]
                    else:
                        state_dict[key] = state_dict_old[key]
            else:
                raise RuntimeError(
                    'No state_dict found in checkpoint file {}'.format(pretrained))

            # delete 'module.'
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            # if list(state_dict.keys())[0][:6] == 'resnet2':
            #     state_dict = {k[8:]:v for k,v in state_dict.items()}

            if list(state_dict.keys())[0][:6] == 'resnet':
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            self.load_state_dict(state_dict, strict=False)
        else:
            logger.error('=> imagenet pretrained model dose not exist')
            logger.error('=> please download it first')
            raise ValueError('imagenet pretrained model does not exist')


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


class MyPoseTriple(nn.Module):

    def __init__(self, resnet, resnet2, resnet3, cfg, resnet_tch=None, resnet_seg=None, **kwargs):
        super(MyPoseTriple, self).__init__()
        # np.random.seed(1314)
        self.resnet = resnet
        self.resnet2 = resnet2
        self.final_resnet = resnet3

        #######
        if cfg.uncertainty_and_time_correction==True:
            self.resnet_l = 0
            self.resnet2_l = 0


        ### 
        if cfg.feature_attack_train==True:    ##
            self.attack_net = feature_attack_generator(mask_num=cfg.feature_attack_mask_num)

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.image_size = cfg.MODEL.IMAGE_SIZE
        self.cfg = cfg

        self.multi_infer = False
        self.flip = False
        self.scale_set = [0.7, 0.9, 1.1, 1.3]
        self.flip_pairs = [[1, 4], [2, 5], [3, 6], [14, 17], [15, 18], [16, 19], [20, 21], [22, 23]]

        ########
        self.each_kp_pl_used_num = np.zeros(cfg.MODEL.NUM_JOINTS)

        self.epoch = 0
        self.first_record_flag = 0    ##

    def get_batch_affine_transform(self, batch_size):
        sf = self.scale_factor
        rf = self.rotation_factor
        # shift_f = 0.1
        # shear_f = 0.10

        batch_trans = []
        for b in range(batch_size):
            r, s = 1, 1
            s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
                if random.random() <= 0.8 else 0
            trans = cv2.getRotationMatrix2D((0, 0), r, s)

            batch_trans.append(trans)
        batch_trans = np.stack(batch_trans, 0)
        batch_trans = torch.from_numpy(batch_trans).cuda()
        return batch_trans

    # def forward(self, x, target, target_weight):
    def forward(self, x, target_weight=None, optimizer=None, epoch=None, unsup_ht1_l=None, unsup_ht2_l=None):

        joint_pairs = [(1, 4), (2, 5), (3, 6), (14, 17), (15, 18), (16, 19),
                       (20, 21), (22, 23)]

        # Training
        if type(x) == list:
            # RandAug
            if self.cfg.CONS_RAND_AUG:
                sup_x, unsup_x, aug_unsup_x = x
            else:
                sup_x, unsup_x = x

            batch_size = x[0].shape[0]

            ######
            sup_fea1, sup_ht1 = self.resnet(sup_x)
            sup_fea2, sup_ht2 = self.resnet2(sup_x)
            sup_fea3, sup_ht3 = self.final_resnet(sup_x)

            # Teacher
            # Easy Augmentation
            with torch.no_grad():   

                unsup_fea1, unsup_ht1 = self.resnet(unsup_x)
                unsup_fea2, unsup_ht2 = self.resnet2(unsup_x)

                if 1:
                    if 1:
          
                        if self.cfg.uncertainty_and_time_correction==True:

                            if epoch > self.cfg.time_correction_warmup_epoch:
                               

                                unsup_ht1_l = unsup_ht1_l
                                unsup_ht2_l = unsup_ht2_l

                                ###
                                unsup_ht1 = unsup_ht1.cpu()
                                unsup_ht2 = unsup_ht2.cpu()
                                unsup_ht1_l = unsup_ht1_l.cpu()
                                unsup_ht2_l = unsup_ht2_l.cpu()

                     
                                preds_1, score1_tmp = get_max_preds_tensor(unsup_ht1.detach())
                                preds_2, score2_tmp = get_max_preds_tensor(unsup_ht2.detach())
                                preds_1_l, score1_l_tmp = get_max_preds_tensor(unsup_ht1_l.detach())
                                preds_2_l, score2_l_tmp = get_max_preds_tensor(unsup_ht2_l.detach())

                                kp1_x = preds_1[:, :, 0]
                                kp1_y = preds_1[:, :, 1]
                                kp2_x = preds_2[:, :, 0]
                                kp2_y = preds_2[:, :, 1]

                                kp1_l_x = preds_1_l[:, :, 0]
                                kp1_l_y = preds_1_l[:, :, 1]
                                kp2_l_x = preds_2_l[:, :, 0]
                                kp2_l_y = preds_2_l[:, :, 1]

                                hm_height = torch.tensor(unsup_ht1.shape[2], dtype=torch.float)
                                hm_width = torch.tensor(unsup_ht1.shape[3], dtype=torch.float)

                                kp_pixel_dist_1 = torch.sqrt(
                                    (kp1_x - kp2_x) * (kp1_x - kp2_x) + (kp1_y - kp2_y) * (kp1_y - kp2_y))  ### 
                                kp_pixel_dist_1 = kp_pixel_dist_1.view((1, kp_pixel_dist_1.shape[0], kp_pixel_dist_1.shape[1], 1))

                                kp_pixel_dist_2 = torch.sqrt(
                                    (kp1_l_x - kp2_x) * (kp1_l_x - kp2_x) + (kp1_l_y - kp2_y) * (kp1_l_y - kp2_y))  ###
                                kp_pixel_dist_2 = kp_pixel_dist_2.view((1, kp_pixel_dist_2.shape[0], kp_pixel_dist_2.shape[1], 1))

                                kp_pixel_dist_3 = torch.sqrt(
                                    (kp1_x - kp2_l_x) * (kp1_x - kp2_l_x) + (kp1_y - kp2_l_y) * (kp1_y - kp2_l_y))  ### 
                                kp_pixel_dist_3 = kp_pixel_dist_3.view(
                                    (1, kp_pixel_dist_3.shape[0], kp_pixel_dist_3.shape[1], 1))

                                kp_pixel_dist_4 = torch.sqrt(
                                    (kp1_l_x - kp2_l_x) * (kp1_l_x - kp2_l_x) + (kp1_l_y - kp2_l_y) * (kp1_l_y - kp2_l_y))  ###
                                kp_pixel_dist_4 = kp_pixel_dist_4.view(
                                    (1, kp_pixel_dist_4.shape[0], kp_pixel_dist_4.shape[1], 1))

               

                                cat_kp_pixel_dist = torch.cat((kp_pixel_dist_1, kp_pixel_dist_2, kp_pixel_dist_3,
                                                               kp_pixel_dist_4), dim=0)

                                _, min_kp_pixel_dist_id = torch.min(cat_kp_pixel_dist, dim=0)

                  
                                ensemble_unsup_ht1_tmp = (0.5 * unsup_ht1 + 0.5 * unsup_ht2).clone()
                                ensemble_unsup_ht2_tmp = (0.5 * unsup_ht1_l + 0.5 * unsup_ht2).clone()
                                ensemble_unsup_ht3_tmp = (0.5 * unsup_ht1 + 0.5 * unsup_ht2_l).clone()
                                ensemble_unsup_ht4_tmp = (0.5 * unsup_ht1_l + 0.5 * unsup_ht2_l).clone()

                                ensemble_unsup_ht1_tmp = ensemble_unsup_ht1_tmp.view((ensemble_unsup_ht1_tmp.shape[0],
                                                                            ensemble_unsup_ht1_tmp.shape[1],
                                                                                      1,
                                                                            ensemble_unsup_ht1_tmp.shape[2],
                                                                            ensemble_unsup_ht1_tmp.shape[3]))
                                ensemble_unsup_ht2_tmp = ensemble_unsup_ht2_tmp.view((ensemble_unsup_ht2_tmp.shape[0],
                                                                                      ensemble_unsup_ht2_tmp.shape[1],
                                                                                      1,
                                                                                      ensemble_unsup_ht2_tmp.shape[2],
                                                                                      ensemble_unsup_ht2_tmp.shape[3]))
                                ensemble_unsup_ht3_tmp = ensemble_unsup_ht3_tmp.view((ensemble_unsup_ht3_tmp.shape[0],
                                                                                      ensemble_unsup_ht3_tmp.shape[1],
                                                                                      1,
                                                                                      ensemble_unsup_ht3_tmp.shape[2],
                                                                                      ensemble_unsup_ht3_tmp.shape[3]))
                                ensemble_unsup_ht4_tmp = ensemble_unsup_ht4_tmp.view((ensemble_unsup_ht4_tmp.shape[0],
                                                                                      ensemble_unsup_ht4_tmp.shape[1],
                                                                                      1,
                                                                                      ensemble_unsup_ht4_tmp.shape[2],
                                                                                      ensemble_unsup_ht4_tmp.shape[3]))

                                #print(ensemble_unsup_ht1_tmp.shape)  ### torch.Size([32, 24, 64, 48])


                                ensemble_unsup_ht_tmp = torch.cat((ensemble_unsup_ht1_tmp, ensemble_unsup_ht2_tmp,
                                            ensemble_unsup_ht3_tmp, ensemble_unsup_ht4_tmp), dim=2)
                                #print(ensemble_unsup_ht_tmp.shape)##### torch.Size([32, 24, 4, 64, 48])

                                ensemble_unsup_ht = torch.zeros((unsup_ht1.shape[0],
                                                                            unsup_ht1.shape[1],
                                                                            unsup_ht1.shape[2],
                                                                            unsup_ht1.shape[3]))
                                ###########
                                for img_id in range(ensemble_unsup_ht_tmp.shape[0]):
                                    for kp_id in range(ensemble_unsup_ht_tmp.shape[1]):
                                        ensemble_unsup_ht[img_id, kp_id] = \
                                            ensemble_unsup_ht_tmp[img_id, kp_id, min_kp_pixel_dist_id[img_id, kp_id, 0]]

                                ensemble_unsup_ht = ensemble_unsup_ht.to('cuda')
                                #print(ensemble_unsup_ht.shape)

                                unsup_ht1 = unsup_ht1.to('cuda')
                                unsup_ht2 = unsup_ht2.to('cuda')
                                preds_1 = preds_1.to('cuda')
                                preds_2 = preds_2.to('cuda')

         

            if self.cfg.CONS_RAND_AUG:
                unsup_x = aug_unsup_x

            unsup_x_trans = unsup_x.clone()
            unsup_x_trans_2 = unsup_x.clone()
            ensemble_unsup_x_trans = unsup_x.clone()

            with torch.no_grad():

                ########
                if self.cfg.MASK_JOINT_NUM * self.cfg.keypoint_perception_cutmix_num * self.cfg.CUTOUT_NUM != 0:
                    raise 'error'

                if self.cfg.MASK_JOINT_NUM > 0 or self.cfg.keypoint_perception_cutmix_num > 0:
                    preds_1, score1_tmp = get_max_preds_tensor(unsup_ht1.detach())
                    preds_2, score2_tmp = get_max_preds_tensor(unsup_ht2.detach())

                    preds_ensemble, _ = get_max_preds_tensor(ensemble_unsup_ht.detach())


                    if self.cfg.keypoint_perception_cutmix_num > 0:

                        unsup_x_trans = \
                            cutmix_based_on_keypoint_perception(unsup_x_trans, preds_2 * 4,
                                                                self.cfg.keypoint_perception_cutmix_num)

                        unsup_x_trans_2 = \
                            cutmix_based_on_keypoint_perception(unsup_x_trans_2, preds_1 * 4,
                                                                self.cfg.keypoint_perception_cutmix_num)

                        ensemble_unsup_x_trans = \
                            cutmix_based_on_keypoint_perception(ensemble_unsup_x_trans, preds_ensemble * 4,
                                                            self.cfg.keypoint_perception_cutmix_num)

                    elif self.cfg.MASK_JOINT_NUM > 0:
                        unsup_x_trans = mask_joint(unsup_x_trans, preds_2 * 4, self.cfg.MASK_JOINT_NUM)
                        unsup_x_trans_2 = mask_joint(unsup_x_trans_2, preds_1 * 4, self.cfg.MASK_JOINT_NUM)
                        ensemble_unsup_x_trans = mask_joint(ensemble_unsup_x_trans, preds_ensemble * 4,
                                                            self.cfg.MASK_JOINT_NUM)
   



            # Transform
            # Apply Affine Transformation again for hard augmentation
            if self.cfg.UNSUP_TRANSFORM:
                with torch.no_grad():
                    theta = self.get_batch_affine_transform(batch_size)
                    grid = F.affine_grid(theta, sup_x.size()).float()

                    unsup_x_trans = F.grid_sample(unsup_x_trans, grid)
                    unsup_x_trans_2 = F.grid_sample(unsup_x_trans_2, grid)
                    ensemble_unsup_x_trans = F.grid_sample(ensemble_unsup_x_trans, grid)

                    ht_grid = F.affine_grid(theta, unsup_ht1.size()).float()

                    unsup_ht_trans1 = F.grid_sample(unsup_ht1.detach(), ht_grid)
                    unsup_ht_trans2 = F.grid_sample(unsup_ht2.detach(), ht_grid)
                    unsup_ht_trans_ensemble = F.grid_sample(ensemble_unsup_ht.detach(), ht_grid)

            else:
                # Raw image
                theta = torch.eye(2, 3).repeat(batch_size, 1, 1).double().cuda()

                unsup_ht_trans1 = unsup_ht1.detach().clone()
                unsup_ht_trans2 = unsup_ht2.detach().clone()
                unsup_ht_trans_ensemble = ensemble_unsup_ht.detach().clone()

            # Student
            # Hard Augmentation
            if self.cfg.feature_attack_train==True:
                _, cons_ht1 = self.resnet(unsup_x_trans, self.attack_net)
                _, cons_ht2 = self.resnet2(unsup_x_trans_2, self.attack_net)
                _, cons_ht_ensemble = self.final_resnet(ensemble_unsup_x_trans, self.attack_net)

            else:
                _, cons_ht1 = self.resnet(unsup_x_trans)
                _, cons_ht2 = self.resnet2(unsup_x_trans_2)
                _, cons_ht_ensemble = self.final_resnet(ensemble_unsup_x_trans)


            score_filter_thread = self.cfg.score_filter_thread
            kp_dist_filter_thread = self.cfg.kp_pixel_dist_filter_thread

            if self.cfg.MASK_JOINT_NUM > 0:  ##
                pass
            else:
                preds_1, score1_tmp = get_max_preds_tensor(unsup_ht1.detach())
                preds_2, score2_tmp = get_max_preds_tensor(unsup_ht2.detach())


            teacher_ensemble_score = (score1_tmp + score2_tmp) / 2.0

        
            score_mask = teacher_ensemble_score.gt(score_filter_thread)   ##### ，
            score_mask = score_mask.view((score_mask.shape[0], score_mask.shape[1], 1))
            score_mask = score_mask.to('cuda')

            score_diff_k = 1.0 - torch.abs(score1_tmp - score2_tmp)
            score_diff_k = score_diff_k.view((score_diff_k.shape[0], score_diff_k.shape[1], 1))
            min_score_T = torch.zeros(score_diff_k.shape).to('cuda')  ## 
            max_score_T = torch.ones(score_diff_k.shape).to('cuda')  ## 
            score_diff_k = torch.max(score_diff_k, min_score_T)
            score_diff_k = torch.min(score_diff_k, max_score_T)
            ######################################################

            kp1_x = preds_1[:, :, 0]
            kp1_y = preds_1[:, :, 1]
            kp2_x = preds_2[:, :, 0]
            kp2_y = preds_2[:, :, 1]
            hm_height = torch.tensor(unsup_ht1.shape[2], dtype=torch.float)
            hm_width = torch.tensor(unsup_ht1.shape[3], dtype=torch.float)
            kp_pixel_dist = torch.sqrt((kp1_x - kp2_x) * (kp1_x - kp2_x) + (kp1_y - kp2_y) * (kp1_y - kp2_y))  ### 
            kp_pixel_dist = kp_pixel_dist.view((kp_pixel_dist.shape[0], kp_pixel_dist.shape[1], 1))


            kp_dist_T = torch.zeros(kp_pixel_dist.shape).to('cuda')
       
            kp_dist_T_value = torch.sqrt((hm_width - 0) * (hm_width - 0) + (hm_height - 0) * (hm_height - 0))
            kp_dist_T[:, :, :] = kp_dist_T_value
            kp_pixel_dist = torch.min(kp_pixel_dist, kp_dist_T - 1)  ###
            ##
            kp_dist_uncertainty_percent = kp_pixel_dist / kp_dist_T_value   ### 

           
            kp_dist_mask = kp_dist_uncertainty_percent.lt(kp_dist_filter_thread)   ####
            kp_dist_mask = kp_dist_mask.view((kp_dist_mask.shape[0], kp_dist_mask.shape[1], 1))
            kp_dist_mask = kp_dist_mask.to('cuda')

            kp_dist_uncertainty_k = (kp_dist_T_value - kp_pixel_dist) / kp_dist_T_value
            

            out_dic = {
                'unsup_x_trans': unsup_x_trans,
                'unsup_x_trans_2': unsup_x_trans_2,
                'ensemble_unsup_x_trans': ensemble_unsup_x_trans,

                'theta': theta,
            }

            if torch.cuda.is_available():
                kp_dist_uncertainty_k = torch.tensor(kp_dist_uncertainty_k)
                kp_dist_uncertainty_k = kp_dist_uncertainty_k.to('cuda')



            return sup_ht1, sup_ht2, sup_ht3,\
                   unsup_ht1, unsup_ht2, ensemble_unsup_ht,\
                   unsup_ht_trans1, unsup_ht_trans2, unsup_ht_trans_ensemble,\
                   cons_ht1, cons_ht2, cons_ht_ensemble,\
                   kp_dist_mask, kp_dist_uncertainty_k, score_mask, score_diff_k,\
                   out_dic

        # Inference
        else:
            batch_size, _, height, width = x.shape

            fea1, ht1 = self.resnet(x)
            fea2, ht2 = self.resnet2(x)
            fea3, ht3 = self.final_resnet(x)

            return ht1, ht2, ht3


def get_pose_net(cfg, is_train, **kwargs):
    if cfg.MODEL.BACKBONE == 'resnet':
        num_layers = cfg.MODEL.EXTRA.NUM_LAYERS
        style = cfg.MODEL.STYLE
        block_class, layers = resnet_spec[num_layers]
        if style == 'caffe':
            block_class = Bottleneck_CAFFE

        resnet = PoseResNet(block_class, layers, cfg, **kwargs)
        resnet2 = PoseResNet(block_class, layers, cfg, **kwargs)

        ###
        if cfg.use_diff_model == True:
            ####################
            num_layers = cfg.OTHER_MODEL.MODEL.EXTRA.NUM_LAYERS
            style = cfg.MODEL.STYLE
            block_class, layers = resnet_spec[num_layers]
            if style == 'caffe':
                block_class = Bottleneck_CAFFE
            resnet3 = PoseResNet(block_class, layers, cfg.OTHER_MODEL, **kwargs)

        else:
            resnet3 = PoseResNet(block_class, layers, cfg, **kwargs)

    elif cfg.MODEL.BACKBONE == 'hrnet':
        resnet = PoseHighResolutionNet(cfg, **kwargs)
        resnet2 = PoseHighResolutionNet(cfg, **kwargs)
        resnet3 = PoseHighResolutionNet(cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        resnet.init_weights(cfg.MODEL.PRETRAINED)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        logger.info('Model 2 => loading pretrained model {}'.format(cfg.MODEL.PRETRAINED))
        resnet2.init_weights(cfg.MODEL.PRETRAINED)

        ##
    if cfg.use_diff_model == True:
        if is_train and cfg.MODEL.INIT_WEIGHTS:
            logger.info('Model 3 => loading pretrained model {}'.format(cfg.OTHER_MODEL.MODEL.PRETRAINED))
            resnet3.init_weights(cfg.OTHER_MODEL.MODEL.PRETRAINED)
    else:
        if is_train and cfg.MODEL.INIT_WEIGHTS:
            logger.info('Model 3 => loading pretrained model {}'.format(cfg.MODEL.PRETRAINED))
            resnet3.init_weights(cfg.MODEL.PRETRAINED)

    model = MyPoseTriple(resnet, resnet2, resnet3, cfg)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        state_dict = torch.load(cfg.MODEL.PRETRAINED, map_location='cpu')
        if 'resnet2.conv1.weight' in state_dict:
            print('pretrained')
            model.load_state_dict(state_dict, strict=False)

    return model

def generate_a_heatmap(arr, centers, max_values, sigma):
    """Generate pseudo heatmap for one keypoint in one frame.

    Args:
        arr (np.ndarray): The array to store the generated heatmaps. Shape: heatmap_h * heatmap_w.
        centers (np.ndarray): The coordinates of corresponding keypoints (of multiple persons). Shape: M * 2.
        max_values (np.ndarray): The confidence score of each keypoint. Shape: M.

    Returns:
        np.ndarray: The generated pseudo heatmap.
    """
    EPS = 1e-3

    sigma = sigma
    heatmap_h, heatmap_w = arr.shape
    #print(centers.shape, max_values.shape)
    for center, max_value in zip(centers, max_values):
        if max_value < EPS:
            continue

        mu_x, mu_y = center[0], center[1]
        st_x = max(int(mu_x - 3 * sigma), 0)
        ed_x = min(int(mu_x + 3 * sigma) + 1, heatmap_w)
        st_y = max(int(mu_y - 3 * sigma), 0)
        ed_y = min(int(mu_y + 3 * sigma) + 1, heatmap_h)
        x = torch.arange(st_x, ed_x, 1).to('cuda')
        y = torch.arange(st_y, ed_y, 1).to('cuda')

        # if the keypoint not in the heatmap coordinate system
        if not (len(x) and len(y)):
            continue
        y = y[:, None]

        patch = torch.exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / 2 / sigma ** 2)
        patch = patch * max_value
        ######
        ###arr[st_y:ed_y, st_x:ed_x] = np.maximum(arr[st_y:ed_y, st_x:ed_x], patch)
        arr[st_y:ed_y, st_x:ed_x] = torch.max(arr[st_y:ed_y, st_x:ed_x], patch)




def cutmix_based_on_keypoint_perception(image, joints, MASK_JOINT_NUM=4):

   
    image_tmp = image.clone()

    ## N,J,2 joints
    N, J = joints.shape[:2]
    _, _, height, width = image.shape
    re_joints = joints[:, :, :2] + torch.randn((N, J, 2)).cuda() * 10
    re_joints = 0 + re_joints.int()
    size = torch.randint(10, 20, (N, J, 2)).int().cuda()

    center_x = copy.deepcopy(re_joints[:, :, 1])
    center_y = copy.deepcopy(re_joints[:, :, 0])

    x0 = re_joints[:, :, 1] - size[:, :, 1]
    y0 = re_joints[:, :, 0] - size[:, :, 0]

    x1 = re_joints[:, :, 1] + size[:, :, 1]
    y1 = re_joints[:, :, 0] + size[:, :, 0]

    x0 = torch.clamp(x0, 0, width-1)
    x1 = torch.clamp(x1, 0, width-1)
    y0 = torch.clamp(y0, 0, height-1)
    y1 = torch.clamp(y1, 0, height-1)

    for i in range(N):
        ind = np.random.choice(J, MASK_JOINT_NUM)
        ind_2 = np.random.choice(J, MASK_JOINT_NUM)    ######
        img_id = np.random.randint(0, N)    #####
        ##

        for idx in range(len(ind)):
            j = ind[idx]
            j2 = ind_2[idx]

            x_start = center_x[i, j] - abs(x0[img_id, j2] - center_x[img_id, j2])    #######
            x_end = center_x[i, j] + abs(x1[img_id, j2] - center_x[img_id, j2])
            y_start = center_y[i, j] - abs(y0[img_id, j2] - center_y[img_id, j2])  ######
            y_end = center_y[i, j] + abs(y1[img_id, j2] - center_y[img_id, j2])

   
            x_start = torch.clamp(x_start, 0, width-1)
            x_end = torch.clamp(x_end, 0, width-1)
            y_start = torch.clamp(y_start, 0, height-1)
            y_end = torch.clamp(y_end, 0, height-1)

   
            offset_y = abs(image[img_id, :, y0[img_id, j2]: y1[img_id, j2], x0[img_id, j2]: x1[img_id, j2]].shape[-2] - \
                            image[i, :, y_start: y_end, x_start: x_end].shape[-2])
            offset_x = abs(image[img_id, :, y0[img_id, j2]: y1[img_id, j2], x0[img_id, j2]: x1[img_id, j2]].shape[-1] - \
                       image[i, :, y_start: y_end, x_start: x_end].shape[-1])
            offset_y_start = 0
            offset_x_start = 0
            offset_y_end = 0
            offset_x_end = 0
            if y_start == 0:
                offset_y_start = offset_y
            if x_start == 0:
                offset_x_start = offset_x
            if y_end == height-1:
                offset_y_end = offset_y
            if x_end == width-1:
                offset_x_end = offset_x


            if image[i, :, y_start: y_end, x_start: x_end].shape[-1] == 0 or \
                    image[i, :, y_start: y_end, x_start: x_end].shape[-2] == 0:  ##
                pass

            elif image[img_id, :, y0[img_id, j2] + offset_y_start: y1[img_id, j2] - offset_y_end,
                        x0[img_id, j2] + offset_x_start: x1[img_id, j2] - offset_x_end].shape[-1] == 0 or \
                    image[img_id, :, y0[img_id, j2] + offset_y_start: y1[img_id, j2] - offset_y_end,
                    x0[img_id, j2] + offset_x_start: x1[img_id, j2] - offset_x_end].shape[-2] == 0:  ##
                image[i, :, y_start: y_end, x_start: x_end] = 0

            elif image[i, :, y_start: y_end, x_start: x_end].shape != image[img_id, :, y0[img_id, j2] + offset_y_start: y1[img_id, j2] - offset_y_end,
                        x0[img_id, j2] + offset_x_start: x1[img_id, j2] - offset_x_end].shape:
                image[i, :, y_start: y_end, x_start: x_end] = 0

            else:    ### keypoint cutmix
                image[i, :, y_start: y_end, x_start: x_end] = \
                        image_tmp[img_id, :, y0[img_id, j2] + offset_y_start: y1[img_id, j2] - offset_y_end,
                            x0[img_id, j2] + offset_x_start: x1[img_id, j2] - offset_x_end]

    return image




class feature_attack_generator(nn.Module):
    def __init__(self, mask_num):   ##
        super(feature_attack_generator, self).__init__()
        ## 
        self.mask_num = mask_num

    def forward(self, fea):    
        attack_mask = torch.ones((fea.shape[0], fea.shape[1], fea.shape[2], fea.shape[3])).bool().to('cuda')

        mask_id = np.random.randint(0, fea.shape[2] * fea.shape[3], (fea.shape[0]))
        

        for img_id in range(mask_id.shape[0]):
            ### 
            row = mask_id[img_id] // fea.shape[3]     ## 
            col = mask_id[img_id] % fea.shape[3]     ## 

            attack_mask[img_id, :, row, col] = 0

        fea = fea * attack_mask

        return fea







