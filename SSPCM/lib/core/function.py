


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time
import copy
import os
import math
import numpy as np
import torch

from core.config import get_model_name
from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images
from core.loss import JointsMSELoss
import cv2
import torch.nn.functional as F

import matplotlib.pyplot as plt
from scipy import stats


logger = logging.getLogger(__name__)

def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict, last_epoch_model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()


    # switch to train mode
    model.train()

    last_epoch_model.train()

    end = time.time()
    dis_batches = []



    for i, (input, target, target_weight, meta) in enumerate(train_loader):

  
        data_time.update(time.time() - end)

        if config.MIXUP_mode==True:
           
            pass

        elif config.uncertainty_and_time_correction==True or config.feature_attack_train==True:
            if type(input) == list:
                # RandAug
                if config.CONS_RAND_AUG:
                    sup_x, unsup_x, aug_unsup_x = input
                else:
                    sup_x, unsup_x = input

            with torch.no_grad():
                _, unsup_ht1_l = last_epoch_model.module.resnet(unsup_x.to('cuda'))
                _, unsup_ht2_l = last_epoch_model.module.resnet2(unsup_x.to('cuda'))
          

            output = model(input, target_weight, optimizer, epoch, unsup_ht1_l, unsup_ht2_l)

        else:
            #print(len(input), input[0].shape, input[1].shape)
            output = model(input)

        if type(target) == list:
            target = [t.cuda(non_blocking=True) for t in target]
            target_weight = [w.cuda(non_blocking=True) for w in target_weight]
        else:
            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

        loss = criterion(output, target, target_weight, meta)

        if type(loss) == tuple:
            sum_loss = loss[0]
            loss_dic = loss[1]
            if config.MODEL.NAME in ['pose_dual', 'my_pose_dual', 'my_pose_triple', 'my_pose_ensemble']:
                pseudo_target = loss[2]
            loss = loss[0]
        else:
            sum_loss = loss
            loss_dic = {}

        # compute gradient and do update step
        optimizer.zero_grad()
        sum_loss.backward()
        optimizer.step()


        # Get the supervised samples
        if config.MODEL.NAME in ['pose_dual', 'pose_cons', 'my_pose_dual', 'my_pose_cons',
                                 'my_pose_triple', 'my_pose_ensemble']:
            if type(target)==list:
                input, target, target_weight, meta = [input[0], target[0], target_weight[0], meta[0]]
            output = output[0]

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        hm_type = 'gaussian'

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                            target.detach().cpu().numpy(), hm_type)
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i % config.PRINT_FREQ == 0 or i == len(train_loader)-1 ) and config.LOCAL_RANK==0:
            # print(list(mapping_model.parameters())[0])
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.6f} ({loss.avg:.6f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)

            for key in loss_dic.keys():
                msg = msg + '\t{}: {:.6f}'.format(key, loss_dic[key])

            logger.info(msg)

            if config.DEBUG.DEBUG:
                if config.MODEL.NAME in ['pose_resnet']:

                    dirs = os.path.join(output_dir,'heatmap')
                    checkdir(dirs)
                    prefix = '{}_{}'.format(os.path.join(dirs, 'train'), i)
                    save_debug_images(config, input, meta, target, pred*4, output,
                                        prefix, None)


def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None, draw_flag=0):

    show_score = []
    show_incons = []
    show_score_qulity = []
    show_incons_qulity = []
    show_target_score = []

    show_ensemble_score = []
    show_triple_score = []

    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()


    criterion = JointsMSELoss(config.LOSS.USE_TARGET_WEIGHT,config)

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 3),
                         dtype=np.float32)
    all_preds_2 = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 3),
                         dtype=np.float32)
    all_preds_3 = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 3),
                         dtype=np.float32)

    all_preds_triple = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 3),
                         dtype=np.float32)

    all_boxes = np.zeros((num_samples, 10))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            if config.MODEL.NAME in ['pose_dual', 'my_pose_dual', 'my_pose_ensemble']:

                if config.MIXUP_mode == True:
                    output_list = model(input, target, target_weight)
                else:
                    output_list = model(input)

                #######output_list = model(input)
                output = output_list[0]
                output_2 = output_list[1]
            elif config.MODEL.NAME in ['my_pose_triple']:
                output_list = model(input)
                output = output_list[0]
                output_2 = output_list[1]
                output_triple = output_list[2]
            else:
             
                if config.MIXUP_mode == True:
                    output = model(input, target, target_weight)
                else:
                    output = model(input)

            if config.TEST.FLIP_TEST:
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                output_flipped = model(input_flipped)
                if config.MODEL.NAME in ['pose_dual', 'my_pose_dual', 'my_pose_ensemble']:
                    output_flipped_2 = output_flipped[1]
                    output_flipped = output_flipped[0]

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]
                else:
                # Else Affine Transform the heatmap to shift 3/4 pixel
                    batch_size,joint_num,height,width = output_flipped.shape
                    shift_x = 1.5/width
                    trans = cv2.getRotationMatrix2D((0,0), 0, 1)
                    trans[0,-1] -= shift_x
                    trans = trans[np.newaxis, :]
                    trans = np.repeat(trans,batch_size,0)
                    theta = torch.from_numpy(trans).cuda()

                    grid = F.affine_grid(theta, output_flipped.size()).float()
                    output_flipped = F.grid_sample(output_flipped, grid)

                output = (output + output_flipped) * 0.5
                # output = output_flipped

                if config.MODEL.NAME in ['pose_dual', 'my_pose_dual', 'my_pose_triple', 'my_pose_ensemble']:
                    output_flipped_2 = flip_back(output_flipped_2.cpu().numpy(),
                                            val_dataset.flip_pairs)
                    output_flipped_2 = torch.from_numpy(output_flipped_2.copy()).cuda()

                    # feature is not aligned, shift flipped heatmap for higher accuracy
                    if config.TEST.SHIFT_HEATMAP:
                        output_flipped_2[:, :, :, 1:] = \
                            output_flipped_2.clone()[:, :, :, 0:-1]
                    else:
                    # Else Affine Transform the heatmap to shift 3/4 pixel
                        batch_size,joint_num,height,width = output_flipped_2.shape
                        shift_x = 1.5/width
                        trans = cv2.getRotationMatrix2D((0,0), 0, 1)
                        trans[0,-1] -= shift_x
                        trans = trans[np.newaxis, :]
                        trans = np.repeat(trans,batch_size,0)
                        theta = torch.from_numpy(trans).cuda()

                        grid = F.affine_grid(theta, output_flipped_2.size()).float()
                        output_flipped_2 = F.grid_sample(output_flipped_2, grid)

                    output_2 = (output_2 + output_flipped_2) * 0.5
                    # output_2 = output_flipped_2

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = criterion(output, target, target_weight)

            hm_type='gaussian'

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy(),hm_type)

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            r = meta['rotation'].numpy()
            score = meta['score'].numpy()
            box = meta['raw_box'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s, r, hm_type)

            if config.MODEL.NAME in ['pose_dual', 'my_pose_dual', 'my_pose_triple', 'my_pose_ensemble']:
                preds_2, maxvals_2 = get_final_preds(
                        config, output_2.clone().cpu().numpy(), c, s, r, hm_type)
                all_preds_2[idx:idx + num_images, :, 0:2] = preds_2[:, :, 0:2]
                all_preds_2[idx:idx + num_images, :, 2:3] = maxvals_2

                preds_3, maxvals_3 = get_final_preds(
                        config, (0.5*output+0.5*output_2).clone().cpu().numpy(), c, s, r, hm_type)
                all_preds_3[idx:idx + num_images, :, 0:2] = preds_3[:, :, 0:2]
                all_preds_3[idx:idx + num_images, :, 2:3] = maxvals_3

                if config.MODEL.NAME in ['my_pose_triple']:
                    preds_triple, maxvals_triple = get_final_preds(
                        config, output_triple.clone().cpu().numpy(), c, s, r, hm_type)
                    all_preds_triple[idx:idx + num_images, :, 0:2] = preds_triple[:, :, 0:2]
                    all_preds_triple[idx:idx + num_images, :, 2:3] = maxvals_triple


            preds_target, maxvals_target = get_final_preds(
                config, target.clone().cpu().numpy(), c, s, r, hm_type)

            #print(target.shape)    torch.Size([32, 24, 64, 48])
            kp1_x = torch.Tensor(preds[:, :, 0])
            kp1_y = torch.Tensor(preds[:, :, 1])
            kp2_x = torch.Tensor(preds_2[:, :, 0])
            kp2_y = torch.Tensor(preds_2[:, :, 1])
            kp3_x = torch.Tensor(preds_triple[:, :, 0])
            kp3_y = torch.Tensor(preds_triple[:, :, 1])
            target_x = torch.Tensor(preds_target[:, :, 0])
            target_y = torch.Tensor(preds_target[:, :, 1])
            hm_height = torch.tensor(target.shape[2], dtype=torch.float)
            hm_width = torch.tensor(target.shape[3], dtype=torch.float)

            kp_pixel_dist = torch.sqrt((kp1_x - kp2_x) * (kp1_x - kp2_x) + (kp1_y - kp2_y) * (kp1_y - kp2_y))  ###
            kp_pixel_dist = kp_pixel_dist.view((kp_pixel_dist.shape[0], kp_pixel_dist.shape[1], 1))

            kp_dist_T = torch.zeros(kp_pixel_dist.shape)
            ######kp_dist_T[:, :, :] = kp_dist_filter_thread
            kp_dist_T_value = torch.sqrt((hm_width - 0) * (hm_width - 0) + (hm_height - 0) * (hm_height - 0))
            kp_dist_T[:, :, :] = kp_dist_T_value
            kp_pixel_dist = torch.min(kp_pixel_dist, kp_dist_T - 1) 
            kp_dist_uncertainty_percent = kp_pixel_dist / kp_dist_T_value
            ######################################

            gt_dist = torch.sqrt((kp3_x - target_x) * (kp3_x - target_x) +
                                       (kp3_y - target_y) * (kp3_y - target_y))  ### 
            gt_dist = gt_dist.view((gt_dist.shape[0], gt_dist.shape[1], 1))

            score_gt_dist = torch.sqrt((kp1_x - target_x) * (kp1_x - target_x) +
                                 (kp1_y - target_y) * (kp1_y - target_y))  ###
            score_gt_dist = score_gt_dist.view((score_gt_dist.shape[0], score_gt_dist.shape[1], 1))


            kp_dist_T = torch.zeros(gt_dist.shape)
            kp_dist_T_value = torch.sqrt((hm_width - 0) * (hm_width - 0) + (hm_height - 0) * (hm_height - 0))
            kp_dist_T[:, :, :] = kp_dist_T_value
            gt_dist = torch.min(gt_dist, kp_dist_T - 1)  ###
            ###
            kp_pos_aqulity = 1 - gt_dist / kp_dist_T_value

            score_gt_dist = torch.min(score_gt_dist, kp_dist_T - 1)
            score_kp_pos_aqulity = 1 - score_gt_dist / kp_dist_T_value
            ################################################################

            #print('lalal')
            ### torch.Size([32, 24, 1]) (32, 24, 1)
            ###print(kp_dist_uncertainty_percent.shape, maxvals_triple.shape)  ##
            kp_dist_uncertainty_percent = kp_dist_uncertainty_percent.reshape(-1)
            maxvals_tmp = maxvals.reshape(-1)
            maxvals_2_tmp = maxvals_2.reshape(-1)
            maxvals_triple = maxvals_triple.reshape(-1)
            kp_pos_aqulity = kp_pos_aqulity.reshape(-1)
            maxvals_target_tmp = maxvals_target.reshape(-1)
            score_kp_pos_aqulity = score_kp_pos_aqulity.reshape(-1)

            show_score.append(maxvals_tmp)
            show_ensemble_score.append((maxvals_tmp + maxvals_2_tmp) / 2.0)
            show_triple_score.append(maxvals_triple / 2.0)
            show_incons.append(kp_dist_uncertainty_percent)
            show_incons_qulity.append(kp_pos_aqulity)
            show_target_score.append(maxvals_target_tmp)
            show_score_qulity.append(score_kp_pos_aqulity)
            #################################################



            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]

            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score

            # print(box)
            all_boxes[idx:idx + num_images, 6:] = np.array(box)
            image_path.extend(meta['image'])
            if config.DATASET.TEST_DATASET == 'posetrack':
                filenames.extend(meta['filename'])
                imgnums.extend(meta['imgnum'].numpy())

            idx += num_images

            if (i % config.PRINT_FREQ == 0 or i == len(val_loader) - 1 ) and config.LOCAL_RANK==0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                if config.DEBUG.DEBUG:
                    # input = model.module.x
                    dirs = os.path.join(output_dir,'heatmap')
                    checkdir(dirs)
                    prefix = '{}_{}'.format(os.path.join(dirs, 'val'), i)
                    save_debug_images(config, input, meta, target, pred*4, output, prefix)

                    if config.MODEL.NAME in ['pose_dual', 'my_pose_dual', 'my_pose_triple', 'my_pose_ensemble']:
                        dirs = os.path.join(output_dir, 'sup_2')
                        checkdir(dirs)
                        prefix = '{}_{}'.format(os.path.join(dirs, 'val'), i)
                        save_debug_images(config, input, meta, target, pred*4, output_2, prefix)

                        if config.MODEL.NAME in ['my_pose_triple']:
                            dirs = os.path.join(output_dir, 'sup_triple')
                            checkdir(dirs)
                            prefix = '{}_{}'.format(os.path.join(dirs, 'val'), i)
                            save_debug_images(config, input, meta, target, pred * 4, output_triple, prefix)
                            save_debug_images(config, input, meta, target, pred * 4, output_triple, prefix)

        np.save(os.path.join(output_dir, 'all_preds.npy'), all_preds)

        if config.LOCAL_RANK!=0:
            return 0
        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums, prefix = 'eval_1')

        _, full_arch_name = get_model_name(config)
        logger.info('The Predictions of Net 1')

        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, full_arch_name)
        else:
            _print_name_value(name_values, full_arch_name)

        if config.MODEL.NAME in ['pose_dual', 'my_pose_dual', 'my_pose_triple', 'my_pose_ensemble']:
            name_values_2, perf_indicator_2 = val_dataset.evaluate(
                config, all_preds_2, output_dir, all_boxes, image_path,
                'head2_pred.mat', imgnums, prefix = 'eval_2')
            logger.info('The Predictions of Net 2')
            if isinstance(name_values_2, list):
                for name_value in name_values_2:
                    _print_name_value(name_value, full_arch_name)
            else:
                _print_name_value(name_values_2, full_arch_name)

            name_values_3, perf_indicator_3 = val_dataset.evaluate(
                config, all_preds_3, output_dir, all_boxes, image_path,
                'ensemble_pred.mat', imgnums, prefix = 'eval_3')

            logger.info('Ensemble Predictions')
            _print_name_value(name_values_3, full_arch_name)

            name_values_triple, perf_indicator_triple = val_dataset.evaluate(
                config, all_preds_triple, output_dir, all_boxes, image_path,
                'triple_pred.mat', imgnums, prefix='eval_triple')
            logger.info('triple model final Predictions')
            _print_name_value(name_values_triple, full_arch_name)




    return perf_indicator


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.4f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def checkdir(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)






