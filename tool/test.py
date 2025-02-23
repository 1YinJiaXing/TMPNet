import os
import time
import random
import numpy as np
import logging
import pickle
import argparse
import collections

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data

from util import config
from util.common_util import AverageMeter, intersectionAndUnionGPU, check_makedirs
from util.data_util import collate_fn
from util.lingjian import CustomDataset  # Import CustomDataset from dataset.py

random.seed(123)
np.random.seed(123)


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
    parser.add_argument('--config', type=str, default='/home/ubunone/YJX/fenge/TMPNet/config/Lingjian/TMPNet.yaml', help='config file')
    parser.add_argument('opts', help='see config/s3dis/s3dis_pointtransformer_repro.yaml for all options', default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg

def dice_loss(pred, target, smooth=1e-5):
    pred = torch.softmax(pred, dim=1)
    target_one_hot = torch.zeros_like(pred).scatter_(1, target.unsqueeze(1), 1)
    intersection = (pred * target_one_hot).sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3)) + smooth)
    return 1 - dice.mean()

def get_logger():

    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def main():
    global args, logger
    args = get_parser()
    logger = get_logger()
    logger.info(args)
    assert args.classes > 1
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))


    if args.arch == 'PointTransformerV2mamba':
        from model.pointtransformer.PointTransformerV2mamba import PointTransformerV2mamba as Model
    else:
        raise Exception('architecture not supported yet'.format(args.arch))
    model = Model(in_channels=args.fea_dim, num_classes=args.classes).cuda()
    logger.info(model)
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()
    if os.path.isfile(args.model_path):
        logger.info("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path)
        state_dict = checkpoint['state_dict']
        new_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=True)
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.model_path, checkpoint['epoch']))
        args.epoch = checkpoint['epoch']
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))
    test(model, criterion)


def test(model, criterion):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    args.batch_size_test = 1
    model.eval()

    check_makedirs(args.save_folder)
    pred_save, label_save = [], []

    test_data = CustomDataset(split='test', root=args.data_root)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size_test, shuffle=False,
                                              num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)

    for idx, (coord, label, offset) in enumerate(test_loader):
        end = time.time()
        coord, label, offset = coord.cuda(non_blocking=True), label.cuda(non_blocking=True), offset.cuda(non_blocking=True)
        x0 = torch.zeros_like(coord).cuda(non_blocking=True)

        file_path = test_data.datapath[idx]
        item_name = os.path.splitext(os.path.basename(file_path))[0]
        pred_save_path = os.path.join(args.save_folder, f'{item_name}_pred.txt')
        label_save_path = os.path.join(args.save_folder, f'{item_name}_label.txt')

        with torch.no_grad():
            pred = model([coord, x0, offset])
            loss = criterion(pred, label)


        pred = pred.max(1)[1].data.cpu().numpy()
        label = label.cpu().numpy()

        np.savetxt(pred_save_path, np.hstack((coord.cpu().numpy(), pred.reshape(-1, 1))), fmt='%.6f %.6f %.6f %d')
        np.savetxt(label_save_path, np.hstack((coord.cpu().numpy(), label.reshape(-1, 1))), fmt='%.6f %.6f %.6f %d')

        pred_tensor = torch.from_numpy(pred).cuda()
        label_tensor = torch.from_numpy(label).cuda()

        intersection, union, target = intersectionAndUnionGPU(pred_tensor, label_tensor, args.classes, args.ignore_label)
        intersection_meter.update(intersection.cpu().numpy())
        union_meter.update(union.cpu().numpy())
        target_meter.update(target.cpu().numpy())

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        batch_time.update(time.time() - end)
        logger.info('Test: [{}/{}] '
                    'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    'Accuracy {accuracy:.4f}.'.format(idx + 1, len(test_loader), batch_time=batch_time,
                                                      accuracy=accuracy))
        pred_save.append(pred)
        label_save.append(label)

    with open(os.path.join(args.save_folder, "pred.pickle"), 'wb') as handle:
        pickle.dump({'pred': pred_save}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(args.save_folder, "label.pickle"), 'wb') as handle:
        pickle.dump({'label': label_save}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU1 = np.mean(iou_class)
    mAcc1 = np.mean(accuracy_class)
    allAcc1 = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    pred_concat = np.concatenate(pred_save)
    label_concat = np.concatenate(label_save)
    pred_tensor = torch.from_numpy(pred_concat).cuda()
    label_tensor = torch.from_numpy(label_concat).cuda()
    intersection, union, target = intersectionAndUnionGPU(pred_tensor, label_tensor, args.classes, args.ignore_label)
    iou_class = intersection.cpu().numpy() / (union.cpu().numpy() + 1e-10)
    accuracy_class = intersection.cpu().numpy() / (target.cpu().numpy() + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection.cpu().numpy()) / (sum(target.cpu().numpy()) + 1e-10)
    logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    logger.info('Val1 result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU1, mAcc1, allAcc1))

    for i in range(len(iou_class)):
        logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}'.format(i, iou_class[i], accuracy_class[i]))
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')


if __name__ == '__main__':
    main()
