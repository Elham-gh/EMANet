import os
import os.path as osp

import numpy as np
import pickle
import json

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from dataset import ValDataset 
from metric import fast_hist, cal_scores
from network import EMANet 
import settings
from PIL import Image

logger = settings.logger


class Session:
    def __init__(self, dt_split):
        torch.cuda.set_device(settings.DEVICE)

        self.log_dir = settings.LOG_DIR
        self.model_dir = settings.MODEL_DIR

        self.net = EMANet(settings.N_CLASSES, settings.N_LAYERS).cuda()
        self.net = DataParallel(self.net, device_ids=[settings.DEVICE])
        dataset = ValDataset(split=dt_split)
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=False, 
                                     num_workers=2, drop_last=False)
        self.hist = 0

    def load_checkpoints(self, name):
        ckp_path = osp.join(self.model_dir, name)
        try:
            obj = torch.load(ckp_path, 
                             map_location=lambda storage, loc: storage.cuda())
            logger.info('Load checkpoint %s.' % ckp_path)
        except FileNotFoundError:
            logger.info('No checkpoint %s!' % ckp_path)
            return

        self.net.module.load_state_dict(obj['net'])

    def inf_batch(self, image, label):
        image = image.cuda()
        label = label.cuda()
        with torch.no_grad():
            logit = self.net(image)

        pred = logit.max(dim=1)[1]
        self.hist += fast_hist(label, pred)
        

def main(ckp_name='models/final.pth'):
    sess = Session(dt_split='val')
    sess.load_checkpoints(ckp_name)
    dt_iter = sess.dataloader
    sess.net.eval()
    
    score_dict = {'mIou': 0, 'fIoU': 0, 'pAcc': 0, 'mAcc': 0}

    for i, [image, label, _] in enumerate(dt_iter):
        sess.inf_batch(image, label)
        if i % 10 == 0:
            logger.info('num-%d' % i)
            scores, cls_iu = cal_scores(sess.hist.cpu().numpy())
            score_dict['mIou'] += score_dict['mIou']
            score_dict['fIoU'] += score_dict['fIoU']
            score_dict['pAcc'] += score_dict['pAcc']
            score_dict['mAcc'] += score_dict['mAcc']
            for k, v in scores.items():
                logger.info('%s-%f' % (k, v))

    print(score_dict)

    with open('/content/EMANet/result.txt', 'w') as f:
        f.write(json.dumps(score_dict))

    scores, cls_iu = cal_scores(sess.hist.cpu().numpy())
    for k, v in scores.items():
        logger.info('%s-%f' % (k, v))
    logger.info('')
    for k, v in cls_iu.items():
        logger.info('%s-%f' % (k, v))


if __name__ == '__main__':
    main()
