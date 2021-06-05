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

    def inf_batch(self, image, label, clustered):
        image = image.cuda()
        label = label.cuda()#[:, :-6]
        
        with torch.no_grad():
            logit = self.net(image)

        pred = logit.max(dim=1)[1]

        clusters = np.unique(clustered)###* b, different clusters
        g = pred[0].cpu().numpy() #* copy of prediction
        
        clustered = np.pad(clustered, ((0, 6), (0, 0)), mode='edge') ###* pad bpd
        
        g[g == 0] = 255 ###* change ignore label for mask
        
        for cluster in clusters:
            mask = cluster == clustered ###* mask definition
            masked = g * mask ###* masking prediction
            
            # print(masked)#.shape)
            l, c = np.unique(masked, return_counts=True) #* nonzero cluster indices
            cs = np.argsort(c)[::-1] #* finding majority vote
            
            ind = 0
            if l[cs[ind]] == 255 and l[cs[ind]] == 0:
              ind += 2
            elif l[cs[ind]] == 255 or l[cs[ind]] == 0:
              ind += 1
            mv = l[cs[ind]]
            # if mv == 0:
            #   print(l, c, cs, mv)
            # print(mv * mask.shape)
            g[mask] = mv #* mask
            # print(np.unique(bpd))
            # print(np.unique(label[0].cpu().numpy()[mask]), mv)
            # print(np.unique(pred[0].cpu().numpy()[mask], return_counts=True), mv)

            # print(l, cs, c, mv)
            # hi

        g[g == 255] = 0
        # print(g.shape)
        # hi
        g = torch.tensor(g[None, :, :]).cuda()
        # print(g.shape)
        # hi
        self.hist += fast_hist(label, g)
        


def main(ckp_name='latest.pth'):
    sess = Session(dt_split='val')
    sess.load_checkpoints(ckp_name)
    dt_iter = sess.dataloader
    sess.net.eval()
    
    bpds = pickle.load(open("/content/drive/MyDrive/datasets/nyu/bpds.pkl", "rb"))

    for i, [image, label, name] in enumerate(dt_iter):
        bpd = bpds[name[0]]
        sess.inf_batch(image, label, bpd)
        score_dict = {'mIou': 0, 'fIoU': 0, 'pAcc': 0, 'mAcc': 0}
        if i % 10 == 0:
            logger.info('num-%d' % i)
            scores, cls_iu = cal_scores(sess.hist.cpu().numpy())
            for k, v in scores.items():
                logger.info('%s-%f' % (k, v))



    scores, cls_iu = cal_scores(sess.hist.cpu().numpy())
    for k, v in scores.items():
        logger.info('%s-%f' % (k, v))
    logger.info('')
    for k, v in cls_iu.items():
        logger.info('%s-%f' % (k, v))

    with open('/content/EMANet/super_result.txt', 'w') as f:
        f.write(scores, '\n', cls_iu)

if __name__ == '__main__':
    main()
