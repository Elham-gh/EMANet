import os
import os.path as osp

import numpy as np

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

    def inf_batch(self, image, label, name, bpd):
        image = image.cuda()
        label = label.cuda()
        with torch.no_grad():
            logit = self.net(image)

        pred = logit.max(dim=1)[1]
        self.hist += fast_hist(label, pred)
        print(pred)
        hi
        # output = pred.cpu().numpy()
        # print(output.max())
        # print(output.min())
        # output = Image.fromarray(output)
        # output.save('/content/EMANet/outputs/' + name + '.jpg')
        out = np.array(pred.cpu(), dtype=np.uint8)
        maximum = out.max()
        extended = (out / maximum * 255).astype('uint8')
        extended = np.squeeze(extended, axis=0)
        # seg_pred = np.asarray(np.argmax(out, axis=0), dtype=np.uint8)
        output_im = Image.fromarray(extended)
        output_im.save('/content/EMANet/outputs/'+ name[0] +'.png')
        print(np.array(output_im).min())
        hi

def main(ckp_name='latest.pth'):
    sess = Session(dt_split='val')
    sess.load_checkpoints(ckp_name)
    dt_iter = sess.dataloader
    sess.net.eval()
    import pickle
    
    bpds = pickle.load(open("/content/drive/MyDrive/nyu/bpds.pkl", "rb"))

    for i, [image, label, name] in enumerate(dt_iter):
        bpd = bpds[name[0]]
        print(bpd)
        hi
        sess.inf_batch(image, label, name, bpds)
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


if __name__ == '__main__':
    main()
