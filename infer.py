import json
import yaml
from easydict import EasyDict as edict
import generate_ocr as go
import os
import torch.optim as optim
from torch.autograd import Variable
from warpctc_pytorch import CTCLoss
import numpy as np
import cv2
import random
import torch.nn as nn
import torch
import torch.nn.functional as F
import os
import unittest
import logging
from crnn import CRNN, weights_init
from train import decode, train_set

def main():
    conf_file = "conf/infer.yml"
    with open(conf_file, 'r') as f:
        args = edict(yaml.load(f))
    model_file = os.path.join(args.model_dir, args.model_file)

    '''
        50 - critical
        40 - error
        30 - warining
        20 - info
        10 - debug
    '''
    logging.getLogger().setLevel(args.debug_level)
    valset = train_set(args.test_root, 1, args.img_h, 
            args.img_w, args.n_class)
    words = valset.words

    with open(model_file, 'rb') as f:
        crnn = torch.load(f).cuda()
    logging.debug("crnn inited \n{}".format(crnn))
    img_path = args.img_to_infer
    logging.info("img to infer {}".format(img_path))
    img_np = cv2.imread(img_path)
    img_np = cv2.resize(img_np, (args.img_w, args.img_h))
    if args.show:
        cv2.imshow("pic", img_np)
        cv2.waitKey(0)
    img_np = np.expand_dims(img_np, axis=0)
    img_tensor = torch.from_numpy(
            img_np.transpose((0, 3, 1, 2))).float()
    preds = crnn(img_tensor.cuda())
    preds_size = Variable(torch.IntTensor([preds.size(0)] * 1))
    _, preds = preds.max(2)
    raw_preds = decode(preds.data, preds_size.data, 
        words, args.n_class, raw=True)
    real_preds = decode(preds.data, preds_size.data, 
        words, args.n_class, raw=False)
    logging.info("raw output {} real output {}".format(
        raw_preds, real_preds))

if __name__ == "__main__":
    main()
