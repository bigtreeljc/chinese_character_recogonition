import json
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
import torchloop.dataset.ocr.generate_ocr as go
from crnn import CRNN, weights_init
# from torchloop.util import fs_utils, tl_logging
logging.basicConfig(level=logging.DEBUG)

def default_bg_dir():
    return "data/bgs"

def default_font_dir():
    return "data/chn_fonts"

def default_font_dir():
    return "data/chn_fonts"

class test(unittest.TestCase):
    def test1(self):
        bg_dir = default_bg_dir()
        pic_pool = go.picture_pool(bg_dir)
        sampled_bg = pic_pool.sample_bg_file(2)
        print("sampled bgs {}".format(sampled_bg))

    def test2(self):
        font_dir = default_font_dir()
        font_pool = go.font_pool(font_dir)
        sampled_font = font_pool.sample_font_file(2)
        print("sampled fonts {}".format(sampled_font))

    def test3(self):
        '''
            some constants
        '''
        n_sampled: int = 1
        txt_to_draw_: str = "你好，李攀"
        ch_h = 64
        '''
            sampling bg
        '''
        bg_dir = default_bg_dir()
        pic_pool = go.picture_pool(bg_dir)
        sampled_bg = pic_pool.sample_bg_file(n_sampled)
        '''
            sampling font
        '''
        font_dir = default_font_dir()
        font_pool = go.font_pool(font_dir)
        sampled_font = font_pool.sample_font_file(n_sampled)
        '''
            testing text drawer
        '''
        bg_file: str = sampled_bg[0]
        font_file: str = sampled_font[0]
        text_drawer_o = go.font_drawer(font_file, bg_file, ch_size=ch_h)
        text_drawer_o.draw_sample_gray_bg(txt_to_draw_, if_show=True)
        '''
            testing colored background
        '''
        text_drawer_o.draw_sample_colored(txt_to_draw_, 
                if_show=True)

    def test4(self):
        '''
            some constants
        '''
        n_sampled: int = 1
        txt_to_draw_: str = "你好，李攀"
        ch_h = 64
        '''
            sampling bg
        '''
        bg_dir = default_bg_dir()
        pic_pool = go.picture_pool(bg_dir)
        sampled_bg = pic_pool.sample_bg_file(n_sampled)
        '''
            sampling font
        '''
        font_dir = default_font_dir()
        font_pool = go.font_pool(font_dir)
        sampled_font = font_pool.sample_font_file(n_sampled)
        '''
            testing text drawer
        '''
        bg_file: str = sampled_bg[0]
        font_file: str = sampled_font[0]
        text_drawer_o = go.font_drawer(font_file, bg_file, ch_size=ch_h)
        text_drawer_o.draw_sample_colored_bg(txt_to_draw_,
                if_show=True, show_bg=False, show_cropped=True)

    def test5(self):
        print("test5")
        corpus_dir = "data/chn_corpus/corpus/"
        file_count = 1

        collector = go.corpus(corpus_dir, file_count)
        collector.sample_text()

    def test_chn_subset(self):
        
        chn_map_path = "data/chn_corpus/word_freq.json"
        sample_size = 30
        chn_set = go.chn_subset(chn_map_path)
        sample_words = chn_set.sample_words(sample_size)
        print("sampled_words {} len sampled {}".format(sample_words, 
            len(sample_words)))

    def test_generator(self):
        n_bg = n_font = 1
        ch_h = 32
        n_pic = 50
        sample_size = 10
        min_len = 4
        max_len = 6
        bg_dir = default_bg_dir()
        font_dir = default_font_dir()

        target_dir = "data/ocr_dataset_train_50_10_val"
        os.makedirs(target_dir, exist_ok=True)
        chn_map_path = "data/chn_corpus/word_freq.json"
        chn_set = go.chn_subset(chn_map_path)
        generator = go.chn_ocr_generator(chn_set, n_chars=sample_size)
        generator.generate(target_dir, bg_dir, font_dir, n_bg,
            n_font, n_pictures=n_pic, ch_h=ch_h, min_len=min_len, 
            max_len=max_len)
        generator.save_vocab(target_dir)

    def test_chn_char(self):
        txt_to_draw_: str = "你好，李攀"
        txt_to_draw_2: str = 'aaaa'
        txt_to_draw_3: str = 'aaaa李攀'
        result = go.check_contain_chinese(txt_to_draw_)
        print("{} \nresult {}".format(txt_to_draw_, result))
        result = go.check_contain_chinese(txt_to_draw_2)
        print("{} \nresult {}".format(txt_to_draw_2, result))
        result = go.check_contain_chinese(txt_to_draw_3)
        print("{} \nresult {}".format(txt_to_draw_3, result))
        result = go.is_chn_char("好")
        print("result {}".format(result))

    def test_edit_dist(self):
        # str1 = "的她她他"
        # str2 = "是人人的"
        str1 = "他个个人我"
        str2 = "他人个人那"
        str1 = ""
        str2 = "asdf"

        print(str1, str2)
        ed = edit_distance(str1, str2)
        print("edit distance {}".format(ed))

    def test_train(self):
        '''
        parameters of train
        '''
        # test_root = "data/ocr_dataset_val"
        # train_root = "data/ocr_dataset"
        train_root = "data/ocr_dataset_train_400_10/"
        test_root = "data/ocr_dataset_train_50_10_val/"
        batch_size = 20
        max_len = 15
        img_h, img_w = 32, 150
        n_hidden = 512
        n_iter = 400
        lr = 0.00005
        cuda = True
        val_interval = 250
        save_interval = 1000
        model_dir = "models"
        debug_level = 20
        experiment = "experiment"
        n_channel = 3
        n_class = 11
        beta = 0.5

        image = torch.FloatTensor(batch_size, n_channel, img_h, img_h)
        text = torch.IntTensor(batch_size * max_len)
        length = torch.IntTensor(batch_size)

        logging.getLogger().setLevel(debug_level)
        '''
            50 - critical
            40 - error
            30 - warining
            20 - info
            10 - debug
        '''
        crnn = CRNN(img_h, n_channel, n_class, n_hidden).cuda()
        crnn.apply(weights_init)

        criterion = CTCLoss().cuda()

        optimizer = optim.RMSprop(crnn.parameters(), lr=lr)
        # optimizer = optim.Adam(crnn.parameters(), lr=lr,
        #                    betas=(beta, 0.999))

        trainset = train_set(train_root, batch_size, img_h, img_w, n_class)
        valset = train_set(test_root, batch_size, img_h, img_w, n_class)

        cur_iter = 0
        for ITER in range(n_iter):
            for train_img, train_label, train_lengths, batch_label in iter(trainset):
                for p in crnn.parameters():
                    p.requires_grad = True
                crnn.train()

                if train_img is None:
                    break
                cur_iter += 1
                loadData(image, train_img)
                loadData(text, train_label)
                loadData(length, train_lengths)
                preds = crnn(train_img.cuda())
                # preds = F.softmax(preds, dim=2)
                # print(preds.shape)
                preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
                # print(batch_label, text, length, len(text), len(length), length.sum(), 
                #     preds.shape, preds_size.shape)
                cost = criterion(preds, text, preds_size, length) / batch_size
                crnn.zero_grad()
                cost.backward()
                optimizer.step()
                print("training-iter {} cost {}".format(ITER, 
                    cost.cpu().detach().numpy()[0]))
                if cur_iter % val_interval == 0:
                    val(crnn, valset, criterion, n_class)
                if cur_iter % save_interval == 0:
                    model_file = os.path.join(model_dir, "crnn_iter{}.pth".format(
                        ITER))
                    print("saving in file {}".format(model_file))
                    with open(model_file, 'wb') as f:
                        torch.save(crnn, f)

def loadData(v, data):
    v.data.resize_(data.size()).copy_(data)

def val(net, valset, criterion, n_class, val_iter=10):
    print("eval...")
    for p in net.parameters():
        p.requires_grad = False
    net.eval()
    
    n_iter = min(val_iter, len(valset))
    words = valset.words
    loss_avg = averager()
    n_correct = 0
    n_correct_ed = 0
    # print(words, len(words))

    for cur_iter in range(n_iter):
        train_img, train_label, train_lengths, batch_label = valset.get_batch()
        # print(train_label)
        preds = net(train_img.cuda())
        # print(preds)
        batch_size = preds.size(1)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, train_label, preds_size, train_lengths) / batch_size
        loss_avg.add(cost)

        _, preds = preds.max(2)
        # print(preds)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        raw_preds = decode(preds.data, preds_size.data, 
            words, n_class, raw=True)
        real_preds = decode(preds.data, preds_size.data, 
            words, n_class, raw=False)
        # print(real_preds, batch_label)
        for pred, target in zip(real_preds, batch_label):
            if pred == target:
                n_correct += 1
            '''
                edit distance
            '''
            len_gt = len(target)
            edit_dist = edit_distance(pred, target)
            n_correct_ed += 1 - edit_dist/len_gt

    for raw_pred, pred, gt in zip(raw_preds, real_preds, batch_label):
        print("{} ==> {}. gt {}".format(raw_pred, pred, gt))

    accuracy = n_correct / float(n_iter * batch_size)
    accuracy_ed = n_correct_ed / float(n_iter * batch_size)
    print("test loss: {}, accuracy {} accuracy ed {}".format(
        loss_avg.val(), accuracy, accuracy_ed))

def edit_distance(str1, str2):
    m, n = len(str1) + 1, len(str2) + 1
    matrix = [[0]*n for __ in range(m)]
    matrix[0][0] = 0

    for i in range(1, m):
        matrix[i][0] = matrix[i-1][0] + 1

    for j in range(1, n):
        matrix[0][j] = matrix[0][j-1] + 1

    cost = 0
    for i in range(1, m):
        for j in range(1, n):
            if str1[i-1] == str2[j-1]:
                cost = 0
            else:
                cost = 1
            matrix[i][j] = min(matrix[i-1][j]+1, matrix[i][j-1]+1,
                matrix[i-1][j-1]+cost) 
    # print(matrix)
    return matrix[m-1][n-1]

def gather(l, inds):
    gathered = [l[ind] for ind in inds]
    return gathered

def decode(t, length, words, n_class, raw=False):
    assert t.numel() == length.sum(), "t {} length {}".format(
        t.numel(), length.sum())
    # print(t, length)
    if length.numel() == 1:
        # print(t.shape)
        length = length[0]
        if raw:
            return ''.join(words[i] for i in t)
        else:
            char_list = []
            for i in range(length):
                if t[i] != 0 and (not (i > 0 and t[i-1] == t[i])):
                    char_list.append(words[t[i]])
            return ''.join(char_list)
        
    texts = []
    index = 0
    for i in range(length.numel()):
        l = length[i]
        texts.append(decode(t[index: index+l], torch.IntTensor([l]),
            words, n_class, raw=raw))
        index += l
    return texts

class train_set:
    def __init__(self, train_dir, batch_size, img_h, img_w, n_labels):
        img_paths = os.listdir(train_dir)
        img_paths = list(filter(lambda x: x.endswith("jpg"), img_paths))
        self.labels = list(map(lambda x: os.path.splitext(x)[0], img_paths))
        self.img_paths = list(map(lambda x: os.path.join(train_dir, x), img_paths))
        self.all_inds = list(range(len(self.img_paths)))
        self.batch_size = batch_size
        self.img_h, self.img_w = img_h, img_w
        self._init_labels(train_dir)

    def _init_labels(self, train_dir):
        label_file = os.path.join(train_dir, "vocab.json")
        with open(label_file, 'r') as f:
            words = json.load(f)
        words.insert(0, '-')
        # print("vocab {}".format(words))
        self.w2id = {}
        for id_, word in enumerate(words):
            self.w2id[word] = id_
        self.words = words
        print(self.w2id, words)

    def __len__(self):
        return len(self.img_paths)

    def __iter__(self):
        self.shuffled_inds = self.all_inds
        random.shuffle(self.shuffled_inds)
        self.cur_ind = 0
        return self

    def __next__(self):
        if self.cur_ind >= self.__len__():
            return None, None, None, None
        inds = self.shuffled_inds[self.cur_ind: self.cur_ind+self.batch_size]
        self.cur_ind += self.batch_size
        batch_img = gather(self.img_paths, inds)
        batch_label = gather(self.labels, inds)
        img_tensor = self._batch_img_to_tensor(batch_img)
        label_tensor, length_tensor = self._batch_label_to_tensor(batch_label)
        return img_tensor, label_tensor, length_tensor, batch_label

    def get_batch(self):
        shuffled_inds = self.all_inds
        inds = random.sample(shuffled_inds, k=self.batch_size)
        batch_img = gather(self.img_paths, inds)
        batch_label = gather(self.labels, inds)
        img_tensor = self._batch_img_to_tensor(batch_img)
        label_tensor, length_tensor = self._batch_label_to_tensor(batch_label)
        return img_tensor, label_tensor, length_tensor, batch_label

    def _batch_img_to_tensor(self, batch_img):
        def read_and_resize(img_file):
            img_np = cv2.imread(img_file)
            img_norm = cv2.resize(img_np, (self.img_w, self.img_h))
            return img_norm

        batch_img = np.array(list(map(lambda x: read_and_resize(x), batch_img)))
        img_tensor = torch.from_numpy(batch_img.transpose((0, 3, 1, 2))).float()
        return img_tensor

    def _batch_label_to_tensor(self, batch_label):
        lengths = np.array(list(map(lambda x: len(x), batch_label))).astype(np.int)
        len_tensor = torch.from_numpy(lengths).int()
        def sentence2ids(sentence):
            sentence_np = np.array([self.w2id[c] for c in sentence]).astype(np.int)
            sentence_tensor = torch.from_numpy(sentence_np)
            return sentence_tensor
        ids = list(map(lambda x: sentence2ids(x), batch_label))
        label_tensor = torch.cat(ids).int()
        return label_tensor, len_tensor

class averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


if __name__ == "__main__":
    unittest.main()
