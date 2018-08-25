import torch.nn.functional as F
import torch.nn as nn
from typing import List

# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn: int, nHidden: int, nOut: int):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        ######
        # h is the base size
        ######
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output

class CRNN(nn.Module):
    def __init__(self, 
            imgH: int, nc: int, nclass: int, 
            nh: int, n_rnn: int=2, leakyRelu: bool=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input_):
        # conv features
        # print("input shape {}".format(input.size()))
        conv = self.cnn(input_)
        b, c, h, w = conv.size()
        # print("conved b {} c {} h {} w {}".format(
        #    b, c, h, w))
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        # print("conv squeezeed and permuted {}".format(
        #    conv.size()))
        # rnn features
        output = self.rnn(conv)
        # print("rnn output {}".format(output.size()))
        # output = F.softmax(output, dim=2)

        return output

def trainBatch(net, criterion, optimizer):
    data = train_iter.next()
    cpu_images, cpu_texts = data
    # img_np = cpu_images.numpy()[0].transpose((1, 2, 0))
    # cv2.imshow("window", img_np)
    # cv2.waitKey(0)
    batch_size = cpu_images.size(0)
    logging.debug("cpu_images {}, cpu_texts {} batch size {}".\
        format(cpu_images.shape, len(cpu_texts), batch_size))
    utils.loadData(image, cpu_images)
    # print("cpu texts {}".format(cpu_texts))

    t, l = converter.encode(cpu_texts)
    logging.debug("encoded t {} l {}".format(
        t, l))
    utils.loadData(text, t)
    utils.loadData(length, l)

    preds = crnn(image)
    # print("preds {}".format(preds.shape))
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    cost = criterion(preds, text, preds_size, length) / batch_size
    crnn.zero_grad()
    cost.backward()
    optimizer.step()
    return cost

