#!/usr/bin/env python3

import argparse
import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader, TensorDataset

import os
import sys
import math

import shutil

import setproctitle

import densenet
import make_graph

import numpy as np
import pdb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=32)
    parser.add_argument('--nEpochs', type=int, default=300)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--save', default='work\\penn_station')
    parser.add_argument('--split',default='C:\\data\\indoor_recognition\\data\\penn_station\\datasplit\\274_split_128.npy.npz')
    parser.add_argument('--nClasses', type=int, default=210)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.save = args.save or 'work\\densenet.base'
    setproctitle.setproctitle(args.save)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save)

    kwargs = {'num_workers': 3, 'pin_memory': True} if args.cuda else {}
    # TODO: There is not any data augmentation. Since we have enough training data, the data augmentation might be trivial.
    # TODO: The input data is from [0, 255] directly, since there is batch normalization layer.

    # load quickdraw datasplit
    data_quickdraw = np.load(args.split)
    data_train = data_quickdraw['data_train'].astype(np.float32)
    # data_train = data_quickdraw['data_train'].astype(np.float32)
    data_train = data_train.reshape((data_train.shape[0], 3, 128, 128))
    data_train = torch.from_numpy(data_train)
    label_train = data_quickdraw['label_train']
    label_train = torch.from_numpy(label_train.astype(int))
    data_val = data_quickdraw['data_val'].astype(np.float32)
    # data_val = data_quickdraw['data_val'].astype(np.float32)
    data_val = data_val.reshape((data_val.shape[0], 3, 128, 128))
    data_val = torch.from_numpy(data_val)
    label_val = data_quickdraw['label_val']
    label_val = torch.from_numpy(label_val.astype(int))

    print ('loaded quickdraw dataset')
    train_d = TensorDataset(data_train, label_train)
    trainLoader = DataLoader(train_d, batch_size=args.batchSz, shuffle=True)
    val_d = TensorDataset(data_val, label_val)
    testLoader = DataLoader(val_d, batch_size=args.batchSz, shuffle=False)


    net = densenet.DenseNet(growthRate=12, depth=100, reduction=0.5,
                            bottleneck=True, nClasses=args.nClasses)
    # net = densenet.DenseNet(growthRate=12, depth=100, reduction=0.5,
    #                         bottleneck=True, nClasses=args.nClasses)


    print('  + Number of params: {}'.format(
        sum([p.data.nelement() for p in net.parameters()])))
    if args.cuda:
        net = net.cuda()

    if args.opt == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=1e-1,
                            momentum=0.9, weight_decay=1e-4)
    elif args.opt == 'adam':
        optimizer = optim.Adam(net.parameters(), weight_decay=1e-4)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), weight_decay=1e-4)

    trainF = open(os.path.join(args.save, 'train.csv'), 'w')
    testF = open(os.path.join(args.save, 'test.csv'), 'w')

    err_best = 1000
    for epoch in range(1, args.nEpochs + 1):
        adjust_opt(args.opt, optimizer, epoch)
        train(args, epoch, net, trainLoader, optimizer, trainF)
        err_test = test(args, epoch, net, testLoader, optimizer, testF)
        torch.save(net, os.path.join(args.save, 'latest.pth'))
        if err_test<err_best:
            torch.save(net, os.path.join(args.save,'best.pth'))
            err_best = err_test

        os.system('./plot.py {} &'.format(args.save))

    trainF.close()
    testF.close()

def train(args, epoch, net, trainLoader, optimizer, trainF):
    net.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    for batch_idx, (data, target) in enumerate(trainLoader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = net(data)
        loss = F.nll_loss(output, target)
        # make_graph.save('/tmp/t.dot', loss.creator); assert(False)
        loss.backward()
        optimizer.step()
        nProcessed += len(data)
        pred = output.data.max(1)[1] # get the index of the max log-probability
        incorrect = pred.ne(target.data).cpu().sum()
        err = 100.*incorrect/len(data)
        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tError: {:.6f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
            loss.data[0], err))

        trainF.write('{},{},{}\n'.format(partialEpoch, loss.data[0], err))
        trainF.flush()

def test(args, epoch, net, testLoader, optimizer, testF):
    net.eval()
    test_loss = 0
    incorrect = 0
    for data, target in testLoader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = net(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        incorrect += pred.ne(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(testLoader) # loss function already averages over batch size
    nTotal = len(testLoader.dataset)
    err = 100.*incorrect/nTotal
    print('\nTest set: Average loss: {:.4f}, Error: {}/{} ({:.0f}%)\n'.format(
        test_loss, incorrect, nTotal, err))

    testF.write('{},{},{}\n'.format(epoch, test_loss, err))
    testF.flush()
    return err

def adjust_opt(optAlg, optimizer, epoch):
    if optAlg == 'sgd':
        if epoch < 80: lr = 1e-1
        elif epoch == 120: lr = 1e-2
        elif epoch == 160: lr = 1e-3
        else: return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

if __name__=='__main__':
    main()
