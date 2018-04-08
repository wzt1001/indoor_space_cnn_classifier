# for a given single image, predict the class using the trained model
# Bolei Zhou
import os
import argparse
import numpy as np

import skimage.io
import torch
from torch.autograd import Variable as V
from torchvision import transforms as trn
from torch.nn import functional as F
import densenet
import pdb
from skimage.transform import rescale, resize, downscale_local_mean

trf_img = trn.Compose([trn.ToPILImage(), trn.ToTensor()])

model_file = 'work\\penn_station\\latest.pth'
# if not os.access(model_file, os.W_OK):
#     model_url = 'http://places.csail.mit.edu/scratch2/quickdraw/densenet.pytorch/work/densenet.base/latest.pth'
#     os.system('wget ' + model_url)

checkpoint = torch.load(model_file)
checkpoint.cpu()
checkpoint.eval()

# load the category list
with open('.\\categories.txt') as f:
    lines = f.readlines()
classes = [item.rstrip() for item in lines]

# test image
img = skimage.io.imread('531_1_1164.png')
img = resize(img, (128, 128))
img = img.reshape((1, 3, 128, 128)).astype(np.float32)
input_img = V(torch.from_numpy(img), volatile=True)

# forward pass
logit = checkpoint.forward(input_img)

# give the top 5 predictions
h_x = F.softmax(logit).data.squeeze()
probs, idx = h_x.sort(0, True)
for i in range(0,5):
    print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))
