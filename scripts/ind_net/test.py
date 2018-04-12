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
import pdb
from skimage.transform import rescale, resize, downscale_local_mean

trf_img = trn.Compose([trn.ToPILImage(), trn.ToTensor()])

model_file = 'output/checkpoint.pth_bak.tar'
# if not os.access(model_file, os.W_OK):
#     model_url = 'http://places.csail.mit.edu/scratch2/quickdraw/densenet.pytorch/work/densenet.base/latest.pth'
#     os.system('wget ' + model_url)

def test(test_loader, model, names, classes):
    """Test the model on the Evaluation Folder
    Args:
        - classes: is a list with the class name
        - names: is a generator to retrieve the filename that is classified
    """
    # switch to evaluate mode
    model.eval()
    # Evaluate all the validation set
    for i, (input, _) in enumerate(test_loader):
        if cuda:
            input = input.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)

        # compute output
        output = model(input_var)
        # Take last layer output
        if isinstance(output, tuple):
            output = output[len(output)-1]

        # print (output.data.max(1, keepdim=True)[1])
        lab = classes[np.asscalar(output.data.max(1, keepdim=True)[1].cpu().np())]
        print ("Images: " + next(names) + ", Classified as: " + lab)

checkpoint = torch.load(model_file)
checkpoint = torch.nn.DataParallel(checkpoint, device_ids=None)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    checkpoint = torch.nn.DataParallel(checkpoint, device_ids=None)

checkpoint.cuda()
checkpoint.eval()

# load the category list
with open('./categories.txt') as f:
    lines = f.readlines()
classes = [item.rstrip() for item in lines]

# test image
img = skimage.io.imread('/media/ztwang/ssd_0/data/indoor_space_cnn_classifier/web/final/0/orignal/02.png')
img = resize(img, (224, 224))
img = img.reshape((1, 3, 224, 224)).astype(np.float32)
input_img = V(torch.from_numpy(img).unsqueeze(0), volatile=True)

# forward pass
logit = checkpoint(input_img)

# give the top 5 predictions
h_x = F.softmax(logit).data.squeeze()
probs, idx = h_x.sort(0, True)
for i in range(0,5):
    print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))
