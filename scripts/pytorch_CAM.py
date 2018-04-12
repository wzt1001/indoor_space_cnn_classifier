# simple implementation of CAM in PyTorch for the networks such as ResNet, DenseNet, SqueezeNet, Inception

import io
import os
# import requests
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import torch.nn as nn
import torch
import numpy as np
import cv2
import psycopg2

# input image
# LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
# IMG_URL = '531_1_1164.png'

def returnCAM(feature_conv, weight_softmax, class_idx):
	# generate the class activation maps upsample to 256x256
	# size_upsample = (256, 256)
	size_upsample = (128, 128)
	bz, nc, h, w = feature_conv.shape
	output_cam = []
	for idx in class_idx:
		cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
		cam = cam.reshape(h, w)
		cam = cam - np.min(cam)
		cam_img = cam / np.max(cam)
		cam_img = np.uint8(255 * cam_img)
		output_cam.append(cv2.resize(cam_img, size_upsample))
	return output_cam


# networks such as googlenet, resnet, densenet already use global average pooling at the end, so CAM could be used directly.
model_id = 3
if model_id == 1:
	net = models.squeezenet1_1(pretrained=True)
	finalconv_name = 'features' # this is the last conv layer of the network
elif model_id == 2:
	net = models.resnet18(pretrained=True)
	finalconv_name = 'layer4'
elif model_id == 3:
	net = models.densenet121(pretrained=False, num_classes=240)
	finalconv_name = 'features'


if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    net = nn.DataParallel(net, device_ids=None)
net.cuda()

checkpoint = torch.load('ind_net/output/checkpoint.pth_bak.tar')
best_prec1 = checkpoint['best_prec1']
net.load_state_dict(checkpoint['state_dict'])

net.eval()
# print(net.module)

# hook the feature extractor
features_blobs = []
def hook_feature(module, input, output):
	features_blobs.append(output.data.cpu().numpy())

# !!!important change  
net.module._modules.get(finalconv_name).register_forward_hook(hook_feature)

# net._modules.get(finalconv_name).register_forward_hook(hook_feature)
# get the softmax weight
params = list(net.parameters())
weight_softmax = np.squeeze(params[-2].cpu().data.numpy())


# normalize = transforms.Normalize(
#    mean=[0.485, 0.456, 0.406],
#    std=[0.229, 0.224, 0.225]
# )

preprocess = transforms.Compose([
   transforms.Scale((224, 224)),
   transforms.ToTensor()
])

# classes = {int(key):value for (key, value)
#	   in requests.get(LABELS_URL).json().items()}
with open('./ind_net/categories.txt') as f:
	lines = f.readlines()
classes = [item.rstrip() for item in lines]


conn_string = "host='localhost' dbname='indoor_position' user='postgres' password='tiancai' port='5432'"
conn = psycopg2.connect(conn_string)
cur = conn.cursor()

query = '''select path, spec_id, id, image_name, cam_id, clip_id, clip_count from penn_station.image_lookup_1000ms'''
cur.execute(query)
result = cur.fetchall()
cur.close()
conn.commit()


# adding new columns
# conn = psycopg2.connect(conn_string)
# cur = conn.cursor()

# query = '''ALTER TABLE penn_station.image_lookup_1000ms ADD COLUMN pred0 text, ADD COLUMN pred1 text, ADD COLUMN pred2 text, ADD COLUMN pred3 text, ADD COLUMN pred4 text, ADD COLUMN prob0 double precision, ADD COLUMN prob1 double precision, ADD COLUMN prob2 double precision, ADD COLUMN prob3 double precision, ADD COLUMN prob4 double precision;'''
# cur.execute(query)
# cur.close()
# conn.commit()

for idx, item in enumerate(result):

	# if idx < 100:
	# 	continue
	img_path = item[0]
	img_spec_id = item[1]
	img_id   = item[2]
	img_name = item[3]
	cam_id   = item[4]
	clip_id  = item[5]
	clip_count = item[6]

	if not os.path.exists(os.path.join(os.getcwd(), "..", "web", "final", "images", str(cam_id), "original")):
		os.makedirs(os.path.join(os.getcwd(), "..", "web", "final", "images", str(cam_id), "original"))

	if not os.path.exists(os.path.join(os.getcwd(), "..", "web", "final", "images", str(cam_id), "heatmap")):
		os.makedirs(os.path.join(os.getcwd(), "..", "web", "final", "images", str(cam_id), "heatmap"))

	img_pil = Image.open(img_path)
	img_pil.save(os.path.join(os.getcwd(), "..", "web", "final", "images", str(cam_id), "original", str(clip_id) + "_" + str(clip_count) + '.png'))

	img_tensor = preprocess(img_pil)
	img_variable = Variable(img_tensor.unsqueeze(0))
	logit = net(img_variable * 224)

	h_x = F.softmax(logit).data.squeeze()
	probs, idx = h_x.sort(0, True)
	# print(logit.data.max(1, keepdim=True)[1])
	# print(classes[np.asscalar(logit.data.max(1, keepdim=True)[1].cpu().np())])
	# output the prediction
	print("correct id: " + str(img_spec_id))
	for i in range(0, 5):
		print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

	conn = psycopg2.connect(conn_string)
	cur = conn.cursor()

	query = '''UPDATE penn_station.image_lookup_1000ms SET pred0 = '%s', pred1 = '%s', pred2 = '%s', pred3 = '%s', pred4 = '%s', prob0 = %s, prob1 = %s, prob2 = %s, prob3 = %s, prob4 = %s WHERE image_name = '%s'; ''' % (classes[idx[0]], classes[idx[1]], classes[idx[2]], classes[idx[3]], classes[idx[4]], probs[0], probs[1], probs[2], probs[3], probs[4], img_name)
	cur.execute(query)
	cur.close()
	conn.commit()

	# generate class activation mapping for the top1 prediction
	# print(features_blobs[0].shape, weight_softmax.shape, [idx[0]])
	CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

	# render the CAM and output
	print('output CAM.jpg for the top1 prediction: %s' % classes[idx[0]])
	img = cv2.imread(img_path)
	height, width, _ = img.shape
	heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
	result = heatmap * 0.3 + img * 0.5
	cv2.imwrite(os.path.join(os.getcwd(), "..", "web", "final", "images", str(cam_id), "heatmap", str(clip_id) + "_" + str(clip_count) + '.png'), result)
