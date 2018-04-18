# simple implementation of CAM in PyTorch for the networks such as ResNet, DenseNet, SqueezeNet, Inception

import io
import os
import csv
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
from scipy.io import loadmat
# import matplotlib
# matplotlib.use('agg')
# import matplotlib.pyplot as plt


colors = loadmat('./semantic-segmentation-pytorch/data/color150.mat')['colors'].tolist()
print(colors)
sem_categories = []
sem_content	= []
sem_dic		= {}

with open('./semantic-segmentation-pytorch/data/object150_info.csv', 'rb') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
	cnt = 0
	for row in spamreader:
		if cnt == 0:
			cnt += 1
			continue
		cnt += 1
		sem_categories.append(row[0].split(',')[0])
		sem_content.append(row[0].split(',')[5])
		sem_dic[sem_categories[-1]] = sem_content[-1]

with open('./ind_net/categories.txt') as f:
	lines = f.readlines()
	classes = [item.rstrip() for item in lines]

def colorDecode(labelmap, mode='BGR'):
	labelmap = labelmap.astype('int')
	code = np.zeros((labelmap.shape[0], labelmap.shape[1]), dtype=np.uint8)
	for i in range(labelmap.shape[0]):
		for j in range(labelmap.shape[1]):
			pixel = labelmap[i, j, :].tolist()[::-1]
			code[i][j] = colors.index(pixel)
	print(code.shape)

	return code
	# plt.hist(code, bins='auto')
	# plt.title("Histogram with 'auto' bins")
	# plt.show()


def unique(dec_map, cam_map):
	dic = {}
	print(dec_map.shape)
	for i in range(dec_map.shape[0]):
		if dec_map[i] in dic:
			dic[dec_map[i]] += cam_map[i]
		else:
			dic[dec_map[i]] = cam_map[i]
	return dic

conn_string = "host='localhost' dbname='indoor_position' user='postgres' password='tiancai' port='5432'"
conn = psycopg2.connect(conn_string)
cur = conn.cursor()

query = '''select path, spec_id, id, image_name, cam_id, clip_id, clip_count from penn_station.image_lookup_1000ms where cam_id = 1 or cam_id = 2 or cam_id = 3 order by spec_id'''
cur.execute(query)
result = cur.fetchall()
cur.close()
conn.commit()

for index, item in enumerate(result):

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

	checkpoint = torch.load('ind_net/output/checkpoint.pth_bak2.tar')
	best_prec1 = checkpoint['best_prec1']
	net.load_state_dict(checkpoint['state_dict'])

	net.eval()
	# print(net.module)

	# normalize = transforms.Normalize(
	#	mean=[0.485, 0.456, 0.406],
	#	std=[0.229, 0.224, 0.225]
	# )

	# hook the feature extractor
	features_blobs = []
	def hook_feature(module, input, output):
		features_blobs.append(output.data.cpu().numpy())

	# !!!important change  
	net.module._modules.get(finalconv_name).register_forward_hook(hook_feature)
	print(len(features_blobs))
	# net._modules.get(finalconv_name).register_forward_hook(hook_feature)

	preprocess = transforms.Compose([
	   transforms.ToTensor()
	])

	# classes = {int(key):value for (key, value)
	#	   in requests.get(LABELS_URL).json().items()}

	# adding new columns
	# conn = psycopg2.connect(conn_string)
	# cur = conn.cursor()

	# query = '''ALTER TABLE penn_station.image_lookup_1000ms ADD COLUMN pred0 text, ADD COLUMN pred1 text, ADD COLUMN pred2 text, ADD COLUMN pred3 text, ADD COLUMN pred4 text, ADD COLUMN prob0 double precision, ADD COLUMN prob1 double precision, ADD COLUMN prob2 double precision, ADD COLUMN prob3 double precision, ADD COLUMN prob4 double precision, ADD COLUMN cat0 text, ADD COLUMN cat1 text, ADD COLUMN cat2 text, ADD COLUMN cat3 text, ADD COLUMN cat4 text, ADD COLUMN pct0 double precision, ADD COLUMN pct1 double precision, ADD COLUMN pct2 double precision, ADD COLUMN pct3 double precision, ADD COLUMN pct4 double precision, ADD COLUMN seg_id text, ADD COLUMN seg_tt text, ADD COLUMN seg_per text;'''
	# cur.execute(query)
	# cur.close()
	# conn.commit()

	# data_indoor = np.load('/home/ztwang/ssd_data/data/indoor_space_cnn_classifier/data/penn_station/datasplit/274_split_244_test.npy.npz')
	# data_val = data_indoor['data_val'].astype(np.float32)
	# data_val = data_indoor['data_val'].astype(np.float32)
	# data_val = data_val.reshape((data_val.shape[0], 3, 224, 224))
	# data_val = data_val[9, :, :, :]
	# data_val = np.transpose(data_val, (2, 1, 0))
	# label_val = data_indoor['label_val']
	# label_val = torch.from_numpy(label_val.astype(int))


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

	img_pil = cv2.resize(np.array(Image.open(img_path)), dsize=(224, 224), interpolation=cv2.INTER_CUBIC) 
	# img_pil = Image.open(img_path)
	# img_pil.save(os.path.join(os.getcwd(), "..", "web", "final", "images", str(cam_id), "original", str(clip_id) + "_" + str(clip_count) + '.png'))
	img_pil = img_pil.reshape(1, 150528).reshape((1, 3, 224, 224))[0, :, :, :].transpose(1, 2, 0)

	img_tensor = preprocess(img_pil)
	img_variable = Variable(img_tensor.unsqueeze(0))
	logit = net(img_variable * 255)

	# get the softmax weight
	params = list(net.parameters())
	weight_softmax = np.squeeze(params[-2].cpu().data.numpy())

	h_x = F.softmax(logit).data.squeeze()
	probs, idx = h_x.sort(0, True)
	# print(logit.data.max(1, keepdim=True)[1])
	# print(classes[np.asscalar(logit.data.max(1, keepdim=True)[1].cpu().np())])
	# output the prediction
	print("correct id: " + img_spec_id + ", top1 id:" + str(idx[0]) + " " + classes[idx[0]])
	for i in range(0, 5):
		print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

	# generate class activation mapping for the top1 prediction
	# print(features_blobs[0].shape, weight_softmax.shape, [idx[0]])

	# print(features_blobs[0])
	CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

	# render the CAM and output
	# print('output CAM.jpg for the top1 prediction: %s' % classes[idx[0]])
	img = cv2.imread(img_path)
	height, width, _ = img.shape

	seg_dir = os.path.join(os.getcwd(), "..", "web", "final", "images", str(cam_id), "segmentation")
	seg_img = cv2.imread(os.path.join(seg_dir, str(clip_id) + "_" + str(clip_count)+ '.png'))
	seg_unique, seg_counts = np.unique(colorDecode(seg_img).reshape(-1), return_counts=True)
	seg_map = {}
	for idx0, item in enumerate(seg_unique):
		seg_map[item] = seg_counts[idx0]

	cam_map = cv2.resize(CAMs[0], (480, 640)).astype('float32')
	cam_map = np.asarray([item * 1.0 / 255 for item in cam_map])
	# print(np.concatenate((colorDecode(seg_img).reshape(-1).astype('float32'), cammap.reshape(-1)), axis = 1).shape)
	decode_map = colorDecode(seg_img).reshape(-1)
	cam_map    = cam_map.reshape(-1).astype('float32')
	interpolation = unique(decode_map, cam_map)
	int_per    = [x/y for x, y in zip(interpolation.values(), seg_map.values())]

	conn = psycopg2.connect(conn_string)
	cur = conn.cursor()
	query = '''UPDATE penn_station.image_lookup_1000ms SET pred0 = '%s', pred1 = '%s', pred2 = '%s', pred3 = '%s', pred4 = '%s', prob0 = %s, prob1 = %s, prob2 = %s, prob3 = %s, prob4 = %s, seg_id = '%s', seg_tt = '%s', seg_per = '%s' WHERE image_name = '%s'; ''' % (classes[idx[0]], classes[idx[1]], classes[idx[2]], classes[idx[3]], classes[idx[4]], probs[0], probs[1], probs[2], probs[3], probs[4], seg_map.keys(), seg_map.values(), int_per, img_name)
	# print(query)
	cur.execute(query)
	cur.close()
	conn.commit()


	alpha = 0.7
	cam_out = cv2.resize(CAMs[0], (224, 224)).reshape(1, 50176).reshape((1, 1, 224, 224))[0, :, :, :].transpose(1, 2, 0)

	# cam_out = np.concatenate((cam_out, cam_out, cam_out), axis = 2)
	# # cam_out = cv2.resize(cam_out, (width, height))
	# background  = np.ones(3, width, height)
	# added_image = cv2.addWeighted(background, 0.4, overlay, 0.1,0)
	# print(cam_out)
	# cv2.addWeighted(cam_out, 0, img, alpha, 0, img)
	# cv2.imwrite(os.path.join(os.getcwd(), "..", "web", "final", "images", str(cam_id), "heatmap", str(clip_id) + "_" + str(clip_count) + '.png'), img)
	heatmap = cv2.applyColorMap(cv2.resize(cam_out,(width, height)), cv2.COLORMAP_JET)
	result = heatmap * 0.3 + img * 0.5
	cv2.imwrite(os.path.join(os.getcwd(), "..", "web", "final", "images", str(cam_id), "heatmap", str(clip_id) + "_" + str(clip_count) + '.png'), result)

