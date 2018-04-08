import cv2
import os
from os import listdir
from os.path import isfile, join
from sys import stdout
import psycopg2
import pickle
import numpy as np
from PIL import Image

def gen_coord(line, time):

	percentage = None
	for idx, timestamp in enumerate(line["time"]):
		if time < timestamp:
			index = idx
			percentage = (time - line["time"][index - 1]) * 1.0 / (timestamp - line["time"][index - 1])
			break
	if percentage is None:
		return None
	else:
		coord_x = (line["points"][index][0] - line["points"][index - 1][0]) * percentage + line["points"][index - 1][0]
		coord_y = (line["points"][index][1] - line["points"][index - 1][1]) * percentage + line["points"][index - 1][1]
		return (coord_x, coord_y)
#preset settings
frame_interval = 50


# connect to server database
conn_string = "host='localhost' dbname='indoor_position' user='postgres' password='tiancai' port='5432'"
conn = psycopg2.connect(conn_string)
cur = conn.cursor()

query = '''select * from penn_station.areas'''
cur.execute(query)
areas = cur.fetchall()
cur.close()
conn.commit()

conn = psycopg2.connect(conn_string)
cur = conn.cursor()
query = '''select id, field_3 from penn_station.routes'''
cur.execute(query)
results = cur.fetchall()
cur.close()
conn.commit()
records = {}
for item in results:
	records[item[0]] = {}
	records[item[0]]["time"] = eval(item[1])
	records[item[0]]["points"] = []

conn = psycopg2.connect(conn_string)
cur = conn.cursor()
query = '''select id, (ST_DumpPoints(geom)).path, ST_X(((ST_DumpPoints(geom)).geom)), ST_Y(((ST_DumpPoints(geom)).geom)) from penn_station.routes'''
cur.execute(query)
results = cur.fetchall()
cur.close()
conn.commit()
for item in results:
	records[item[0]]["points"].append((item[2], item[3]))
# print(records)
print("--- parameters loaded...\n--- begin categorizing")
# for points in results:

place_title = "penn_station"
root_dir  	= os.path.join(os.getcwd(), "..", "data", place_title)

left_dir	= os.path.join(root_dir, "1")
forward_dir = os.path.join(root_dir, "3")
right_dir   = os.path.join(root_dir, "4")
back_dir	= os.path.join(root_dir, "5")

output_dir  	= os.path.join(root_dir, "extracted_%sms" % str(frame_interval))

left_files 		= [join(left_dir, f) for f in listdir(left_dir) if isfile(join(left_dir, f))]
forward_files 	= [join(forward_dir, f) for f in listdir(forward_dir) if isfile(join(forward_dir, f))]
right_files 	= [join(right_dir, f) for f in listdir(right_dir) if isfile(join(right_dir, f))]
back_files 		= [join(back_dir, f) for f in listdir(back_dir) if isfile(join(back_dir, f))]

if not len(left_files) == len(forward_files) == len(right_files) == len(back_files) == len(top_files):
	print("!!! file count not consistent")
# print(left_files, forward_files, right_files, back_files, top_files)
if not os.path.exists(output_dir):
	os.makedirs(output_dir)

os.chdir(output_dir)

all_files = [left_files, forward_files, right_files, back_files]
total_cnt = 0
fail_cnt  = 0
output_files = [""] * len(left_files)
lookup_table = {"2-1": 0, "2-3": 1, "2-4": 2, "2-5": 3, "2-6": 4, "2-8-2": 5, "2-9": 6, "2-10": 7}

# create table to store image infos
# conn = psycopg2.connect(conn_string)
# cur = conn.cursor()
# query = '''
# 	drop table if exists penn_station.image_lookup_%sms; create table penn_station.image_lookup_%sms
# 	(
# 	  image_name text,
# 	  id integer,
# 	  spec_id text,
# 	  path text,
# 	  lat double precision,
# 	  lon double precision
# 	);

# 	drop table if exists penn_station.missing; create table penn_station.missing
# 	(
# 	  lat double precision,
# 	  lon double precision
# 	)

# ''' % (str(frame_interval), str(frame_interval))
# # print(query)
# cur.execute(query)
# cur.close()
# conn.commit()
# print("--- table penn_station.image_lookup_%sms created" % (str(frame_interval)))
all_files = [[], forward_files, right_files, back_files]
all_files = []

for cam_id, direction in enumerate(all_files):
	for clip_id in range(len(direction)):
		print("------ begin extracting from cam_id: %s, clip_id %s" % (cam_id, clip_id) )
		vidcap = cv2.VideoCapture(direction[clip_id])
		# success, image = vidcap.read()
		count = 0
		# if not success:
		# 	print("video reading error for %s" % direction[clip_id])
		# 	continue
		while True:
			current_line = records[lookup_table[os.path.splitext((os.path.basename(direction[clip_id])))[0]]]
			vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * frame_interval) )
			current_time = count * 1.0 * frame_interval / 1000
			coord = gen_coord(current_line, current_time)
			# print(current_line, count * 1.0 * frame_interval / 1000)
			if coord is not None:
				conn = psycopg2.connect(conn_string)
				cur = conn.cursor()
				query = '''select id from penn_station.areas a where ST_Intersects(ST_PointFromText('POINT(%s %s)', 4326), ST_Transform(a.geom, 4326)) is True''' % (coord[0], coord[1])
				# print(query)
				cur.execute(query)

				if not cur.rowcount == 0:
					area_id = cur.fetchone()[0]
					success, image = vidcap.read()
					cur.close()
					conn.commit()

					if (image is not None):
						#print(count * frame_interval)

						spec_id  = str(area_id) + str(cam_id)
						filename = "%s_%s_%s.png" % (spec_id, str(cam_id), count)
						cv2.imwrite(filename, image)	 # save frame as JPEG file

						conn = psycopg2.connect(conn_string)
						cur = conn.cursor()
						query = '''insert into penn_station.image_lookup_%sms values('%s', %s, '%s', '%s', %s, %s)''' % (str(frame_interval), filename, area_id, spec_id, os.path.join(os.getcwd(), filename), coord[0], coord[1])
						# print(query)
						cur.execute(query)
						cur.close()
						conn.commit()

						stdout.write("\rcam_id %s, area_id %s, saved cnt %s, not_in_any_area cnt %s, clip_id %s" % (str(cam_id), area_id, total_cnt - fail_cnt, fail_cnt, clip_id))
						stdout.flush()

					else:
						print(count * frame_interval, "disrupted")
						count += 1
						#total_cnt += 1
						#fail_cnt += 1
						#continue
						#print(success, image)
						#print("\nend of one video")
						continue

				else:
					cur.close()
					conn.commit()

					conn = psycopg2.connect(conn_string)
					cur = conn.cursor()
					query = '''insert into penn_station.missing values(%s, %s)''' % (coord[0], coord[1])
					# print(query)
					cur.execute(query)
					cur.close()
					conn.commit()

					stdout.write("\rcam_id %s, saved cnt %s, not_in_any_area cnt %s, clip_id %s" % (str(cam_id),  total_cnt - fail_cnt, fail_cnt, clip_id))
					stdout.flush()

					fail_cnt += 1
				
			else:
				print("break because coord is None")
				break
			if cv2.waitKey(10) == 27: # exit if Escape is hit
				break
			count += 1
			total_cnt += 1

conn = psycopg2.connect(conn_string)
cur = conn.cursor()
query = '''select spec_id, count(spec_id) as cnt1 from penn_station.image_lookup_%sms group by spec_id having count(spec_id) > 100; ''' % str(frame_interval)
# print(query)
cur.execute(query)
results = cur.fetchall()

cur.close()
conn.commit()

spec_cat = {}

# for shitty classifer code
# for spec_id in results:
# 	conn = psycopg2.connect(conn_string)
# 	cur = conn.cursor()
# 	query = '''select * from penn_station.image_lookup_%sms where spec_id = '%s'; ''' % (str(frame_interval), spec_id[0])
# 	# print(query)
# 	cur.execute(query)
# 	images = cur.fetchall()
	
# 	output_dir = os.path.join(os.getcwd(), '..', 'categories', spec_id[0])
# 	if not os.path.exists(output_dir):
# 		os.makedirs(output_dir)

# 	spec_cat[spec_id[0]] = []
# 	for image in images:
# 		resized_image = cv2.resize(np.array(Image.open(image[3])), dsize=(200, 200), interpolation=cv2.INTER_CUBIC)
# 		cv2.imwrite(os.path.join(output_dir, os.path.basename(image[3])), resized_image)
# 	cur.close()
# 	conn.commit()


#	for bolei's code
per_train = 0.8
per_val = 0.08
per_test = 0.1

data_train = np.array([], dtype=np.uint8).reshape(0, 49152)
label_train = np.array([], dtype=np.uint8)

data_val = np.array([], dtype=np.uint8).reshape(0, 49152)
label_val = np.array([], dtype=np.uint8)

data_test = np.array([], dtype=np.uint8).reshape(0, 49152)
label_test = np.array([], dtype=np.uint8)

for idx, spec_id in enumerate(results):
	conn = psycopg2.connect(conn_string)
	cur = conn.cursor()
	query = '''select * from penn_station.image_lookup_%sms where spec_id = '%s'; ''' % (str(frame_interval), spec_id[0])
	# print(query)


	cur.execute(query)
	images = cur.fetchall()

	idx = int(idx)
	category = spec_id

	# print(images)
	data_category = np.array([cv2.resize(np.array(Image.open(fname[3])), dsize=(128, 128), interpolation=cv2.INTER_CUBIC) for fname in images])	

	# data_category = np.load(file_np).astype(np.int8)
	total_cnt = len(data_category)
	num_train = int(per_train * total_cnt)
	num_val = int(per_val * total_cnt)
	num_test = int(per_test * total_cnt)
	print (num_train, num_val, num_test)
	# generate split

	train_category = data_category[:num_train].reshape(-1, 49152)
	val_category = data_category[num_train:num_train+num_val].reshape(-1, 49152)
	test_category = data_category[num_train+num_val:num_train+num_val+num_test].reshape(-1, 49152)

	print(data_train.shape, train_category.shape)

	# concatenate: TODO: change this to pre-assign to speed up
	data_train = np.concatenate((data_train, train_category), axis=0)
	label_train = np.concatenate((label_train, np.ones((num_train,), dtype=int) * idx), axis=0)

	data_val = np.concatenate((data_val, val_category), axis=0)
	label_val = np.concatenate((label_val, np.ones((num_val,), dtype=int) * idx), axis=0)

	data_test = np.concatenate((data_test, test_category), axis=0)
	label_test = np.concatenate((label_test, np.ones((num_test,), dtype=int) * idx), axis=0)

	print('num_train=%d num_val=%d num_test=%d' % (data_train.shape[0], data_val.shape[0], data_test.shape[0]))

# 	with open(os.path.join(os.getcwd(), '..', 'npy', ), 'wb') as f:
output_dir = os.path.join(os.getcwd(), '..', 'datasplit')
if not os.path.exists(output_dir):
	os.makedirs(output_dir)
np.savez( os.path.join(output_dir, '%s_split_%s.npy' % (num_train, "128")), data_train=data_train, label_train=label_train, data_val=data_val, label_val=label_val, data_test=data_test, label_test=label_test)


