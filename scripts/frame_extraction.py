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
print("parameters loaded...\nbegin categorizing")
# for points in results:

place_title = "penn_station"
root_dir  	= os.path.join(os.getcwd(), "..", "data", place_title)

left_dir    = os.path.join(root_dir, "1")
forward_dir = os.path.join(root_dir, "3")
right_dir   = os.path.join(root_dir, "4")
back_dir    = os.path.join(root_dir, "5")

output_dir  	= os.path.join(root_dir, "extracted_50ms")

left_files 	    = [join(left_dir, f) for f in listdir(left_dir) if isfile(join(left_dir, f))]
forward_files 	= [join(forward_dir, f) for f in listdir(forward_dir) if isfile(join(forward_dir, f))]
right_files 	= [join(right_dir, f) for f in listdir(right_dir) if isfile(join(right_dir, f))]
back_files 		= [join(back_dir, f) for f in listdir(back_dir) if isfile(join(back_dir, f))]

if not len(left_files) == len(forward_files) == len(right_files) == len(back_files) == len(top_files):
	print("file count not consistent")
# print(left_files, forward_files, right_files, back_files, top_files)
if not os.path.exists(output_dir):
	os.makedirs(output_dir)

os.chdir(output_dir)

all_files = [left_files, forward_files, right_files, back_files]
total_cnt = 0
fail_cnt  = 0
output_files = [""] * len(left_files)
lookup_table = {"2-1": 1, "2-3": 2, "2-4": 3, "2-5": 4, "2-6": 5, "2-8-2": 6, "2-9": 7, "2-10": 8}
#all_files = []

for a, direction in enumerate(all_files):
	for i in range(len(direction)):
		vidcap = cv2.VideoCapture(direction[i])
		success, image = vidcap.read()
		count = 0
		success = True
		while success:
			vidcap.set(cv2.CAP_PROP_POS_MSEC, (count*50))
			# print(os.path.splitext((os.path.basename(direction[i])))[0])
			coord = gen_coord(records[lookup_table[os.path.splitext((os.path.basename(direction[i])))[0]]], count * 0.05)
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
						filename = "frame%s_scene%s_%s.jpg" % (count, i, str(a + 1))
						cv2.imwrite(filename, image)     # save frame as JPEG file

						conn = psycopg2.connect(conn_string)
						cur = conn.cursor()
						query = '''insert into penn_station.image_lookup_50ms values('%s', %s, '%s', '%s')''' % (filename, area_id, str(area_id + 1) + str(a + 1), os.path.join(os.getcwd(), filename))
						# print(query)
						cur.execute(query)
						cur.close()
						conn.commit()

						stdout.write("\rfor camera %s, scene %s, saved cnt %s, not in area cnt %s, in area %s" % (str(a + 1), i, total_cnt - fail_cnt, fail_cnt, area_id))
						stdout.flush()

				else:
					cur.close()
					conn.commit()
					fail_cnt += 1
				
			else:
				break
			if cv2.waitKey(10) == 27: # exit if Escape is hit
				break
			count += 1
			total_cnt += 1

conn = psycopg2.connect(conn_string)
cur = conn.cursor()
query = '''select distinct(spec_id) from penn_station.image_lookup_50ms; '''
# print(query)
cur.execute(query)
results = cur.fetchall()

cur.close()
conn.commit()

spec_cat = {}

for spec_id in results:
	conn = psycopg2.connect(conn_string)
	cur = conn.cursor()
	query = '''select * from penn_station.image_lookup_50ms where spec_id = '%s'; ''' % (spec_id[0])
	# print(query)
	cur.execute(query)
	images = cur.fetchall()
	spec_cat[spec_id[0]] = []
	for image in images:
		spec_cat[spec_id[0]].append(image[3])
	cur.close()
	conn.commit()
	x = np.array([cv2.resize(np.array(Image.open(fname)), dsize=(128, 128), interpolation=cv2.INTER_CUBIC) for fname in spec_cat[spec_id[0]]])
# 	with open(os.path.join(os.getcwd(), '..', 'npy', ), 'wb') as f:
	x.dump(os.path.join(os.getcwd(), '..', 'npy', spec_id[0] + '.npy'))




# for i in range(len(left_files)):
#	s = pano.Stitch(output_files[i][:-1])
# 	s.leftshift()
#	s.showImage('left')
#	s.rightshift()
#	print ("done")
#	cv2.imwrite("scene%s.jpg" % i, s.leftImage)
#	print ("image written")
#	cv2.destroyAllWindows()
