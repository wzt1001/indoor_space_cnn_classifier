import pyproj
import os
import json
import geojson
import math
from pprint import pprint
from geojson import MultiLineString, MultiPoint
import psycopg2
import csv



def sigmoid(x):
	return 1 / (0.18 + math.exp(-x))


csv_lookup = {}

with open('/media/ztwang/ssd_0/data/indoor_space_cnn_classifier/scripts/semantic-segmentation-pytorch/data/object150_info.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for idx, row in enumerate(spamreader):
        if idx == 0:
        	continue
        print(row[0].split(',')[5])
        csv_lookup[int(row[0].split(',')[0]) - 1] = row[0].split(',')[5]

conn_string = "host='localhost' dbname='indoor_position' user='postgres' password='tiancai' port='5432'"
conn = psycopg2.connect(conn_string)
cur = conn.cursor()

query = '''select * from penn_station.area_lookup_50ms'''
cur.execute(query)
lookup_center_data = cur.fetchall()
cur.close()
conn.commit()
lookup_center = {}
for item in lookup_center_data:
	lookup_center[item[1]] = (item[2], item[3])

conn_string = "host='localhost' dbname='indoor_position' user='postgres' password='tiancai' port='5432'"
conn = psycopg2.connect(conn_string)
cur = conn.cursor()

query = '''select image_name, lat, lon, spec_id, clip_id, clip_count, prob0, pred0, prob1, pred1, prob2, pred2, prob3, pred3, prob4, pred4, seg_id, seg_per, cam_id from penn_station.image_lookup_1000ms where cam_id = 0 order by clip_id, clip_count'''
cur.execute(query)
data = cur.fetchall()
cur.close()
conn.commit()





#Read JSON data into the datastore variable

output_lines = './geojson/output_lines.js'
output_points = './geojson/output_points.js'
output_links  = './geojson/output_links.js'
output_pie    = './geojson/output_pie.js'
output_scatter    = './geojson/output_scatter.js'
output_similar_imgs    = './geojson/output_similar_imgs.js'

degree  = 29.00
# convert = 111111
lines   = []
#Use the new datastore datastructure
total_coords = []
temp_line_id = None
temp_line	= []
line = []
clip_id = []
clip_count = []
pred0 = []
prob0 = []
pred1 = []
prob1 = []
pred2 = []
prob2 = []
pred3 = []
prob3 = []
pred4 = []
prob4 = []
seg_id = []
seg_per= []
cam_id = []
correct = []
image_name = []

for idx, item in enumerate(data):
	image_name.append(item[0])
	clip_id.append(item[4])
	clip_count.append(item[5])
	prob0.append(item[6])
	pred0.append(item[7])
	prob1.append(item[8])
	pred1.append(item[9])
	prob2.append(item[10])
	pred2.append(item[11])
	prob3.append(item[12])
	pred3.append(item[13])
	prob4.append(item[14])
	pred4.append(item[15])
	seg_id.append(item[16])
	seg_per.append(item[17])
	cam_id.append(item[18])

	if item[3] != item[7]:
		correct.append('false')
	else:
		correct.append('true')

	total_coords.append((item[1], item[2]))
	if idx > 0 and item[4] != temp_line_id and temp_line_id is not None:
		lines.append(line)
		line = []
	temp_line_id = item[4]

	# generate routes
	line.append((item[1], item[2]))

lines.append(line)
line_scale = 40

dump = geojson.dumps(MultiLineString(lines), sort_keys=True)

with open(output_lines, 'wb') as output_file:
	output_file.write('var lines = ')
	json.dump(dump, output_file, indent=2) 

with open(output_links, 'wb') as output_file:

	properties_output = []
	for a, x in enumerate(prob0):
		if correct[a] == False:
			properties_output.append(prob0[a])
			properties_output.append(prob1[a])
			properties_output.append(prob2[a])
			properties_output.append(prob3[a])
			properties_output.append(prob4[a])
		else:
			properties_output.append(prob1[a])
			properties_output.append(prob2[a])
			properties_output.append(prob3[a])
			properties_output.append(prob4[a])

	output_file.write('''var links = {'type': 'FeatureCollection',
                'features': [''')
	for idx, point in enumerate(total_coords):
		# if idx == 1:
		# 	break
		# print(idx, pred1[idx])
		# generate links between current point and other areas
		if correct[idx] == False:
			output_file.write('''{"type":"Feature", "properties": {"line-width": %s, "index": %s}, "geometry":{"type":"LineString","coordinates":''' % (('{0:.8f}'.format(float(prob0[idx]) * line_scale), idx)))
			output_file.write('''[[%s, %s], [%s, %s]]''' % (point[0], point[1], lookup_center[pred0[idx]][0], lookup_center[pred0[idx]][1]))
			output_file.write('''}}, ''')

		print('''{"type":"Feature", "properties": {"line-width": %s, "index": %s}, "geometry":{"type":"LineString","coordinates":''' % (('{0:.8f}'.format(float(prob0[idx]) * line_scale), idx)))

		output_file.write('''{"type":"Feature", "properties": {"line-width": %s, "index": %s}, "geometry":{"type":"LineString","coordinates":''' % (('{0:.8f}'.format(float(prob1[idx]) * line_scale), idx)))
		output_file.write('''[[%s, %s], [%s, %s]]''' % (point[0], point[1], lookup_center[pred1[idx]][0], lookup_center[pred1[idx]][1]))
		output_file.write('''}}, ''')

		output_file.write('''{"type":"Feature", "properties": {"line-width": %s, "index": %s}, "geometry":{"type":"LineString","coordinates":''' % (('{0:.8f}'.format(float(prob1[idx]) * line_scale), idx)))
		output_file.write('''[[%s, %s], [%s, %s]]''' % (point[0], point[1], lookup_center[pred2[idx]][0], lookup_center[pred2[idx]][1]))
		output_file.write('''}}, ''')

		output_file.write('''{"type":"Feature", "properties": {"line-width": %s, "index": %s}, "geometry":{"type":"LineString","coordinates":''' % (('{0:.8f}'.format(float(prob1[idx]) * line_scale), idx)))
		output_file.write('''[[%s, %s], [%s, %s]]''' % (point[0], point[1], lookup_center[pred3[idx]][0], lookup_center[pred3[idx]][1]))
		output_file.write('''}}, ''')
		
		output_file.write('''{"type":"Feature", "properties": {"line-width": %s, "index": %s}, "geometry":{"type":"LineString","coordinates":''' % (('{0:.8f}'.format(float(prob1[idx]) * line_scale), idx)))
		output_file.write('''[[%s, %s], [%s, %s]]''' % (point[0], point[1], lookup_center[pred4[idx]][0], lookup_center[pred4[idx]][1]))
		output_file.write('''}},''')

	output_file.seek(-1, os.SEEK_END)
	output_file.write(''']}''')

with open(output_points, 'wb') as output_file:
	output_file.write('''var points = \n{\n
		"type":"FeatureCollection","metadata":{"count":%s},"features":[
		''' % (len(output_points)))
	for idx, point in enumerate(total_coords):
		# if idx == 40:
		# 	break
		# print(clip_id[idx], idx, point[0])
		# print(prob0[idx])
		# print(round(float(prob0[idx]), 8))
		output_file.write('''{"type":"Feature","properties":{"mag": %s, "other_pred": {"%s": %s, "%s": %s, "%s": %s, "%s": %s}, "correct": %s, "index": %s, "clip_id": %s, "clip_count": %s}, "geometry":{"type":"Point","coordinates":[%s, %s]}, "id":"%s"},''' % (round(float(prob0[idx]), 8), pred1[idx], round(float(prob1[idx]), 8), pred2[idx], round(float(prob2[idx]), 8), pred3[idx], round(float(prob3[idx]), 8), pred4[idx], round(float(prob4[idx]), 8), correct[idx], idx, clip_id[idx], clip_count[idx], point[0], point[1], image_name[idx]))
	output_file.seek(-1, os.SEEK_END)
	output_file.write('''],"bbox":[-178.7298,-55.755,5,179.1278,59.8935,664]\n}\n''' % ())


with open(output_pie, 'wb') as output_file:
	output_file.write('''var pie_data = [''')
	for idx, point in enumerate(total_coords):

		output_file.write('''[''' )
		for b, item in enumerate(eval(seg_per[idx])):
			output_file.write(''' {'value':%s, 'name':'%s'},''' % (float(item) * 300, csv_lookup[eval(seg_id[idx])[b]]))
		output_file.seek(-1, os.SEEK_END)
		output_file.write('''],''')

	output_file.seek(-1, os.SEEK_END)
	output_file.write(''']''')
		

with open(output_scatter, 'wb') as output_file:

	batch = 40
	scatter_output = {}
	scatter_data   = []
	select_elements = [147, 41, 14, 50, 6, 0, 8, 115, 43, 80, 69, 138, 44]
	select_elements_dic = {'147': 0, '41': 1, '14': 2, '50': 3, '6': 4, '0': 5, '8': 6, '115': 7, '43': 8, '80': 9, '69': 10, '138':11, '44': 12}
	for key, value in csv_lookup.iteritems():
		scatter_output[key] = []
	for idx in range(len(total_coords) / batch):
		# print(total_coords[idx * batch])
		temp_store = {}
		for key, value in csv_lookup.iteritems():
			temp_store[key] = 0
		for i in range(batch):
			temp_seg_id = eval(seg_id[idx * batch + i])
			temp_seg_per = eval(seg_per[idx * batch + i])
			# print(temp_seg_id, temp_seg_per)
			for f in range(len(temp_seg_id)):
				# print(temp_seg_id[f], temp_seg_per[f])
				temp_store[temp_seg_id[f]] += temp_seg_per[f]

		# print(sorted(temp_store.items(), key=lambda x: x[1], reverse=True))
		temp_store = [(k, v) for k, v in (sorted(temp_store.items(), key=lambda x: x[1], reverse=True)) if k in select_elements]
		print(temp_store)
		print(idx)
		scatter_output[idx]= temp_store


	for key, value in scatter_output.iteritems():
		for avg_point in value:
			# print(select_elements_dic[avg_point[0]])
			scatter_data.append([select_elements_dic[str(avg_point[0])], key, round(sigmoid(avg_point[1]) * 1.8, 2)])
			# print(key)
		print(scatter_data)

	temp_total = len(total_coords) / batch
 	scatter_hours = []
 	for idx in range(temp_total):
		scatter_hours.append('{0:.1f}'.format(idx * 1.0 / temp_total * 100) + '%')
	print(scatter_hours)
	scatter_days  = [value for key, value in csv_lookup.iteritems() if (key - 1) in [147, 41, 14, 50, 6, 0, 8, 115, 43, 80, 69, 138, 44]]
	print(csv_lookup)
	# print(scatter_days)
	output_file.write('''var scatter_data_orginal = {'hours': %s, 'days': %s, 'scatter_data': %s} 
		''' % (scatter_hours, scatter_days, scatter_data) )

with open(output_similar_imgs, 'wb') as output_file:

	temp_search = {}
	for idx, point in enumerate(total_coords):
		
		conn = psycopg2.connect(conn_string)
		cur = conn.cursor()
		query = '''select clip_id, clip_count, cam_id from penn_station.image_lookup_1000ms where spec_id = '%s'  order by random() limit 2''' % (pred1[idx])
		cur.execute(query)
		data0 = cur.fetchall()
		cur.close()
		conn.commit()

		conn = psycopg2.connect(conn_string)
		cur = conn.cursor()
		query = '''select clip_id, clip_count, cam_id from penn_station.image_lookup_1000ms where spec_id = '%s'  order by random() limit 2''' % (pred2[idx])
		cur.execute(query)
		data1 = cur.fetchall()
		cur.close()
		conn.commit()

		conn = psycopg2.connect(conn_string)
		cur = conn.cursor()
		query = '''select clip_id, clip_count, cam_id from penn_station.image_lookup_1000ms where spec_id = '%s'  order by random() limit 2''' % (pred3[idx])
		cur.execute(query)
		data2 = cur.fetchall()
		cur.close()
		conn.commit()

		conn = psycopg2.connect(conn_string)
		cur = conn.cursor()
		query = '''select clip_id, clip_count, cam_id from penn_station.image_lookup_1000ms where spec_id = '%s'  order by random() limit  2''' % (pred4[idx])
		cur.execute(query)
		data3 = cur.fetchall()
		cur.close()
		conn.commit()
		print(data1)
		# print(data1)
		temp_search[idx] = ['images/' + str(data0[0][2]) + '/original/' + str(data0[0][0]) + '_' + str(data0[0][1]) + '.jpg', 'images/' + str(data0[1][2]) + '/original/' + str(data0[1][0]) + '_' + str(data0[1][1]) + '.jpg', 'images/' + str(data1[0][2]) + '/original/' + str(data1[0][0]) + '_' + str(data1[0][1]) + '.jpg', 'images/' + str(data1[1][2]) + '/original/' + str(data1[1][0]) + '_' + str(data1[1][1]) + '.jpg', 'images/' + str(data2[0][2]) + '/original/' + str(data2[0][0]) + '_' + str(data2[0][1]) + '.jpg', 'images/' + str(data2[1][2]) + '/original/' + str(data2[1][0]) + '_' + str(data2[1][1]) + '.jpg', 'images/' + str(data3[0][2]) + '/original/' + str(data3[0][0]) + '_' + str(data3[0][1]) + '.jpg', 'images/' + str(data3[1][2]) + '/original/' + str(data3[1][0]) + '_' + str(data3[1][1]) + '.jpg']
		print(idx)
		# print(temp_search)

	output_file.write('''var output_similar_imgs = %s''' % str(temp_search))

	# output_file.seek(-1, os.SEEK_END)
	# output_file.write(''']''')







