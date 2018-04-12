import pyproj
import os
import json
import geojson
import math
from pprint import pprint
from geojson import MultiLineString, MultiPoint
import psycopg2

conn_string = "host='localhost' dbname='indoor_position' user='postgres' password='tiancai' port='5432'"
conn = psycopg2.connect(conn_string)
cur = conn.cursor()

query = '''select image_name, lat, lon, spec_id, clip_id, clip_count, prob0, pred0 from penn_station.image_lookup_1000ms where cam_id = 0 order by clip_id, clip_count'''
cur.execute(query)
data = cur.fetchall()
cur.close()
conn.commit()

#Read JSON data into the datastore variable

output_lines = './geojson/output_lines.js'
output_points = './geojson/output_points.js'

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
correct = []

for idx, item in enumerate(data):
	clip_id.append(item[4])
	clip_count.append(item[5])
	prob0.append(item[6])
	pred0.append(item[7])
	if item[3] != item[7]:
		correct.append('false')
	else:
		correct.append('true')
	total_coords.append((item[1], item[2]))
	if idx > 0 and item[4] != temp_line_id and temp_line_id is not None:
		lines.append(line)
		line = []
	temp_line_id = item[4]
	line.append((item[1], item[2]))


lines.append(line)

dump = geojson.dumps(MultiLineString(lines), sort_keys=True)

with open(output_lines, 'wb') as output_file:
	output_file.write('var lines = ')
	json.dump(dump, output_file, indent=2) 

with open(output_points, 'wb') as output_file:
	output_file.write('''var points = \n{\n
		"type":"FeatureCollection","metadata":{"generated":1455162818000,"url":"http://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=2015-01-01&endtime=2015-12-31&minmagnitude=6","title":"USGS Earthquakes","status":200,"api":"1.1.1","count":%s},"features":[
		''' % (len(output_points)))
	for idx, point in enumerate(total_coords):
		# if idx == 40:
		# 	break
		print(clip_id[idx], idx, point[0])
		print(prob0[idx])
		print(round(float(prob0[idx]), 1))
		output_file.write('''{"type":"Feature","properties":{"mag": %s, "correct": %s, "index": %s, "clip_id": %s, "clip_count": %s}, "geometry":{"type":"Point","coordinates":[%s, %s]},"id":"us10003mrv"},''' % (round(float(prob0[idx]), 1), correct[idx], idx, clip_id[idx], clip_count[idx], point[0], point[1]))
	output_file.seek(-1, os.SEEK_END)
	output_file.write('''],"bbox":[-178.7298,-55.755,5,179.1278,59.8935,664]\n}\n''' % ())

# {"type":"Feature","properties":{"mag": } "geometry":{"type":"Point","coordinates":[-135.708,-54.4856,10]},"id":"us10003mrv"}


# geojson_plain_top = '''
# {
#	   "id": "route",
#	   "type": "line",
#	   "source": {
#		  "type": "geojson",
#		  "data": 
# '''


# geojson_plain_bottom = '''
#		  }
#	   },
#	   "layout": {
#		  "line-join": "round",
#		  "line-cap": "round"
#	   },
#	   "paint": {
#		  "line-color": "#888",
#		  "line-width": 8
#	   }
#	});
# }
# '''





