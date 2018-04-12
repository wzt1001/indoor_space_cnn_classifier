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

query = '''select image_name, lat, lon, spec_id, clip_id, clip_count from penn_station.image_lookup_1000ms where cam_id = 0 order by clip_id, clip_count'''
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
temp_line    = []
line = []

for idx, item in enumerate(data):
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

dump = geojson.dumps(MultiPoint(total_coords), sort_keys=True)

with open(output_points, 'wb') as output_file:
    output_file.write('var points = ')
    json.dump(dump, output_file, indent=2)

{"type":"Feature","properties":{}, "geometry":{"type":"Point","coordinates":[-135.708,-54.4856,10]},"id":"us10003mrv"}
# geojson_plain_top = '''
# {
#       "id": "route",
#       "type": "line",
#       "source": {
#          "type": "geojson",
#          "data": 
# '''


# geojson_plain_bottom = '''
#          }
#       },
#       "layout": {
#          "line-join": "round",
#          "line-cap": "round"
#       },
#       "paint": {
#          "line-color": "#888",
#          "line-width": 8
#       }
#    });
# }
# '''





