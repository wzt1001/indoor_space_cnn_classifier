import pyproj
import os
import json
import geojson
import math
from pprint import pprint
from geojson import MultiLineString


def offset(lon, lat, de, dn):
	# lat, lon Position, decimal degrees
	# dn, de offsets in meters

	# Earth's radius, sphere
	R  = 6378137

	# Coordinate offsets in radians
	dLon = de / (R * math.cos(math.pi * lat / 180))
	dLat = dn / R

	# OffsetPosition, decimal degrees
	lonO = lon + dLon * 180 / math.pi
	latO = lat + dLat * 180 / math.pi
	return (lonO, latO)


#prompt the user for a file to import
data = json.load(open('trajectory.json'))

#Read JSON data into the datastore variable

output_filename = 'output.js'

# start = '''
# {
#             "type": "Feature",
#             "properties": {},
#             "geometry": {
#                "type": "LineString",
#                "coordinates": 

# '''
# bottom = '''
#             }
# '''

degree  = 29.00
# convert = 111111
origin_lat = 40.751180
origin_lon = -73.993971
lines   = []
print(math.cos(math.radians(degree)),math.sin(math.radians(degree)))
#Use the new datastore datastructure
for line_id, item in enumerate(data):
	coords = []
	for point in item:
		offset_lat = (-math.sin(math.radians(degree)) * point[1] + math.cos(math.radians(degree)) * point[2])
		offset_lon = (math.cos(math.radians(degree)) * point[1] + math.sin(math.radians(degree)) * point[2])
		# coords.append((offset_lon + origin_lon, offset_lat + origin_lat))
		coords.append(offset(origin_lon, origin_lat, offset_lon, offset_lat))

		# coords.append((offset_lon, offset_lat))

	lines.append(coords)

dump = geojson.dumps(MultiLineString(lines), sort_keys=True)

with open(output_filename, 'wb') as output_file:
    output_file.write('var dataset = ')
    json.dump(dump, output_file, indent=2) 

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





