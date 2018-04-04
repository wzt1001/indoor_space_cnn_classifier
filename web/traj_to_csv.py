import pyproj
import os
import json
import geojson
import math
from pprint import pprint
from geojson import MultiLineString
from osgeo import ogr
import pandas as pd

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

output_filename = 'output.csv'

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

multiline = ogr.Geometry(ogr.wkbMultiLineString)

output_layer = [[], [], []]

#Use the new datastore datastructure
for line_id, item in enumerate(data):
	line = ogr.Geometry(ogr.wkbLineString)
	timestamps = []
	for point in item:
		offset_lat = (-math.sin(math.radians(degree)) * point[1] + math.cos(math.radians(degree)) * point[2])
		offset_lon = (math.cos(math.radians(degree)) * point[1] + math.sin(math.radians(degree)) * point[2])
		# coords.append((offset_lon + origin_lon, offset_lat + origin_lat))
		line.AddPoint(offset(origin_lon, origin_lat, offset_lon, offset_lat)[0], offset(origin_lon, origin_lat, offset_lon, offset_lat)[1])
		timestamps.append(point[0])
	output_layer[0].append(line_id)
	output_layer[1].append(line.ExportToWkt())
	output_layer[2].append(timestamps)

# dump = geojson.dumps(MultiLineString(lines), sort_keys=True)
# dump = multiline.ExportToWkt()

# with open(output_filename, 'wb') as output_file:
#     json.dump(dump, output_file, indent=2) 

# with open(csvfile, "w") as output:
#     writer = csv.writer(output, lineterminator='\n')
#     for val in res:
#         writer.writerow([val])    

my_df = pd.DataFrame(output_layer).transpose()
my_df.to_csv(output_filename, index=False, header=False)
print my_df

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





