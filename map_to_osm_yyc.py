import json
import numpy as np
import os


output_dict = {}

mvf_folder = "/home/kevinmeng/workspace/mappedin/VPS/MVF/YYC"

with open(mvf_folder + "/map.geojson", "r") as f:
    m = json.load(f)

# Get 2nd floor
second_floor = [level for level in m if level['externalId'] == "MAP-A490ZOYJ"] # Currently for Mezzanine

second_floor_id = second_floor[0]['id']
print(f"floor id: {second_floor_id}")

with open(mvf_folder + "/space/" + second_floor_id + ".geojson", "r") as f:
    floor = json.load(f)

min_lon = float('inf')
max_lon = float('-inf')
min_lat = float('inf')
max_lat = float('-inf')

output_dict['elements'] = []

idx = 0

with open(mvf_folder + "/space/" + second_floor_id + ".geojson", "r") as f:
    space = json.load(f)

for feature in space['features']:
    
    if feature['geometry']['type'] == 'Polygon':
        coordinates = feature['geometry']['coordinates'][0]
        
        indexes = []
        
        for i, coordinate in enumerate(coordinates):
            # Update min/max values for each coordinate
            min_lon = min(min_lon, coordinate[0])
            max_lon = max(max_lon, coordinate[0])
            min_lat = min(min_lat, coordinate[1])
            max_lat = max(max_lat, coordinate[1])
            
            output_dict['elements'].append({"type": "node", 
                                          "id": idx,
                                          "lat": coordinate[1],
                                          "lon": coordinate[0],
                                          "visible": True})
            indexes.append(idx)
            idx += 1

        output_dict['elements'].append({"type": "way",
                                      "tags": {"leisure": "playground"}, 
                                      "id": idx,
                                      "visible": True,
                                      "nodes": [i for i in indexes]})

    idx += 1
    indexes = []

output_dict['bounds'] = {
    'minlat': float(min_lat),
    'maxlat': float(max_lat),
    'minlon': float(min_lon),
    'maxlon': float(max_lon)
}

with open('yyc_osm.json', 'w') as f:
    json.dump(output_dict, f)

