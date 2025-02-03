import json
import numpy as np
import os


output_dict = {}

mvf_folder = "/home/kevinmeng/workspace/mappedin/VPS/MVF Mappedin Office  - Den 1880"

with open(mvf_folder + "/map.geojson", "r") as f:
    m = json.load(f)

# Get 2nd floor
print(m)
second_floor = [level for level in m if level['externalId'] == 'Level 2']
second_floor_id = second_floor[0]['id']

with open(mvf_folder + "/floor.geojson", "r") as f:
    floor = json.load(f)

second_floor_features = [feature for feature in floor['features'] if feature['properties']['id'] == second_floor_id]
second_floor_outline = second_floor_features[0]['geometry']['coordinates'][0]
# Convert coordinates list into numpy arrays for x and y
second_floor_outline_coords = np.array(second_floor_outline)
min_lon = min(second_floor_outline_coords[:, 0])
max_lon = max(second_floor_outline_coords[:, 0])
min_lat = min(second_floor_outline_coords[:, 1])
max_lat = max(second_floor_outline_coords[:, 1])

output_dict['bounds'] = {}
output_dict['bounds']['minlat'] = float(min_lat)
output_dict['bounds']['maxlat'] = float(max_lat)
output_dict['bounds']['minlon'] = float(min_lon)
output_dict['bounds']['maxlon'] = float(max_lon)

output_dict['elements'] = []

idx = 0

# Point
with open(os.path.join(mvf_folder, 'annotation', second_floor_id + '.geojson'), "r") as f:
    annotations = json.load(f)

for annotation in annotations['features']:
    output_dict['elements'].append({"type": "node", 
                                    "id": idx,
                                    "lat": annotation['geometry']['coordinates'][1],
                                    "lon": annotation['geometry']['coordinates'][0],
                                    "visible": True})
    idx += 1

# Obstruction: LineString and Polygon
with open(os.path.join(mvf_folder, 'obstruction', second_floor_id + '.geojson'), "r") as f:
    obstructions = json.load(f)

for obstruction in obstructions['features']:
    
    coordinates = obstruction['geometry']['coordinates']
    if obstruction['geometry']['type'] == 'Polygon':
        coordinates = coordinates[0]

    indexes = []
    
    for i, coordinate in enumerate(coordinates):
        output_dict['elements'].append({"type": "node", 
                                        "id": idx,
                                        "lat": coordinate[1],
                                        "lon": coordinate[0],
                                        "visible": True})
        indexes.append(idx)
        idx += 1

    if obstruction['geometry']['type'] == 'LineString':
        output_dict['elements'].append({"type": "way",
                                        "tags": {"leisure": "playground"}, 
                                        "id": idx,
                                        "visible": True,
                                        "nodes": [i for i in indexes],})
    elif obstruction['geometry']['type'] == 'Polygon':
        output_dict['elements'].append({"type": "relation",
                                        "tags": {},
                                        "id": idx,
                                        "visible": True,
                                        "members": [{
                                              "type": "node",
                                              "ref": i,
                                              'role': 'outer'
                                              } for i in indexes]
                                        })
    idx += 1
    indexes = []

with open(os.path.join(mvf_folder, 'window', second_floor_id + '.geojson'), "r") as f:
    windows = json.load(f)

for window in windows['features']:
    coordinates = window['geometry']['coordinates']
    indexes = []
    
    for coordinate in coordinates:
        output_dict['elements'].append({
            "type": "node", 
            "id": idx,
            "lat": coordinate[1],
            "lon": coordinate[0],
            "visible": True
        })
        indexes.append(idx)
        idx += 1

    output_dict['elements'].append({
        "type": "way",
        "tags": {"barrier": "hedge"}, 
        "id": idx,
        "visible": True,
        "nodes": [i for i in indexes],
    })
    idx += 1
    indexes = []
########################################################
with open(os.path.join(mvf_folder, 'entrance', second_floor_id + '.geojson'), "r") as f:
    entrances = json.load(f)

for entrance in entrances['features']:
    coordinates = entrance['geometry']['coordinates']
    indexes = []
    
    for coordinate in coordinates:
        output_dict['elements'].append({
            "type": "node", 
            "id": idx,
            "lat": coordinate[1],
            "lon": coordinate[0],
            "visible": True
        })
        indexes.append(idx)
        idx += 1

    output_dict['elements'].append({
        "type": "way",
        "tags": {"barrier": "fence"}, 
        "id": idx,
        "visible": True,
        "nodes": [i for i in indexes],
    })
    idx += 1
    indexes = []



with open('den_osm.json', 'w') as f:
    json.dump(output_dict, f)

