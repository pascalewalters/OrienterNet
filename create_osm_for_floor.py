import json
import numpy as np
import os

def create_osm_for_floor(floor_id, floor_name, mvf_folder):
    output_dict = {}
    output_dict['elements'] = []
    
    min_lon = float('inf')
    max_lon = float('-inf')
    min_lat = float('inf')
    max_lat = float('-inf')
    
    idx = 0
    
    # Read the space data for this floor
    with open(os.path.join(mvf_folder, "space", f"{floor_id}.geojson"), "r") as f:
        space = json.load(f)
    
    for feature in space['features']:
        if feature['geometry']['type'] == 'Polygon':
            # Process all coordinate lists in the polygon
            for coordinate_list in feature['geometry']['coordinates']:
                indexes = []
                
                for coordinate in coordinate_list:
                    min_lon = min(min_lon, coordinate[0])
                    max_lon = max(max_lon, coordinate[0])
                    min_lat = min(min_lat, coordinate[1])
                    max_lat = max(max_lat, coordinate[1])
                    
                    output_dict['elements'].append({
                        "type": "node",
                        "id": idx,
                        "lat": coordinate[1],
                        "lon": coordinate[0],
                        "visible": True
                    })
                    indexes.append(idx)
                    idx += 1
                
                # Close the polygon by adding first node again
                indexes.append(indexes[0])

                output_dict['elements'].append({
                    "type": "way",
                    "tags": {"obstruction": "wall"},
                    "id": idx,
                    "visible": True,
                    "nodes": indexes
                })
                
                idx += 1
        
        else:
            idx += 1

    output_dict['bounds'] = {
        'minlat': float(min_lat),
        'maxlat': float(max_lat),
        'minlon': float(min_lon),
        'maxlon': float(max_lon)
    }
    
    os.makedirs('yyc_osms', exist_ok=True)
    
    with open(os.path.join('yyc_osms', f'{floor_name}_osm.json'), 'w') as f:
        json.dump(output_dict, f)

def main():
    # Load the map IDs from combined-output.geojson
    with open("/home/kevinmeng/workspace/mappedin/VPS/Mappedin_VPS_Data-20250127T163206Z-001/Mappedin_VPS_Data/YYC_VPS/combined-output.geojson", 'r') as f:
        maps = json.load(f)
    
    # Get unique map IDs
    map_ids = set(feature['properties']['map'] for feature in maps['features'])
    
    mvf_folder = "/home/kevinmeng/workspace/mappedin/VPS/MVF/YYC"
    with open(os.path.join(mvf_folder, "floor.geojson"), 'r') as f:
        floors = json.load(f)
    
    # Create mapping of floor IDs to names
    floor_names = {
        floor['properties']['id']: floor['properties']['id']
        for floor in floors['features']
    }
    
    # Create OSM file for each floor
    for map_id in map_ids:
        map_id = f"f_{map_id}"
        if map_id in floor_names:
            floor_name = floor_names[map_id]
            print(f"Processing floor: {floor_name} (ID: {map_id})")
            create_osm_for_floor(map_id, floor_name, mvf_folder)
        else:
            print(f"Warning: No floor name found for map ID: {map_id}")

if __name__ == "__main__":
    main()