import json
import os
import shutil
from pathlib import Path

def organize_photos_by_map():
    # Define paths
    base_dir = "/home/kevinmeng/workspace/mappedin/VPS/Mappedin_VPS_Data-20250127T163206Z-001/Mappedin_VPS_Data/YYC_VPS"
    photos_dir = os.path.join(base_dir, "photos")
    geojson_path = os.path.join(base_dir, "combined-output.geojson")
    output_base_dir = os.path.join(base_dir, "photos_by_map")

    # Create the output base directory if it doesn't exist
    os.makedirs(output_base_dir, exist_ok=True)

    # Read and parse the geojson file
    with open(geojson_path, 'r') as f:
        data = json.load(f)

    # Dictionary to keep track of photos count per map
    map_counts = {}

    # Process each feature
    for feature in data['features']:
        map_id = feature['properties']['map']
        image_name = feature['properties']['imageUrl']
        
        # Create map directory if it doesn't exist
        map_dir = os.path.join(output_base_dir, map_id)
        os.makedirs(map_dir, exist_ok=True)

        # Source and destination paths
        source_path = os.path.join(photos_dir, image_name)
        dest_path = os.path.join(map_dir, image_name)

        # Copy the file if it exists
        if os.path.exists(source_path):
            shutil.copy2(source_path, dest_path)
            map_counts[map_id] = map_counts.get(map_id, 0) + 1
        else:
            print(f"Warning: Image not found: {image_name}")

    # Print statistics
    print("\nPhoto organization complete!")
    print("\nStatistics:")
    print("-----------")
    for map_id, count in map_counts.items():
        print(f"Map {map_id}: {count} photos")
    print(f"\nTotal maps: {len(map_counts)}")
    print(f"Total photos organized: {sum(map_counts.values())}")

if __name__ == "__main__":
    organize_photos_by_map()