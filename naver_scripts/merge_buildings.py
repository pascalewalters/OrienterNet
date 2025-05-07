import glob
import os
import json
import shutil
import dill


BUILDINGS = ["coex", "gangnam", "hyundai"]
DATA_FOLDER = "/home/pascale/Documents/VPS/WayIL/Naver/"
NEW_IMAGE_FOLDER = "/home/pascale/Documents/VPS/WayIL/Naver/merged_images/"

if not os.path.exists(NEW_IMAGE_FOLDER):
    os.makedirs(NEW_IMAGE_FOLDER)

split_files = []
geojson_files = []

for building in BUILDINGS:
    split_files.extend(
        glob.glob(
            os.path.join(DATA_FOLDER, building, "orienternet_data", "*splits.json")
        )
    )
    geojson_files.extend(
        glob.glob(os.path.join(DATA_FOLDER, building, "orienternet_data", "*.geojson"))
    )

merged_splits = {"train": {}, "val": {}, "test": {}}
for split_file in split_files:
    with open(split_file, "r") as f:
        data = json.load(f)

    for key in data.keys():
        for map_id in data[key].keys():
            if map_id not in merged_splits[key]:
                merged_splits[key][map_id] = []
            merged_splits[key][map_id].extend(data[key][map_id])

            # for image_name in data[key][map_id]:
            #     # Copy the image into new image folder
            #     if "coex" in split_file:
            #         image_path = os.path.join(
            #             DATA_FOLDER, "coex", "Coex", "1F", "release"
            #         )
            #         if key == "train":
            #             image_path = os.path.join(
            #                 image_path,
            #                 "mapping",
            #                 "sensors",
            #                 "records_data",
            #                 "images",
            #                 f"{image_name}.jpg",
            #             )
            #         elif key == "val":
            #             image_path = os.path.join(
            #                 image_path,
            #                 "validation",
            #                 "sensors",
            #                 "records_data",
            #                 "images",
            #                 f"{image_name}.jpg",
            #             )
            #         else:
            #             image_path = os.path.join(
            #                 image_path,
            #                 "test",
            #                 "sensors",
            #                 "records_data",
            #                 "images",
            #                 f"{image_name}.jpg",
            #             )
            #         assert os.path.exists(
            #             image_path
            #         ), f"Image path does not exist: {image_path}"

            #     elif "gangnam" in split_file:
            #         image_path = os.path.join(DATA_FOLDER, "gangnam", "GangnamStation")
            #         if "B1" in split_file:
            #             floor = "B1"
            #         else:
            #             floor = "B2"

            #         if key == "train":
            #             search_string = os.path.join(
            #                 image_path,
            #                 floor,
            #                 "release",
            #                 "mapping",
            #                 "sensors",
            #                 "records_data",
            #                 "*",
            #                 "images",
            #                 f"{image_name}.jpg",
            #             )
            #         else:
            #             search_string = os.path.join(
            #                 image_path,
            #                 floor,
            #                 "release",
            #                 "validation",
            #                 "sensors",
            #                 "records_data",
            #                 "*",
            #                 "galaxy",
            #                 f"{image_name}.jpg",
            #             )
            #         image_path = glob.glob(search_string)
            #         # assert (
            #         #     len(image_path) == 1
            #         # ), f"Image path does not exist: {search_string}"
            #         image_path = image_path[0]

            #     else:
            #         image_path = os.path.join(
            #             DATA_FOLDER, "hyundai", "HyundaiDepartmentStore"
            #         )
            #         if "1F" in split_file:
            #             floor = "1F"
            #         elif "4F" in split_file:
            #             floor = "4F"
            #         else:
            #             floor = "B1"
            #         if key == "train":
            #             search_string = os.path.join(
            #                 image_path,
            #                 floor,
            #                 "release",
            #                 "mapping",
            #                 "sensors",
            #                 "records_data",
            #                 "*",
            #                 "images",
            #                 f"{image_name}.jpg",
            #             )
            #         else:
            #             search_string = os.path.join(
            #                 image_path,
            #                 floor,
            #                 "release",
            #                 "validation",
            #                 "sensors",
            #                 "records_data",
            #                 "*",
            #                 "galaxy",
            #                 f"{image_name}.jpg",
            #             )

            #         image_path = glob.glob(search_string)
            #         assert (
            #             len(image_path) == 1
            #         ), f"Image path does not exist: {search_string}"
            #         image_path = image_path[0]

            #     # print(f"Copying {image_path} to {NEW_IMAGE_FOLDER}")
            #     # Copy the image to the new folder
            #     # new_image_path = os.path.join(
            #     #     NEW_IMAGE_FOLDER, os.path.basename(image_path)
            #     # )
            #     # shutil.move(image_path, new_image_path)

# # Save the merged splits to a new JSON file
# with open(os.path.join(DATA_FOLDER, "merged_splits.json"), "w") as f:
#     json.dump(merged_splits, f, indent=4)

# exit()

geojson_data = []
for geojson_file in geojson_files:
    with open(geojson_file, "r") as f:
        data = json.load(f)

    geojson_data.extend(data["features"])

data["features"] = geojson_data
# Save the merged geojson to a new file
with open(os.path.join(DATA_FOLDER, "merged.geojson"), "w") as f:
    json.dump(data, f, indent=4)
