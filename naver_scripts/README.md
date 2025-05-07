# Naver Scripts for OrienterNet Preprocessing

Naver Labs has published an indoor localization dataset on their website ([link](https://www.naverlabs.com/en/storyDetail/211)).
They use this dataset for training [their WayIL method](https://rllab-snu.github.io/projects/WayIL/).
Since the code for WayIL is not released, I have used this data to train [OrienterNet](https://github.com/pascalewalters/OrienterNet).
The WayIL paper reveals that they got working results with the OrienterNet method and the Naver Labs dataset (results in Table I).

This folder contains scripts to preprocess the Naver Labs data into a format for training with OrienterNet.

## Preprocessing Steps

The following venues are in the Naver Labs dataset:

* COEX 1F
* Gangnam Station B1
* Gangnam Station B2
* Hyundai Department Store 1F
* Hyundai Department Store 4F
* Hyundai Department Store B1

### 1. Map in Self-serve

Find an image of the venue's floorplan online and create a map in Self-serve.

* [Coex](https://app.mappedin.com/editor/edit/67f9209d00fb95000b48d140)
* [Gangnam Station](https://app.mappedin.com/editor/edit/6807f7c28a135d000b0f9612)
* [Hyundai Department Store](https://app.mappedin.com/editor/edit/6801374252aee2000ba6d2d1)

#### A note on floorplans

* Naver Maps is similar to Google Maps in Korea. They have these venues mapped in their platform, however the UI isn't great and it's all in Korean
* With some consultation from Yunbo, we determined that the Hyundai Department Store is the Pangyo location (https://www.ehyundai.com/DP/lang/en/DEP000002.do?branchCd=B00148000)
* South Korea has restrictions on geographic data, which is why they don't have Google Maps or Apple Maps there. Additionally, searching for locations in Self-serve is difficult, because our georeference address lookup does not have South Korean addresses
    * https://en.wikipedia.org/wiki/Restrictions_on_geographic_data_in_South_Korea

### 2. Download the data

For example:

```
wget https://challenge.naverlabs.com/kapture/GangnamStation_B1_release_mapping.tar.gz
```

### 3. Generate a PLY file from the trajectory

Run the script in `get_ply_from_trajectory.py`. Set `DATA_DIR` to the folder containing the trajectory.txt file.

### 4. Load the map and trajectory in the ECEF transform tool

You'll need to get the map ID, key, and secret from the Developers tab in Self-serve.
Load the .ply from the previous step in the tool, then get the transform parameters.

### 5. Get indoor data in real-world coordinates

Run the script in `apply_ecef_transform.py` to get combined-output.geojson and image_splits.json.
You'll need to get the ECEF transforms for the train, validation, and test splits.

### 6. Load the MVF data

Run the script in `load_map_selfserve.py` to get raster map .npy files, MVF data pickle, area_index_mapping.json and line_index_mapping.json.
