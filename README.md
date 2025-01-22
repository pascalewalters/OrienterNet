<p align="center">
  <h1 align="center"><ins>OrienterNet</ins><br>Visual Localization in 2D Public Maps<br>with Neural Matching</h1>
  <p align="center">
    <a href="https://psarlin.com/">Paul-Edouard&nbsp;Sarlin</a>
    ·
    <a href="https://danieldetone.com/">Daniel&nbsp;DeTone</a>
    ·
    <a href="https://scholar.google.com/citations?user=WhISCE4AAAAJ&hl=en">Tsun-Yi&nbsp;Yang</a>
    ·
    <a href="https://scholar.google.com/citations?user=Ta4TDJoAAAAJ&hl=en">Armen&nbsp;Avetisyan</a>
    ·
    <a href="https://scholar.google.com/citations?hl=en&user=49_cCT8AAAAJ">Julian&nbsp;Straub</a>
    <br>
    <a href="https://tom.ai/">Tomasz&nbsp;Malisiewicz</a>
    ·
    <a href="https://scholar.google.com/citations?user=484sccEAAAAJ&hl=en">Samuel&nbsp;Rota&nbsp;Bulo</a>
    ·
    <a href="https://scholar.google.com/citations?hl=en&user=MhowvPkAAAAJ">Richard&nbsp;Newcombe</a>
    ·
    <a href="https://scholar.google.com/citations?hl=en&user=CxbDDRMAAAAJ">Peter&nbsp;Kontschieder</a>
    ·
    <a href="https://scholar.google.com/citations?user=AGoNHcsAAAAJ&hl=en">Vasileios&nbsp;Balntas</a>
  </p>
  <h2 align="center">CVPR 2023</h2>
  <h3 align="center">
    <a href="https://sarlinpe-orienternet.hf.space">Web demo</a>
    | <a href="https://colab.research.google.com/drive/1zH_2mzdB18BnJVq48ZvJhMorcRjrWAXI?usp=sharing">Colab</a>
    | <a href="https://arxiv.org/pdf/2304.02009.pdf">Paper</a> 
    | <a href="https://psarlin.com/orienternet">Project Page</a>
    | <a href="https://youtu.be/wglW8jnupSs">Video</a>
  </h3>
  <div align="center"></div>
</p>
<p align="center">
    <a href="https://psarlin.com/orienternet"><img src="assets/teaser.svg" alt="teaser" width="60%"></a>
    <br>
    <em>OrienterNet is a deep neural network that can accurately localize an image<br>using the same 2D semantic maps that humans use to orient themselves.</em>
</p>

##

This repository hosts the source code for OrienterNet, a research project by Meta Reality Labs. OrienterNet leverages the power of deep learning to provide accurate positioning of images using free and globally-available maps from OpenStreetMap. As opposed to complex existing algorithms that rely on 3D point clouds, OrienterNet estimates a position and orientation by matching a neural Bird's-Eye-View with 2D maps.

## Installation

OrienterNet requires Python >= 3.8 and [PyTorch](https://pytorch.org/).  To run the demo, clone this repo and install the minimal requirements:

```bash
git clone https://github.com/facebookresearch/OrienterNet
python -m pip install -r requirements/demo.txt
```

To run the evaluation and training, install the full requirements:

```bash
python -m pip install -r requirements/full.txt
```

## Demo ➡️ [![hf](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-md.svg)](https://sarlinpe-orienternet.hf.space) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zH_2mzdB18BnJVq48ZvJhMorcRjrWAXI?usp=sharing)

Try our minimal demo - take a picture with your phone in any city and find its exact location in a few seconds!
- [Web demo with Gradio and Huggingface Spaces](https://sarlinpe-orienternet.hf.space)
- [Cloud demo with Google Colab](https://colab.research.google.com/drive/1zH_2mzdB18BnJVq48ZvJhMorcRjrWAXI?usp=sharing)
- Local demo with Jupyter nobook [`demo.ipynb`](./demo.ipynb)

<p align="center">
    <a href="https://huggingface.co/spaces/sarlinpe/OrienterNet"><img src="assets/demo.jpg" alt="demo" width="60%"></a>
    <br>
    <em>OrienterNet positions any image within a large area - try it with your own images!</em>
</p>

## Indoor Notes

### MVF to OpenStreetMap (OSM)

First, convert the MVF to the OSM-like format. Download the Den 1880 MVF from [here](https://drive.google.com/file/d/1_y_rN2MG-SKr4CGt4Mrmwv_0nigQWmkl/view?usp=drive_link). Extract the .zip file into a folder of geojson files. Then, run the script:

```
python map_to_osm.py
```

This only reads in annotations and obstructions from the MVF. Annotations are nodes (points), and obstructions are ways (lines) and areas (polygons). In OSM, ways and areas are collections of nodes.

The script generates a file den_osm.json.

### Run Inference with OrienterNet

```
python warp_image.py
```

You'll need to set the value of `image_path` to the path to an input image.

Notes:
* In the MapEncoder model, `embedding dim = 16`, `output dim = 8`, `num classes = {'areas': 7, 'ways': 10, 'nodes': 33}`
* These classes are set in `maploc.osm.parser.Patterns`
  * I think they are based on tags that come from OSM
* In OSM, areas are made up of relations. These can be nodes or ways. In the MVF, areas are just polygons, so I generate areas only from nodes.
  * You can see this in `maploc.osm.data.MapArea.from_relation`
  * From what I've seen, MVF polygons are filled, so I don't consider any inner nodes
* Since I didn't consider any OSM classes when parsing the MVF, they are only given the classes `node`, `way`, and `area`
  * This is in `maploc.osm.data.MapData.from_osm`
  * When rasterized, areas are `parking`, lines are `playground`, and nodes are `grass`. This is in `maploc.osm.raster.render_raster_masks`
  * These classes don't mean anything. I just picked random classes so that inference would run
* Instead of querying OSM, read in the json from the first step in `maploc.osm.tiling.TilingManager.from_bbox`

Next steps:
* Train the method end-to-end with our indoor data
  * Need to get it in the same format as this method
* Take a look at the classes from OSM, select classes that are meaningful for MVF
* The method automatically gets a bounding box from an initial guess of the address. This makes sense for an outdoor query, since the area of interest is large. For the indoor case, we could get a bounding box around the map data

## Evaluation

#### Mapillary Geo-Localization dataset

<details>
<summary>[Click to expand]</summary>

To obtain the dataset:

1. Create a developper account at [mapillary.com](https://www.mapillary.com/dashboard/developers) and obtain a free access token.
2. Run the following script to download the data from Mapillary and prepare it:

```bash
python -m maploc.data.mapillary.prepare --token $YOUR_ACCESS_TOKEN
```

By default the data is written to the directory `./datasets/MGL/`. Then run the evaluation with the pre-trained model:

```bash
python -m maploc.evaluation.mapillary --experiment OrienterNet_MGL model.num_rotations=256
```

This downloads the pre-trained models if necessary. The results should be close to the following:

```
Recall xy_max_error: [14.37, 48.69, 61.7] at (1, 3, 5) m/°
Recall yaw_max_error: [20.95, 54.96, 70.17] at (1, 3, 5) m/°
```

This requires a GPU with 11GB of memory. If you run into OOM issues, consider reducing the number of rotations (the default is 256):

```bash
python -m maploc.evaluation.mapillary [...] model.num_rotations=128
```

To export visualizations for the first 100 examples:

```bash
python -m maploc.evaluation.mapillary [...] --output_dir ./viz_MGL/ --num 100 
```

To run the evaluation in sequential mode:

```bash
python -m maploc.evaluation.mapillary --experiment OrienterNet_MGL --sequential model.num_rotations=256
```
The results should be close to the following:
```
Recall xy_seq_error: [29.73, 73.25, 91.17] at (1, 3, 5) m/°
Recall yaw_seq_error: [46.55, 88.3, 96.45] at (1, 3, 5) m/°
```
The sequential evaluation uses 10 frames by default. To increase this number, add:
```bash
python -m maploc.evaluation.mapillary [...] chunking.max_length=20
```


</details>

#### KITTI dataset

<details>
<summary>[Click to expand]</summary>

1. Download and prepare the dataset to `./datasets/kitti/`:

```bash
python -m maploc.data.kitti.prepare
```

2. Run the evaluation with the model trained on MGL:

```bash
python -m maploc.evaluation.kitti --experiment OrienterNet_MGL model.num_rotations=256
```

You should expect the following results:

```
Recall directional_error: [[50.33, 85.18, 92.73], [24.38, 56.13, 67.98]] at (1, 3, 5) m/°
Recall yaw_max_error: [29.22, 68.2, 84.49] at (1, 3, 5) m/°
```

You can similarly export some visual examples:

```bash
python -m maploc.evaluation.kitti [...] --output_dir ./viz_KITTI/ --num 100
```

To run in sequential mode:
```bash
python -m maploc.evaluation.kitti --experiment OrienterNet_MGL --sequential model.num_rotations=256
```
with results:
```
Recall directional_seq_error: [[81.94, 97.35, 98.67], [52.57, 95.6, 97.35]] at (1, 3, 5) m/°
Recall yaw_seq_error: [82.7, 98.63, 99.06] at (1, 3, 5) m/°
```

</details>

#### Aria Detroit & Seattle

We are currently unable to release the dataset used to evaluate OrienterNet in the CVPR 2023 paper.

## Training

#### MGL dataset

We trained the model on the MGL dataset using 3x 3090 GPUs (24GB VRAM each) and a total batch size of 12 for 340k iterations (about 3-4 days) with the following command:

```bash
python -m maploc.train experiment.name=OrienterNet_MGL_reproduce
```

Feel free to use any other experiment name. Configurations are managed by [Hydra](https://hydra.cc/) and [OmegaConf](https://omegaconf.readthedocs.io) so any entry can be overridden from the command line. You may thus reduce the number of GPUs and the batch size via:

```bash
python -m maploc.train experiment.name=OrienterNet_MGL_reproduce \
  experiment.gpus=1 data.loading.train.batch_size=4
```

Be aware that this can reduce the overall performance. The checkpoints are written to `./experiments/experiment_name/`. Then run the evaluation:

```bash
# the best checkpoint:
python -m maploc.evaluation.mapillary --experiment OrienterNet_MGL_reproduce
# a specific checkpoint:
python -m maploc.evaluation.mapillary \
    --experiment OrienterNet_MGL_reproduce/checkpoint-step=340000.ckpt
```

#### KITTI

To fine-tune a trained model on the KITTI dataset:

```bash
python -m maploc.train experiment.name=OrienterNet_MGL_kitti data=kitti \
    training.finetune_from_checkpoint='"experiments/OrienterNet_MGL_reproduce/checkpoint-step=340000.ckpt"'
```

## Interactive development

We provide several visualization notebooks:

- [Visualize predictions on the MGL dataset](./notebooks/visualize_predictions_mgl.ipynb)
- [Visualize predictions on the KITTI dataset](./notebooks/visualize_predictions_kitti.ipynb)
- [Visualize sequential predictions](./notebooks/visualize_predictions_sequences.ipynb)

## OpenStreetMap data

<details>
<summary>[Click to expand]</summary>

To make sure that the results are consistent over time, we used OSM data downloaded from [Geofabrik](https://download.geofabrik.de/) in November 2021. By default, the dataset scripts `maploc.data.[mapillary,kitti].prepare` download pre-generated raster tiles. If you wish to use different OSM classes, you can pass `--generate_tiles`, which will download and use our prepared raw `.osm` XML files.

You may alternatively download more recent files from [Geofabrik](https://download.geofabrik.de/). Download either compressed XML files as `.osm.bz2` or binary files `.osm.pbf`, which need to be converted to XML files `.osm`, for example using Osmium: ` osmium cat xx.osm.pbf -o xx.osm`.

</details>

## License

The MGL dataset is made available under the [CC-BY-SA](https://creativecommons.org/licenses/by-sa/4.0/) license following the data available on the Mapillary platform. The model implementation and the pre-trained weights follow a [CC-BY-NC](https://creativecommons.org/licenses/by-nc/2.0/) license. [OpenStreetMap data](https://www.openstreetmap.org/copyright) is licensed under the [Open Data Commons Open Database License](https://opendatacommons.org/licenses/odbl/).

## BibTex citation

Please consider citing our work if you use any code from this repo or ideas presented in the paper:
```
@inproceedings{sarlin2023orienternet,
  author    = {Paul-Edouard Sarlin and
               Daniel DeTone and
               Tsun-Yi Yang and
               Armen Avetisyan and
               Julian Straub and
               Tomasz Malisiewicz and
               Samuel Rota Bulo and
               Richard Newcombe and
               Peter Kontschieder and
               Vasileios Balntas},
  title     = {{OrienterNet: Visual Localization in 2D Public Maps with Neural Matching}},
  booktitle = {CVPR},
  year      = {2023},
}
```


