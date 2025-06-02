# SparseXMIL

This repository contains the code for the paper "SparseXMIL: Leveraging spatial convolutions for context-aware and memory-efficient classification of whole slide images in digital pathology".

All the materials to reproduce the experiments are available in this [onedrive repository](https://centralesupelec-my.sharepoint.com/:u:/g/personal/loic_le-bescond_centralesupelec_fr/EUT0qiy0t1lIppKHN3_PGTQBnO0X_et0tElqxP860YsvzA?e=cR4pWq).

## Docker/Singularity

A Dockerfile and Singularity recipe are available in the repository to help you set up the environment.
You may build the Docker image with the following command:

```
docker build -t sparsexmil ./docker/
```

You may then run the Docker image with the following command:

```
docker run --gpus all -it -v $(pwd):/SparseXMIL sparsexmil
```

Alernatively, you may build the Singularity image with the following command:

```
singularity build sparsexmil.sif ./docker/sparsexmil.def
```

You may then run the Singularity image with the following command:

```
singularity run --no-home --nv -B $(pwd):/SparseXMIL sparsexmil.sif
```

## Installation

The following instructions have been tested with cuda 11.3 on an NVIDIA A6000 GPU with conda. For other versions, you
may need to adapt the installation instructions (see [here](https://github.com/shwoo93/MinkowskiEngine/tree/bbc30ef581ea6deb505976b663f5fc2358a83749) for more details on MinkowskiEngine installation with 
other cuda versions).

To set up the environnement, we recommend using conda. You may install then the dependencies with the following commands:

First start by creating a conda environnement:

```
conda create -n sparsexmil python=3.8
conda activate sparsexmil
```

Then install pytorch, openblas and cudatoolkit:

```
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install openblas-devel==0.3.2 -c anaconda
conda install cudatoolkit-dev==11.3.1 -c conda-forge
```

To check if the installation of cudatooolkit is correct, you can run the following command:

```
nvcc --version
```

Then, you may install openslide:

```
conda install openslide==3.4.1 -c conda-forge
```

We used a modified version of MinkowskiEngine, which is available in the submodule MinkowskiEngine. To install it, you may run the following commands:
```
git submodule update --init --recursive
git submodule update --recursive --remote
cd MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```

Finally, you may install the other dependencies:

```
pip install scikit-image==0.21.0 scikit-learn==1.2.2 pandas==2.0.2 h5py==3.8.0 opencv-python==4.7.0.72 openslide-python==1.2.0 matplotlib==3.7.1
```

To run TransMIL implementation, you may additionally install the following dependencies:

```
pip install nystrom_attention==0.0.11
```

To run GCN-MIL implementation, you may additionally install the following dependencies:

```
pip install torch_geometric==2.4.0 torch_scatter==2.1.2 torch_cluster==1.6.3
```

An environment file is also available in the repository as reference.

## Data

The data used in this paper all come from the [TCGA database](https://portal.gdc.cancer.gov/).

* For BRCA, we used the data from the [TCGA-BRCA](https://portal.gdc.cancer.gov/projects/TCGA-BRCA) project.
* For LUNG, we used the data from the [TCGA-LUAD](https://portal.gdc.cancer.gov/projects/TCGA-LUAD) and 
[TCGA-LUSC](https://portal.gdc.cancer.gov/projects/TCGA-LUSC) projects.
* For KIDNEY, we used the data from the [TCGA-KIRC](https://portal.gdc.cancer.gov/projects/TCGA-KIRC), 
[TCGA-KIRP](https://portal.gdc.cancer.gov/projects/TCGA-KIRP),
and [TCGA-KICH](https://portal.gdc.cancer.gov/projects/TCGA-KICH) projects.

To get a precise listing of the slide used in each experiment, you may refer to the corresponding _dataset.csv_ files 
in the [onedrive repository](https://centralesupelec-my.sharepoint.com/:u:/g/personal/loic_le-bescond_centralesupelec_fr/EUT0qiy0t1lIppKHN3_PGTQBnO0X_et0tElqxP860YsvzA?e=cR4pWq).

## Preprocessing

All the preprocessing code is available in the `tile_extraction` folder.

To extract the patches from the slides, you may run the following command:

```
python ./tile_extraction/create_patches_fp.py --source <path_to_slides> --save_dir <path_to_save_dir> \
--dataset <path_to_csv_dataset_file> --extension .svs --patch --seg --stitch 
```

For convenience, the patches extracted for each experiment are available in the [onedrive repository](https://centralesupelec-my.sharepoint.com/:u:/g/personal/loic_le-bescond_centralesupelec_fr/EUT0qiy0t1lIppKHN3_PGTQBnO0X_et0tElqxP860YsvzA?e=cR4pWq).

To extract features vector from patch coordinates, you may run the following command:

```
python ./tile_extraction/extract_features_fp.py ./extract_features_fp.py --data_h5_dir <path_to_h5_files> \
 --data_slide_dir <path_to_slides> --slide_ext .svs --csv_path <path_to_csv_file> --feat_dir <path_to_feat_dir>
```

## Training

To train the model, you may run the following command:

```
python ./training.py ---experiment_folder <path_to_experiment_folder> --experiment_name <your_experiment_name> \
    --feats_folder <path_to_feat_dir> --patches_folder <path_to_h5_files> --perc_tiles_per_wsi 0.2 \
    --sparse-map-downsample 256 --slides_label_filepath <path_to_csv_dataset_file> --split_id <number_split_id> \
    --split <path_to_csv_split> --batch_size 16 --lr 2e-4 --epochs 100 --reg $REG --model <name_of_the_model> \
    --optimizer Adam --test_time_augmentation
```

The `--model` argument can take the following values:
* "xmil": for the SparseXMIL model
* "dense_xmil": for using Neural Image Compression with Xception architecture (same as SparseXMIL but with a dense architecture)
* "nic": for using Neural Image Compression with the architecture proposed in the paper
* "transmil": for the TransMIL model
* "dgcn": for the GCN-MIL model
* "sparseconvmil": for the SparseConvMIL model
*  "attention": for the Attention-MIL model
*  "average": for the Average-MIL model

A script to train all the experiments is available in the `scripts` folder. In the paper, when we refer to the 
"second training setting", it corresponds to setting batch_size to 1, perc_tiles_per_wsi to 0, and removing the
--test_time_augmentation argument.

For additional information on the arguments, you may refer to the `training.py` file.

## Evaluation

To evaluate the model, you may run the following command:

```
python ./testing.py --experiment_folder <path_to_folder_with_experiments> --experiment_name <your_experiment_name> \
    --feats_folder <path_to_feat_dir> --patches_folder <path_to_h5_files> --slides_label_filepath <path_to_csv_dataset_file> \
    --perc_tiles_per_wsi 0.2  --sparse-map-downsample 256 --batch_size 16 --model <name_of_the_model> --use_ema \
    --test_time_augmentation
```

The argument share the same meaning as for the training script. For the "second training setting", you ay also set 
perc_tiles_per_wsi to 0, the batch_size to 1, and remove the --test_time_augmentation argument.

To run sensitivity experiments, you may use the following arguments:

* `--shuffle_locations` to enable the shuffling of the patch locations
* `--shuffle_mode` which can be set either to "idx" to shuffle the patch locations by index or to "absolute_positions" 
to assign random coordinates to the patches

A script to evaluate all the experiments is available in the `scripts` folder.

For additional information on the arguments, you may refer to the `testing.py` file.

## Interpretation

To run the interpretation experiments, you may run the following command:

```
python interpret.py --experiment_folder <path_to_folder_with_experiments> --experiment_name <your_experiment_name> \
--feats_folder <path_to_feat_dir> --slide_folder <path_to_raw_slides> --patches_folder <path_to_h5_files> \
--slides_label_filepath <path_to_csv_dataset_file> --perc_tiles_per_wsi 0.2 --sparse-map-downsample 256 \
--split_id <number_split_to_use> --model <name_of_the_model> --annotated_data <path_to_annotate_data_csv_file> \
 --use_ema --test_time_augmentation --save_heatmaps --save_mask_tissues
```

Here again, the arguments share the same meaning as for the training and testing script. For "attention" and "dgcn" models,
you need to set the perc_tiles_per_wsi to 0, the batch_size to 1, and remove the --test_time_augmentation argument as we
used the second training setting for the interpretation of these models.

The argument `--annotated_data` is used to specify the csv file containing the id of the slides for the interpretation
experiments (to avoid running the interpretation experiments on all the slides). The csv file is available in the
[onedrive repository](https://centralesupelec-my.sharepoint.com/:u:/g/personal/loic_le-bescond_centralesupelec_fr/EUT0qiy0t1lIppKHN3_PGTQBnO0X_et0tElqxP860YsvzA?e=cR4pWq). The argument `--save_heatmaps` is used to save the heatmaps produced by the interpretation 
experiments as .npy files, and the argument `--save_mask_contours` similarly saves the mask of the tissue for 
each slide (both are necessary to run the evaluation script). 
You may remove these arguments when evaluation is not needed to save disk space. 

You may now evaluate the interpretation experiments by running the following command:

```
python eval_interpret.py --heatmap_path <path_to_heatmap_folder> --annotations_path <path_to_annotations_folder> \
--masks_path <path_to_masks_folder>
```

with `--heatmap_path` the path to the folder containing the heatmaps (.npy files) produced by the `interpret.py` script, 
and `--annotations_path` the path to the folder containing the ground truth annotations. Ground truth annotations can be
retrieved [here](https://sites.google.com/view/aipath-dataset/home/rcc-region-and-subtyping).

## Scalability

To run the scalability experiments, you may use the script `script_scalability.sh` available in the `scripts` folder.

## Reference

If you find this code useful in your research then please cite:

```
@unpublished{lebescond:hal-04531177,
  TITLE = {{SparseXMIL: Leveraging sparse convolutions for context-aware and memory-efficient classification of whole slide images in digital pathology}},
  AUTHOR = {Le Bescond, Lo{\"i}c and Lerousseau, Marvin and Andre, Fabrice and Talbot, Hugues},
  URL = {https://hal.science/hal-04531177},
  YEAR = {2025},
  MONTH = Jan,
  KEYWORDS = {Multiple Instance Learning ; Convolutional Neural Networks ; Computational Pathology ; Resource Efficient},
  HAL_ID = {hal-04531177},
  HAL_VERSION = {v2},
}
```

## License
XMIL is GNU AGPLv3 licensed, as found in the LICENSE file.
