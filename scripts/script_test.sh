#!/bin/bash

EXP_PATH="./BRCA/subtyping/"
PATCH_DIR="./BRCA/pre_processing/patches"
FEATS_DIR="./BRCA/feats/pt_files"
SLIDE_LABELS="./BRCA/subtyping/dataset_brca.csv"
BATCH_SIZE=16 #set to 1 for second training setting with attentionMIL, TransMIL and GCN-MIL
PERC_TILES=0.2 #set to 0 for second training setting with attentionMIL, TransMIL and GCN-MIL(take all tiles)
SPARSE_MAP_DOWN=256
MODEL=("xmil" "transmil"  "attention" "sparseconvmil" "dgcn" "average") # remove models you don't want to test
TASK="brca_subtyping" # task name (either brca_subtyping, brca_thrd, brca_mhrd, brca_T1vsRest, lung_subtyping or kidney_subtyping)
TAG_TTA=", non_aug" # tag in the experiment to indicate if test time augmentation was used (leave to "" if not used)

for model in "${MODEL[@]}"
{
  echo "Testing $model..."
  python ./testing.py --experiment_folder $EXP_PATH --experiment_name "final, $TASK, $model$TAG_TTA"  \
    --feats_folder $FEATS_DIR --patches_folder $PATCH_DIR --slides_label_filepath $SLIDE_LABELS \
    --perc_tiles_per_wsi $PERC_TILES --sparse-map-downsample $SPARSE_MAP_DOWN --batch_size $BATCH_SIZE -model $model \
    --use_ema \
    --test_time_augmentation # Comment to remove test time augmentation, --shuffle_locations --shuffle_mode "absolute_positions" # uncomment for sensitivity analysis
  echo "Done."
}
