#!/bin/bash

EXP_PATH="./BRCA/subtyping/"
EXP_NAME="final, brca_subtyping, xmil"
PATCH_DIR="./BRCA/subtyping/patches"
FEATS_DIR="./BRCA/subtyping/feats/pt_files/"
SLIDE_LABELS="./BRCA/subtyping/dataset_brca.csv"
BATCH_SIZE=16 #set to 1 for second training setting with attentionMIL, TransMIL and GCN-MIL
PERC_TILES=0.2 #set to 0 for second training setting with attentionMIL, TransMIL and GCN-MIL(take all tiles)
SPARSE_MAP_DOWN=256
LR=2e-4
REG=1e-7
EPOCHS=100 #150 for averageMIL
MODEL="xmil"
OPTIMIZER="Adam"
SPLITS_FOLDER="./BRCA/subtyping/splits"

for i in {0..9}
do
    echo "Processing split $i"

    python ./training.py --experiment_folder $EXP_PATH --experiment_name "$EXP_NAME" --feats_folder $FEATS_DIR \
    --patches_folder $PATCH_DIR  --perc_tiles_per_wsi $PERC_TILES --sparse-map-downsample $SPARSE_MAP_DOWN \
    --slides_label_filepath $SLIDE_LABELS --split_id $i --split "$SPLITS_FOLDER/split_$i.csv" --batch_size $BATCH_SIZE \
    --lr $LR --epochs $EPOCHS --reg $REG --model $MODEL  --optimizer "$OPTIMIZER" \
    --test_time_augmentation # comment to remove test time augmentation (second training setting)

done