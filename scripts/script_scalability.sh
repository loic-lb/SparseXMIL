#!/bin/bash

EXP_PATH="./BRCA/scalability/train_gpu_mem_time"
PATCH_DIR="./BRCA/subtyping/patches"
FEATS_DIR="./BRCA/subtyping/feats/pt_files/"
SLIDE_LABELS="./BRCA/subtyping/dataset_brca.csv"
BATCH_SIZE=(1 4 8 16 32)
PERC_TILES=0.2 # To set to 0. when not using instance sampling
SPARSE_MAP_DOWN=256
MODEL=("xmil" "transmil" "attention" "sparseconvmil" "dgcn" "average" "dense_xmil" "nic")
SEED=(128 256 512)
for seed in "${SEED[@]}"
{
    exp_path="$EXP_PATH""_seed_""$seed"
    mkdir "$exp_path"
    for batch_size in "${BATCH_SIZE[@]}"
    {
        exp_path_batch="$exp_path""/batch_size_""$batch_size"
        mkdir "$exp_path_batch"
        for model in "${MODEL[@]}"
        {
          echo "Benchmarking gpu memory/elapsed time $model..."
          nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv \
          -l 1 > "$exp_path_batch""/""$model""_gpu_mem_log.csv" & nvidia_smi_pid=$!
          python ./benchmark_gpu_mem_time.py --experiment_path "$exp_path_batch" --feats_folder $FEATS_DIR \
          --patches_folder $PATCH_DIR --slides_label_filepath $SLIDE_LABELS --batch_size $batch_size \
          --perc_tiles_per_wsi $PERC_TILES --sparse-map-downsample $SPARSE_MAP_DOWN --model $model \
          --seed $seed --training # remove training tag when computing for inference
          kill $nvidia_smi_pid
          echo "Done."
        }
    }
}
