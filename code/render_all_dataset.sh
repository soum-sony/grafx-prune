#!/bin/bash

DATASET_DIR="/home/soumya/Mixing_secrets_test_set/dataset"
GPU_COUNT=4  # Number of GPUs available
GPU_INDEX=0  # Start GPU index

# Loop through all song directories
for SONG in "$DATASET_DIR"/*/; do
    SONG_NAME=$(basename "$SONG")

    # Run the command in the background on a specific GPU
    CUDA_VISIBLE_DEVICES=$GPU_INDEX python3 train.py \
        config=prune_hybrid_1e_2 \
        dataset=mixing_secrets_excerpts \
        song="$SONG_NAME" &

    # Update GPU index in a round-robin fashion
    GPU_INDEX=$(( (GPU_INDEX + 1) % GPU_COUNT ))

    # To prevent too many processes launching at once, you can limit the number of concurrent jobs:
    while (( $(jobs | wc -l) >= GPU_COUNT )); do
        sleep 5  # Wait a bit before checking again
    done
done

# Wait for all background jobs to finish
wait
