#!/bin/bash
gpus=$1

# Create screens
for i in $gpus; do
    screen_name="scr${i}"
    (sleep 1; screen -d;) &
    screen -S "$screen_name"
done

# Run something in each screen
for i in $gpus; do
    COMMAND="source activate tensorflow; export CUDA_VISIBLE_DEVICES=${i}; ./run_scripts/run_sequentially.sh; source deactivate;"
    screen_name="scr${i}"
    screen -S "$screen_name" -X stuff "$COMMAND"$'\n'
done
