#!/bin/bash

# Define the list of specific script files.
scripts=("grpo_pedants.sh" "grpo_bertscore.sh" "grpo_bem.sh" "grpo_preferenceBert")

# Loop over each specified script file.
for script in "${scripts[@]}"; do
  echo "Submitting job for $script"
  srun --qos=huge-long --partition=clip --account=clip --time=30:00:00 \
       --gres=gpu:rtxa6000:4 --cpus-per-task=4 --mem=128GB bash "$script"
done
