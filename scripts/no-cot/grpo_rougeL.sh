# source ~/.bashrc
# export TRITON_CACHE_DIR=/tmp/triton_cache
# micromamba activate openrlhf
# cd /fs/nexus-scratch/zli12321/active-topic-modeling/deepresearch/openrlhf_rl
module load cuda
module load gcc/11.2.0

set -x
rm -r /tmp/ray/*
ls /tmp/ray/

# Start ray
# export RAY_TMPDIR=/tmp/r
export TRANSFORMERS_CACHE=/fs/clip-scratch/lizongxia
export HF_HOME=/fs/clip-scratch/lizongxia
export RAY_TMPDIR=/fs/clip-scratch/lizongxia/tmp/ray


ray start --head --node-ip-address 0.0.0.0 --num-gpus 4


ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json='{"working_dir": "long_form_rl/scripts/no-cot"}' \
  -- python3 -m openrlhf.cli.train_ppo_ray \
  --ref_num_nodes 1 \
  --ref_num_gpus_per_node 1 \
  --reward_num_nodes 0 \
  --reward_num_gpus_per_node 0 \
  --actor_num_nodes 1 \
  --actor_num_gpus_per_node 1 \
  --vllm_num_engines 1 \
  --vllm_tensor_parallel_size 1 \
  --pretrain Qwen/Qwen2.5-3B-Instruct \
  --remote_rm_url ../../reward_functions_no_cot/rougeL/rougeL_reward.py \
  --save_path ../../grpo_weights/el5/Qwen-3B-no-cot-mixed/rougeL \
  --micro_train_batch_size 4 \
  --train_batch_size 128 \
  --micro_rollout_batch_size 4 \
  --rollout_batch_size 256 \
  --n_samples_per_prompt 4 \
  --max_epochs 2 \
  --prompt_max_len 1024 \
  --max_samples 100000 \
  --generate_max_len 1024 \
  --init_kl_coef 1e-3 \
  --gamma 1.0 \
  --use_kl_loss \
  --kl_estimator k3 \
  --advantage_estimator group_norm \
  --zero_stage 3 \
  --bf16 \
  --actor_learning_rate 1e-6 \
  --prompt_data zli12321/mixed_long_form \
  --apply_chat_template \
  --input_key prompt \
  --label_key ground_truth \
  --normalize_reward \
  --adam_offload \
  --flash_attn \
  --gradient_checkpointing \
  --packing_samples \
  --use_wandb YOUR_WANDB_KEY_HERE \
  --save_steps -1 \
  --enable_prefix_caching