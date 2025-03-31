# source ~/.bashrc
# export TRITON_CACHE_DIR=/tmp/triton_cache
# micromamba activate openrlhf
# cd /fs/nexus-scratch/zli12321/active-topic-modeling/deepresearch/openrlhf_rl
module load cuda
module load gcc/11.2.0

set -x

# Start ray
# wandn key 5e11bfa8cf4062940486d279ecd9e70617d4ac7a
# export RAY_TMPDIR=/tmp/r
ray start --head --node-ip-address 0.0.0.0 --num-gpus 4


ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json='{"working_dir": "/fs/nexus-scratch/zli12321/active-topic-modeling/deepresearch/openrlhf_rl/scripts"}' \
  -- python3 -m openrlhf.cli.train_ppo_ray \
  --ref_num_nodes 1 \
  --ref_num_gpus_per_node 1 \
  --reward_num_nodes 0 \
  --reward_num_gpus_per_node 0 \
  --actor_num_nodes 1 \
  --actor_num_gpus_per_node 1 \
  --vllm_num_engines 1 \
  --vllm_tensor_parallel_size 1 \
  --pretrain /fs/nexus-scratch/zli12321/active-topic-modeling/LLaMA-Factory/saves/qwen-0.5B-mixed_reasoning/full/sft \
  --remote_rm_url /fs/nexus-scratch/zli12321/active-topic-modeling/deepresearch/openrlhf_rl/reward_functions/rougeL_reward.py \
  --save_path /fs/clip-scratch/lizongxia/grpo_weights/el5/rougeL \
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
  --prompt_data zli12321/el5 \
  --apply_chat_template \
  --input_key prompt \
  --label_key ground_truth \
  --normalize_reward \
  --adam_offload \
  --flash_attn \
  --gradient_checkpointing \
  --packing_samples \
  --use_wandb 5e11bfa8cf4062940486d279ecd9e70617d4ac7a \
  --save_steps 12 \
  --enable_prefix_caching