# Log Evaluator (Substage C) GRPO training with the VERL framework.
# Tested on the hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 image.
#
# Prerequisites:
#   1. Clone/install VERL: https://github.com/volcengine/verl
#   2. Prepare train.parquet / test.parquet with `process_data_for_parquet.py`
#   3. Download the base policy model (paper uses Qwen3-0.6B; this script shows Qwen3-4B).
#
# Edit the paths below to match your environment before running.

# ---- Paths (edit me) ----
VERL_MAIN=${VERL_MAIN:-/path/to/verl/trainer/main_ppo.py}
DATA_DIR=${DATA_DIR:-../../../data/stage2_log_evaluator}        # contains train.parquet / test.parquet
REWARD_FN=${REWARD_FN:-./reward_router.py}
BASE_MODEL=${BASE_MODEL:-/path/to/Qwen3-4B-Instruct-2507}

export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_$SLURM_JOBID}
# export WANDB_API_KEY=<your_wandb_key>   # optional, only if logging to W&B

set -x

python3 "${VERL_MAIN}" \
    algorithm.adv_estimator=grpo \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/test.parquet" \
    data.train_batch_size=377 \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    custom_reward_function.path="${REWARD_FN}" \
    custom_reward_function.name="compute_score_llmroute" \
    actor_rollout_ref.model.path="${BASE_MODEL}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console", "tensorboard"]' \
    trainer.project_name='verl_grpo_decor_log_evaluator' \
    trainer.experiment_name='qwen3_route_log_evaluator' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.total_epochs=10 $@
