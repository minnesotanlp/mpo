#!/usr/bin/env bash
set -euo pipefail

###############################################################################
#  CLI arguments
#     $1 – reward-model (RM) endpoint URL
#     $2 – meta-reward-model (MRM) endpoint URL
###############################################################################
if (( $# < 2 )); then
  echo "Usage: $0 <rm_address> <mrm_address>"
  exit 1
fi

rm_address=$1
mrm_address=$2

rm_params=32B
mrm_params=32B

exp_type="mpoppo"
rubric_type="iter0"
prompt="evaluation_rubric_real_iter_0.txt"

###############################################################################
#  Paths & constants
###############################################################################

# Use MPOPPO_ROOT if set; fallback to your explicit path
trl_dir="${MPOPPO_ROOT:-/lustre/fs0/scratch/zkim/Development/mpo-old}"
SCRIPT="$trl_dir/examples/scripts/mpoppo.py"

WANDB_ENTITY="iterater"
WANDB_PROJECT="mpoppo-new"
DATASET="essay_writing"
TASK="essay_writing"
PROMPT_DIR="$trl_dir/trl/extras/mpoppo/prompts/essay_writing"

###############################################################################
#  Main runner
###############################################################################
run_experiment() {
    local exp_type=$1         # mpoppo / mpogrpo / ppo …
    local rubric_type=$2      # e.g. iter0
    local rm_params=$3        # reward-model size
    local mrm_params=$4       # meta-reward-model size
    local prompt_name=$5      # prompt file

    # ------------------------------------------------------------------------
    #  GPU layout
    # ------------------------------------------------------------------------
    local CUDA_DEVICES="${CUDA_DEVICES_OVERRIDE:-0,1,2,3,4,5,6,7}"
    local ACC_CONFIG="${ACC_CONFIG_OVERRIDE:-$trl_dir/examples/accelerate_configs/deepspeed_zero2.yaml}"

    local num_processes="${NUM_PROCESSES_OVERRIDE:-$(awk -F',' '{print NF}' <<< "$CUDA_DEVICES")}"

    # ------------------------------------------------------------------------
    #  Naming & bookkeeping
    # ------------------------------------------------------------------------
    local policy_model="policy-1.5b-iclr"
    local model_name
    if [[ "$exp_type" == "mpogrpo" ]]; then
        model_name="${rubric_type}-${rm_params}_${mrm_params}"
    else
        model_name="${rubric_type}-${rm_params}"
    fi

    local output_dir="$trl_dir/models/${policy_model}/${TASK}/${exp_type}/${model_name}"
    local exp_name="${policy_model}-ew-${exp_type}-${model_name}"

    # gradient accumulation scaling
    local grad_acc_steps=16

    # MPOPPO/MPOGRPO interval
    local num_mpo_interval=99999999
    [[ "$exp_type" == "mpogrpo"  || "$exp_type" == "mpoppo" ]] && num_mpo_interval=2

    local _mrm_address=$mrm_address

    # ------------------------------------------------------------------------
    #  Display run-time configuration
    # ------------------------------------------------------------------------
    printf -- "==============  Experiment %s  ==============\n" "$exp_name"
    printf "CUDA_DEVICES       : %s\n" "$CUDA_DEVICES"
    printf "ACC_CONFIG         : %s\n" "$ACC_CONFIG"
    printf "num_processes      : %s\n" "$num_processes"
    printf "grad_acc_steps     : %s\n" "$grad_acc_steps"
    printf "num_mpo_interval   : %s\n" "$num_mpo_interval"
    printf "rm_address         : %s\n" "$rm_address"
    printf "mrm_address        : %s\n" "$_mrm_address"
    printf -- "==============================================\n\n"

    # ------------------------------------------------------------------------
    # Launch policy fine-tuning
    # ------------------------------------------------------------------------
    WANDB__SERVICE_WAIT=10 CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
    accelerate launch --config_file "$ACC_CONFIG" \
        --num_processes "$num_processes" \
        "$SCRIPT" \
        --dataset_name "$DATASET" \
        --task_name "$TASK" \
        --wandb_entity "$WANDB_ENTITY" \
        --wandb_project "$WANDB_PROJECT" \
        --exp_name "$exp_name" \
        --init_rm_prompt "$PROMPT_DIR/$prompt_name" \
        --output_dir "$output_dir" \
        --learning_rate 3e-6 \
        --num_ppo_epochs 4 \
        --num_mpo_interval "$num_mpo_interval" \
        --save_n_updates 2 \
        --num_mpo_samples 10 \
        --num_mini_batches 1 \
        --per_device_train_batch_size 2 \
        --dataloader_drop_last True \
        --gradient_accumulation_steps "$grad_acc_steps" \
        --local_rollout_forward_batch_size 24 \
        --total_episodes 10000 \
        --model_name_or_path "Qwen/Qwen2.5-1.5B-Instruct" \
        --sft_model_path   "Qwen/Qwen2.5-1.5B-Instruct" \
        --response_length 700 \
        --missing_eos_penalty 1.0 \
        --kl_coef 0.0 \
        --stop_token "eos" \
        --reward_model_address "$rm_address"  \
        --meta_reward_model_address "$_mrm_address"
}

run_experiment "$exp_type" "$rubric_type" "$rm_params" "$mrm_params" "$prompt"
