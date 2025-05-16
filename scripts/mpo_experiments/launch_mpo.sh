#!/usr/bin/env bash
set -euo pipefail

###############################################################################
#  CLI arguments
#     $1 – path to .pem key
#     $2 – reward-model (RM) endpoint URL
#     $3 – meta-reward-model (MRM) endpoint URL
#     $4 – [optional] ssh user@host for launch_rm_mrm.bash
###############################################################################
if (( $# < 3 )); then
  echo "Usage: $0 <pem_key> <rm_address> <mrm_address> [remote_host]"
  exit 1
fi

pem_key=$1
rm_address=$2
mrm_address=$3
remote_host=$4   # default unchanged

###############################################################################
#  Paths & constants
###############################################################################
trl_dir="$HOME/Development/trl"
SCRIPT="$trl_dir/examples/scripts/mpo.py"

WANDB_ENTITY="iterater"
WANDB_PROJECT="mpo-new"
DATASET="math_reasoning"
TASK="math_reasoning"
PROMPT_DIR="$trl_dir/trl/extras/mpo/prompts/math_reasoning"

###############################################################################
#  Main runner
###############################################################################
run_experiment() {
    local exp_type=$1         # mpo / ppo …
    local rubric_type=$2      # e.g. iter0
    local rm=$3               # reward-model size          (e.g. 1.5b)
    local mrm=$4              # meta-reward-model size     (e.g. 3b)
    local prompt_name=$5      # prompt file

    # ------------------------------------------------------------------------
    #  GPU layout
    # ------------------------------------------------------------------------
    local CUDA_DEVICES="0,1,2,3"
    local ACC_CONFIG="$trl_dir/examples/accelerate_configs/deepspeed_zero2_4gpus.yaml"

    # ------------------------------------------------------------------------
    #  Naming & bookkeeping
    # ------------------------------------------------------------------------
    local policy_model="policy-1.5b"
    local model_name
    if [[ "$exp_type" == "mpo" ]]; then
        model_name="${rubric_type}-${rm}_${mrm}"
    else
        model_name="${rubric_type}-${rm}"
    fi

    local exp_name="${policy_model}-math-${exp_type}-${model_name}"
    local output_dir="$trl_dir/models/${policy_model}/${TASK}/${exp_type}/${model_name}"
    if [ -d $output_dir ]; then
        printf "$output_dir already exists. Skipped.\n"
        return
    fi

    # gradient accumulation scaling
    local grad_acc_steps=8

    # MPO interval
    local num_mpo_interval=99999999
    [[ "$exp_type" == "mpo" ]] && num_mpo_interval=20

    local _mrm_address=$mrm_address
    [[ $rm == $mrm ]] && _mrm_address=$rm_address

    # ------------------------------------------------------------------------
    #  Display run-time configuration
    # ------------------------------------------------------------------------
    printf -- "==============  Experiment %s  ==============\n" "$exp_name"
    printf "CUDA_DEVICES       : %s\n" "$CUDA_DEVICES"
    printf "ACC_CONFIG         : %s\n" "$ACC_CONFIG"
    printf "grad_acc_steps     : %s\n" "$grad_acc_steps"
    printf "num_mpo_interval   : %s\n" "$num_mpo_interval"
    printf "rm_address         : %s\n" "$rm_address"
    printf "mrm_address        : %s\n" "$_mrm_address"
    printf -- "==============================================\n\n"

    # ------------------------------------------------------------------------
    #  Spin-up RM & MRM servers and wait for them to be ready
    # ------------------------------------------------------------------------
    # bash "$trl_dir/scripts/mpo_experiments/launch_rm_mrm.sh" \
    #     "$pem_key" "$remote_host" \
    #     "${rm^^}" "${mrm^^}"

    # sleep 5

    # ------------------------------------------------------------------------
    # Launch policy fine-tuning
    # ------------------------------------------------------------------------
    WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
    accelerate launch --config_file "$ACC_CONFIG" \
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
        --save_n_updates 20 \
        --num_mpo_samples 16 \
        --num_mini_batches 1 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps "$grad_acc_steps" \
        --local_rollout_forward_batch_size 48 \
        --total_episodes 7500 \
        --model_name_or_path "Qwen/Qwen2.5-1.5B-Instruct" \
        --sft_model_path   "Qwen/Qwen2.5-1.5B-Instruct" \
        --response_length 1000 \
        --missing_eos_penalty 1.0 \
        --kl_coef 0.02 \
        --stop_token "eos" \
        --reward_model_address "$rm_address"  \
        --meta_reward_model_address "$_mrm_address"
    sleep 10
}

###############################################################################
#  Sweep
###############################################################################
exp_type="mpo"
rubric_type="iter0"
prompt="evaluation_rubric_real_iter_0.txt"
# rubric_type="autoprompt"
# prompt="evaluation_rubric_autoprompt.txt"
# declare -a rms=("1.5b" "7b" "14b" "3b")
# declare -a mrms=("7b" "14b" "1.5b" "3b")
declare -a rms=("32b")
declare -a mrms=("32b")
for rm in "${rms[@]}"; do
    for mrm in "${mrms[@]}"; do
        # run_experiment "$exp_type" "$rubric_type" "$rm" "$mrm" "$prompt"
        run_experiment "$exp_type" "$rubric_type" "$rm" "$rm" "$prompt"
    done
done