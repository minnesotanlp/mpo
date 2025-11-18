# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import transformers
from packaging import version
from transformers import TrainingArguments


@dataclass
class MPOGRPOConfig(TrainingArguments):
    r"""
    Configuration class for the [`MPOGRPOTrainer`].

    This extends GRPO-style training (value-free, reference-free) with MPO-specific knobs for reward/meta-reward model
    usage and prompt evolution.

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.
    """

    # MPO-specific fields
    task_name: str = field(
        default=None,
        metadata={
            "help": "Name of the task. Should be one of ['essay_writing', 'summarization', 'math_reasoning', 'ethical_reasoning']"
        },
    )
    exp_name: str = field(
        default=os.path.basename(__file__)[:-3],
        metadata={"help": "Name of this experiment."},
    )
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": "Name of wandb entity."},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "Name of wandb project."},
    )
    init_rm_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "Path to initial version of RM evaluation prompt."},
    )
    reward_model_address: Optional[str] = field(
        default=None,
        metadata={"help": "Address to the reward model."},
    )
    meta_reward_model_address: Optional[str] = field(
        default=None,
        metadata={"help": "Address to the meta reward model."},
    )
    # MPO cadence (half of MPOPPO defaults due to multiple responses per prompt in GRPO)
    num_mpo_interval: int = field(
        default=5,
        metadata={"help": "Number of batch steps to run before updating the RM prompt using MPO steps (reduced for GRPO)."},
    )
    num_mpo_samples: int = field(
        default=10,
        metadata={"help": "Number of episodes to consider when conducting MPO steps (reduced for grouped sampling)."},
    )

    # GRPO parameters
    model_init_kwargs: Optional[dict[str, Any]] = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `AutoModelForCausalLM.from_pretrained`, used when the `model` argument is provided as a string."
        },
    )
    disable_dropout: bool = field(
        default=False,
        metadata={"help": "Whether to disable dropout in the model."},
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={
            "help": "Whether to only keep the column `prompt` in the dataset. Keep False if custom reward needs extras."
        },
    )
    max_prompt_length: Optional[int] = field(
        default=512,
        metadata={"help": "Maximum length of the prompt; longer prompts are truncated on the left."},
    )
    num_generations: Optional[int] = field(
        default=8,
        metadata={
            "help": "Number of generations per prompt to sample. Effective batch (num_processes * per_device_batch_size * "
            "gradient_accumulation_steps) must be divisible by this value."
        },
    )
    max_completion_length: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum length of the generated completion."},
    )
    ds3_gather_for_generation: bool = field(
        default=True,
        metadata={
            "help": "For DeepSpeed ZeRO-3: gather policy weights for faster generation (disable to fit larger models)."
        },
    )
    shuffle_dataset: bool = field(
        default=True,
        metadata={"help": "Whether to shuffle the training dataset."},
    )

    # Generation parameters
    temperature: float = field(
        default=0.9,
        metadata={"help": "Sampling temperature."},
    )
    top_p: float = field(
        default=1.0,
        metadata={"help": "Cumulative probability for nucleus sampling."},
    )
    top_k: Optional[int] = field(
        default=50,
        metadata={"help": "Top-k cutoff; None disables top-k."},
    )
    min_p: Optional[float] = field(
        default=None,
        metadata={
            "help": "Minimum token probability scaled by most-likely token prob (0.0–1.0). Typical 0.01–0.2 or None."
        },
    )
    repetition_penalty: float = field(
        default=1.0,
        metadata={"help": "Penalize reuse of tokens from prompt/so-far text."},
    )
    cache_implementation: Optional[str] = field(
        default=None,
        metadata={"help": "Cache implementation for faster generation (when not using vLLM)."},
    )

    # vLLM parameters
    use_vllm: bool = field(
        default=False,
        metadata={"help": "Use vLLM for generation (reserve a GPU for the server)."},
    )
    vllm_server_host: str = field(
        default="0.0.0.0",
        metadata={"help": "Host of vLLM server."},
    )
    vllm_server_port: int = field(
        default=8000,
        metadata={"help": "Port of vLLM server."},
    )
    vllm_server_timeout: float = field(
        default=120.0,
        metadata={"help": "Seconds to wait for vLLM server readiness before failing."},
    )
    vllm_guided_decoding_regex: Optional[str] = field(
        default=None,
        metadata={"help": "Regex for vLLM guided decoding. None disables."},
    )

    # Training parameters (GRPO)
    learning_rate: float = field(
        default=1e-6,
        metadata={
            "help": "Initial learning rate for AdamW (overrides TrainingArguments default)."
        },
    )
    beta: float = field(
        default=0.0,
        metadata={
            "help": "KL coefficient. Set to 0.0 for ref-free MPOGRPO."
        },
    )
    num_iterations: int = field(
        default=1,
        metadata={"help": "Number of iterations per batch (μ in the algorithm)."},
    )
    epsilon: float = field(
        default=0.2,
        metadata={"help": "Epsilon value for clipping."},
    )
    epsilon_high: Optional[float] = field(
        default=None,
        metadata={
            "help": "Upper-bound epsilon value for clipping (DAPO recommends 0.28). Defaults to epsilon if unset."
        },
    )
    reward_weights: Optional[list[float]] = field(
        default=None,
        metadata={"help": "Weights per reward function; if None, all set to 1.0."},
    )
    scale_rewards: bool = field(
        default=True,
        metadata={"help": "Normalize rewards by their standard deviation."},
    )
    use_weighted_mean: bool = field(
        default=True,
        metadata={"help": "Whether to use weighted mean of rewards."},
    )
    mask_truncated_completions: bool = field(
        default=False,
        metadata={
            "help": "If True, completions truncated without EOS are masked out of loss/reward aggregation to reduce noise."
        },
    )
    sync_ref_model: bool = field(
        default=False,
        metadata={"help": "Keep for GRPO compatibility; MPOGRPO is ref-free, so this is ignored."},
    )
    ref_model_mixup_alpha: float = field(
        default=0.6,
        metadata={"help": "GRPO compatibility; ignored (ref-free)."},
    )
    ref_model_sync_steps: int = field(
        default=512,
        metadata={"help": "GRPO compatibility; ignored (ref-free)."},
    )
    log_completions: bool = field(
        default=False,
        metadata={"help": "Whether to log sample completions during training."},
    )
    num_completions_to_print: Optional[int] = field(
        default=None,
        metadata={"help": "Number of completions to print/log when log_completions is enabled."},
    )
    wandb_log_unique_prompts: bool = field(
        default=False,
        metadata={"help": "Whether to deduplicate prompts when logging to wandb."},
    )
    loss_type: Optional[str] = field(
        default="grpo",
        metadata={"help": "Loss type for GRPO variants."},
    )
    use_liger_loss: bool = field(
        default=False,
        metadata={"help": "Use Liger fused GRPO loss if available."},
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Enable HF gradient checkpointing on the model."},
    )

    # VLLM deprecated fields (kept for interface compatibility)
    vllm_device: Optional[str] = field(
        default=None,
        metadata={"help": "Deprecated. vLLM device selection should be done at server launch time."},
    )
    vllm_gpu_memory_utilization: Optional[float] = field(
        default=None,
        metadata={
            "help": "Deprecated. Use `gpu_memory_utilization` in vLLM server configuration instead."
        },
    )
    vllm_dtype: Optional[str] = field(
        default=None,
        metadata={"help": "Deprecated. Control dtype at vLLM server configuration instead."},
    )
    vllm_max_model_len: Optional[int] = field(
        default=None,
        metadata={"help": "Deprecated. Control max_model_len at vLLM server configuration instead."},
    )
    vllm_enable_prefix_caching: Optional[bool] = field(
        default=None,
        metadata={"help": "Deprecated. Control prefix caching at vLLM server configuration instead."},
    )

    # Compatibility fields (parsed but not used by MPOGRPO core)
    sft_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "SFT model path (for parity with PPO-style launchers)."},
    )
    num_ppo_epochs: int = field(
        default=4,
        metadata={"help": "Parsed for launcher compatibility; not used in MPOGRPO loop."},
    )
    save_n_updates: int = field(
        default=20,
        metadata={"help": "Parsed for launcher compatibility; not used in MPOGRPO loop."},
    )
    num_mini_batches: int = field(
        default=1,
        metadata={"help": "Parsed for launcher compatibility; not used in MPOGRPO loop."},
    )
    local_rollout_forward_batch_size: Optional[int] = field(
        default=None,
        metadata={"help": "Parsed for launcher compatibility; not used in MPOGRPO loop."},
    )
    total_episodes: Optional[int] = field(
        default=None,
        metadata={"help": "Parsed for launcher compatibility; not used in MPOGRPO loop."},
    )
    response_length: Optional[int] = field(
        default=350,
        metadata={"help": "Parsed for launcher compatibility; not used in MPOGRPO loop."},
    )
    missing_eos_penalty: Optional[float] = field(
        default=None,
        metadata={"help": "Parsed for launcher compatibility; not used in MPOGRPO loop."},
    )
    stop_token: Optional[str] = field(
        default=None,
        metadata={"help": "Parsed for launcher compatibility; not used in MPOGRPO loop."},
    )
    stop_token_id: Optional[int] = field(
        default=None,
        metadata={"help": "Parsed for launcher compatibility; not used in MPOGRPO loop."},
    )
    kl_coef: float = field(
        default=0.0,
        metadata={"help": "Parsed for launcher compatibility; MPOGRPO is ref-free so typically 0.0."},
    )

    def __post_init__(self):
        super().__post_init__()

        if self.epsilon_high is None:
            self.epsilon_high = self.epsilon

        # Deprecation warnings from GRPO config
        if self.vllm_device is not None:
            warnings.warn(
                "`vllm_device` is deprecated and will be removed in version 0.18.0. To use vLLM, start a vLLM server "
                "with the `trl vllm-serve` command.",
                DeprecationWarning,
            )

        if self.vllm_gpu_memory_utilization is not None:
            warnings.warn(
                "`vllm_gpu_memory_utilization` is deprecated and will be removed in v0.18. To control the GPU memory "
                "utilization for vLLM, you should now use the `gpu_memory_utilization` parameter in the vLLM server "
                "configuration.",
                DeprecationWarning,
            )

        if self.vllm_dtype is not None:
            warnings.warn(
                "`vllm_dtype` is deprecated and will be removed in version 0.18.0. To control the data type for vLLM "
                "generation, you should now use the `dtype` parameter in the vLLM server configuration.",
                DeprecationWarning,
            )

        if self.vllm_max_model_len is not None:
            warnings.warn(
                "`vllm_max_model_len` is deprecated and will be removed in version 0.18.0. To control the "
                "`max_model_len` for vLLM, you should now use the `max_model_len` parameter in the vLLM server "
                "configuration.",
                DeprecationWarning,
            )

        if self.vllm_enable_prefix_caching is not None:
            warnings.warn(
                "`vllm_enable_prefix_caching` is deprecated and will be removed in version 0.18.0. To control prefix "
                "caching in vLLM, you should now use the `enable_prefix_caching` parameter in the vLLM server "
                "configuration.",
                DeprecationWarning,
            )

        # Transformers compatibility check (copied from GRPO config)
        if version.parse(transformers.__version__) < version.parse("4.37.0"):
            raise ValueError("GRPO requires transformers>=4.37.0")
