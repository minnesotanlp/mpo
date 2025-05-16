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
from dataclasses import dataclass, field
from typing import Literal, Optional

from ..trainer.utils import OnPolicyConfig


@dataclass
class MPOConfig(OnPolicyConfig):
    r"""
    Configuration class for the [`MPOTrainer`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        task_name (`str`, *optional*, defaults to `None`):
            Name of the task. Should be one of ['essay_writing', 'summarization', 'math_reasoning', 'ethical_reasoning']
        exp_name (`str`, *optional*, defaults to `os.path.basename(__file__)[:-3]`):
            Name of this experiment.
        wandb_entity (`str`, *optional*, defaults to `None`):
            Name of wandb entity.
        wandb_project (`str`, *optional*, defaults to `None`):
            Name of wandb project.
        init_rm_prompt (`str`, *optional*, defaults to `None`):
            Path to initial version of RM evaluation prompt.
        reward_model_address (`str`, *optional*, defaults to `None`):
            Address to the reward model.
        meta_reward_model_address (`str`, *optional*, defaults to `None`):
            Address to the meta reward model.
        model_adapter_name (`str` or `None`, *optional*, defaults to `None`):
            Name of the train target PEFT adapter, when using LoRA with multiple adapters.
        ref_adapter_name (`str` or `None`, *optional*, defaults to `None`):
            Name of the reference PEFT adapter, when using LoRA with multiple adapters.
        num_ppo_epochs (`int`, *optional*, defaults to `4`):
            Number of epochs to train.
        num_mpo_interval (`int`, *optional*, defaults to `10`):
            Number of batch steps to run before updating the RM prompt using MPO steps.
        num_mpo_samples (`int`, *optional*, defaults to `20`):
            Number of episodes to consider when conducting MPO steps.
        save_n_updates (`int`, *optional*, defaults to `20`):
            Number of updates to save a checkpoint.
        whiten_rewards (`bool`, *optional*, defaults to `False`):
            Whether to whiten the rewards.
        kl_coef (`float`, *optional*, defaults to `0.05`):
            KL coefficient.
        kl_estimator (`Literal["k1", "k3"]`, *optional*, defaults to `"k1"`):
            Which estimator for KL-Divergence to use from [Approximating KL Divergence](http://joschu.net/blog/kl-approx.html).
            Defaults to "k1", a straightforward, unbiased estimator. Can be set to "k3", an unbiased estimator with
            lower variance which "appears to be a strictly better estimator". Cannot be set to "k2", as it is used for
            logging purposes.
        cliprange (`float`, *optional*, defaults to `0.2`):
            Clip range.
        vf_coef (`float`, *optional*, defaults to `0.1`):
            Value function coefficient.
        cliprange_value (`float`, *optional*, defaults to `0.2`):
            Clip range for the value function.
        gamma (`float`, *optional*, defaults to `1.0`):
            Discount factor.
        lam (`float`, *optional*, defaults to `0.95`):
            Lambda value for GAE.
        ds3_gather_for_generation (`bool`, *optional*, defaults to `True`):
            This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for generation,
            improving generation speed. However, disabling this option allows training models that exceed the VRAM
            capacity of a single GPU, albeit at the cost of slower generation.
    """

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
    wandb_entity: str = field(
        default=None,
        metadata={"help": "Name of wandb entity."},
    )
    wandb_project: str = field(
        default=None,
        metadata={"help": "Name of wandb project."},
    )
    init_rm_prompt: str = field(
        default=None,
        metadata={"help": "Path to initial version of RM evaluation prompt."},
    )
    reward_model_address: str = field(
        default=None,
        metadata={"help": "Address to the reward model."},
    )
    meta_reward_model_address: str = field(
        default=None,
        metadata={"help": "Address to the meta reward model."},
    )
    model_adapter_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the train target PEFT adapter, when using LoRA with multiple adapters."},
    )
    ref_adapter_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the reference PEFT adapter, when using LoRA with multiple adapters."},
    )
    num_ppo_epochs: int = field(
        default=4,
        metadata={"help": "Number of epochs to train."},
    )
    num_mpo_interval: int = field(
        default=10,
        metadata={"help": "Number of batch steps to run before updating the RM prompt using MPO steps."},
    )
    num_mpo_samples: int = field(
        default=20,
        metadata={"help": "Number of episodes to consider when conducting MPO steps."},
    )
    save_n_updates: int = field(
        default=20,
        metadata={"help": "Number of updates to save a checkpoint."},
    )
    whiten_rewards: bool = field(
        default=False,
        metadata={"help": "Whether to whiten the rewards."},
    )
    kl_coef: float = field(
        default=0.05,
        metadata={"help": "KL coefficient."},
    )
    kl_estimator: Literal["k1", "k3"] = field(
        default="k1",
        metadata={
            "help": "Which estimator for KL-Divergence to use from Approximating KL Divergence "
            "(http://joschu.net/blog/kl-approx.html). Defaults to 'k1', a straightforward, unbiased estimator. Can be "
            "set to 'k3', an unbiased estimator with lower variance which 'appears to be a strictly better "
            "estimator'. Cannot be set to 'k2', as it is used for logging purposes."
        },
    )
    cliprange: float = field(
        default=0.2,
        metadata={"help": "Clip range."},
    )
    vf_coef: float = field(
        default=0.1,
        metadata={"help": "Value function coefficient."},
    )
    cliprange_value: float = field(
        default=0.2,
        metadata={"help": "Clip range for the value function."},
    )
    gamma: float = field(
        default=1.0,
        metadata={"help": "Discount factor."},
    )
    lam: float = field(
        default=0.95,
        metadata={"help": "Lambda value for GAE."},
    )
    ds3_gather_for_generation: bool = field(
        default=True,
        metadata={
            "help": "This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for "
            "generation, improving generation speed. However, disabling this option allows training models that "
            "exceed the VRAM capacity of a single GPU, albeit at the cost of slower generation."
        },
    )
