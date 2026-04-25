# Copyright 2026 Zhejiang University Team. All rights reserved.
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

# DISCLAIMER: This code is strongly influenced by https://huggingface.co/docs/diffusers/api/schedulers/lcm

from diffusers import LCMScheduler
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from config import config


from diffusers.schedulers.scheduling_lcm import BaseOutput, logging, randn_tensor
from diffusers.schedulers.scheduling_lcm import register_to_config

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class BBDMSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.FloatTensor
    predicted_original_sample: torch.FloatTensor


class BBDMScheduler(LCMScheduler):
    order = 1

    @register_to_config
    def __init__(self, num_train_timesteps: int = 1000, original_inference_steps: int = 1000):

        # setable values
        super().__init__(num_train_timesteps, original_inference_steps)
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy().astype(np.int64))
        self.custom_timesteps = False

        self._step_index = None
        self._begin_index = None

    def m(self, timestep) -> torch.FloatTensor:
        m_t = timestep / self.num_train_timesteps
        return m_t

    def sigma(self, timestep) -> torch.FloatTensor:
        sigma_t = 2 * (self.m(timestep) - self.m(timestep) ** 2) * config.s
        return sigma_t

    def sqrt_sigma(self, timestep) -> torch.FloatTensor:
        sqrt_sigma_t = torch.sqrt(self.sigma(timestep))
        return sqrt_sigma_t

    def sigma_previous(self, timestep, prev_timestep) -> torch.FloatTensor:
        sigma_previous_t = self.sigma(timestep) - self.sigma(prev_timestep) * (1 - self.m(timestep)) ** 2 / (
                1 - self.m(prev_timestep)) ** 2
        return sigma_previous_t

    def tilde_sigma(self, timestep, prev_timestep) -> torch.FloatTensor:
        tilde_sigma_t = self.sigma_previous(timestep, prev_timestep) * self.sigma(prev_timestep) / self.sigma(timestep)
        return tilde_sigma_t

    def sqrt_tilde_sigma(self, timestep, prev_timestep) -> torch.FloatTensor:
        sqrt_tilde_sigma_t = torch.sqrt(self.tilde_sigma(timestep, prev_timestep))
        return sqrt_tilde_sigma_t

    def c_x(self, timestep, prev_timestep) -> torch.FloatTensor:
        c_xt = self.sigma(prev_timestep) / self.sigma(timestep) * (1 - self.m(timestep)) / (
                1 - self.m(prev_timestep)) + self.sigma_previous(timestep, prev_timestep) / self.sigma(timestep) * (
                       1 - self.m(prev_timestep))
        return c_xt

    def c_y(self, timestep, prev_timestep) -> torch.FloatTensor:
        c_yt = self.m(prev_timestep) - self.m(timestep) * (1 - self.m(timestep)) / (
                1 - self.m(prev_timestep)) * self.sigma(prev_timestep) / self.sigma(timestep)
        return c_yt

    def c_epsilon(self, timestep, prev_timestep) -> torch.FloatTensor:
        c_epsilon_t = (1 - self.m(prev_timestep)) * self.sigma_previous(timestep, prev_timestep) / self.sigma(
            timestep) * self.sqrt_sigma(timestep) / (1 - self.m(timestep))
        return c_epsilon_t

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    @property
    def step_index(self):
        return self._step_index

    @property
    def begin_index(self):
        """
        The index for the first timestep. It should be set from pipeline with `set_begin_index` method.
        """
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0):
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.

        Args:
            begin_index (`int`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def set_timesteps(
            self,
            num_inference_steps: Optional[int] = None,
            device: Union[str, torch.device] = None,
            original_inference_steps: Optional[int] = 1000,
            timesteps: Optional[List[int]] = None,
            strength: int = 1.0,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`, *optional*):
                The number of diffusion steps used when generating samples with a pre-trained model. If used,
                `timesteps` must be `None`.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            original_inference_steps (`int`, *optional*):
                The original number of inference steps, which will be used to generate a linearly-spaced timestep
                schedule (which is different from the standard `diffusers` implementation). We will then take
                `num_inference_steps` timesteps from this schedule, evenly spaced in terms of indices, and use that as
                our final timestep schedule. If not set, this will default to the `original_inference_steps` attribute.
            timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of equal spacing between timesteps on the training/distillation timestep
                schedule is used. If `timesteps` is passed, `num_inference_steps` must be `None`.
        """
        # 0. Check inputs
        if num_inference_steps is None and timesteps is None:
            raise ValueError("Must pass exactly one of `num_inference_steps` or `custom_timesteps`.")

        if num_inference_steps is not None and timesteps is not None:
            raise ValueError("Can only pass one of `num_inference_steps` or `custom_timesteps`.")

        # 1. Calculate the BBDM original training/distillation timestep schedule.
        original_steps = (
            original_inference_steps if original_inference_steps is not None else self.config.original_inference_steps
        )

        if original_steps > self.config.num_train_timesteps:
            raise ValueError(
                f"`original_steps`: {original_steps} cannot be larger than `self.config.train_timesteps`:"
                f" {self.config.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                f" maximal {self.config.num_train_timesteps} timesteps."
            )

        # BBDM Timesteps Setting
        # The skipping step parameter k from the paper.
        k = self.config.num_train_timesteps // original_steps
        # BBDM Training/Distillation Steps Schedule
        # Currently, only a linearly-spaced schedule is supported (same as in the BBDM distillation scripts).
        bbdm_origin_timesteps = np.asarray(list(range(1, int(original_steps * strength) + 1))) * k - 1

        # 2. Calculate the BBDM inference timestep schedule.
        if timesteps is not None:
            # 2.1 Handle custom timestep schedules.
            train_timesteps = set(bbdm_origin_timesteps)
            non_train_timesteps = []
            for i in range(1, len(timesteps)):
                if timesteps[i] >= timesteps[i - 1]:
                    raise ValueError("`custom_timesteps` must be in descending order.")

                if timesteps[i] not in train_timesteps:
                    non_train_timesteps.append(timesteps[i])

            if timesteps[0] >= self.config.num_train_timesteps:
                raise ValueError(
                    f"`timesteps` must start before `self.config.train_timesteps`:"
                    f" {self.config.num_train_timesteps}."
                )

            # Raise warning if timestep schedule does not start with self.config.num_train_timesteps - 1
            if strength == 1.0 and timesteps[0] != self.config.num_train_timesteps - 1:
                logger.warning(
                    f"The first timestep on the custom timestep schedule is {timesteps[0]}, not"
                    f" `self.config.num_train_timesteps - 1`: {self.config.num_train_timesteps - 1}. You may get"
                    f" unexpected results when using this timestep schedule."
                )

            # Raise warning if custom timestep schedule contains timesteps not on original timestep schedule
            if non_train_timesteps:
                logger.warning(
                    f"The custom timestep schedule contains the following timesteps which are not on the original"
                    f" training/distillation timestep schedule: {non_train_timesteps}. You may get unexpected results"
                    f" when using this timestep schedule."
                )

            # Raise warning if custom timestep schedule is longer than original_steps
            if len(timesteps) > original_steps:
                logger.warning(
                    f"The number of timesteps in the custom timestep schedule is {len(timesteps)}, which exceeds the"
                    f" the length of the timestep schedule used for training: {original_steps}. You may get some"
                    f" unexpected results when using this timestep schedule."
                )

            timesteps = np.array(timesteps, dtype=np.int64)
            self.num_inference_steps = len(timesteps)
            self.custom_timesteps = True

            init_timestep = min(int(self.num_inference_steps * strength), self.num_inference_steps)
            t_start = max(self.num_inference_steps - init_timestep, 0)
            timesteps = timesteps[t_start * self.order:]
        else:
            if num_inference_steps > self.config.num_train_timesteps:
                raise ValueError(
                    f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`:"
                    f" {self.config.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                    f" maximal {self.config.num_train_timesteps} timesteps."
                )

            self.num_inference_steps = num_inference_steps

            if num_inference_steps > original_steps:
                raise ValueError(
                    f"`num_inference_steps`: {num_inference_steps} cannot be larger than `original_inference_steps`:"
                    f" {original_steps} because the final timestep schedule will be a subset of the"
                    f" `original_inference_steps`-sized initial timestep schedule."
                )

            # BBDM Inference Steps Schedule
            bbdm_origin_timesteps = bbdm_origin_timesteps[::-1].copy()

            # Select (approximately) evenly spaced indices from bbdm_origin_timesteps.
            inference_indices = np.linspace(0, len(bbdm_origin_timesteps) - 1, num=num_inference_steps, endpoint=True)
            inference_indices = np.floor(inference_indices).astype(np.int64)
            timesteps = bbdm_origin_timesteps[inference_indices]

        self.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=torch.long)

        self._step_index = None
        self._begin_index = None


    def step(
            self,
            model_output: torch.FloatTensor,
            timestep: int,
            sample: torch.FloatTensor,
            y: torch.FloatTensor,
            generator: Optional[torch.Generator] = None,
            return_dict: bool = True,
    ) -> Union[BBDMSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`int`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            y (`torch.FloatTensor`):
                y.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_bbdm.BBDMSchedulerOutput`] or `tuple`.
        Returns:
            [`~schedulers.scheduling_utils.BBDMSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_bbdm.BBDMSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.
        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        prev_step_index = self.step_index + 1
        if prev_step_index < len(self.timesteps):
            prev_timestep = self.timesteps[prev_step_index]
        else:
            prev_timestep = timestep

        if config.noise_correction:
            model_output = model_output - torch.mean(model_output)
            model_output = model_output / torch.std(model_output)

        m_t = self.m(timestep).view(-1, 1, 1, 1)

        if config.prediction_type == "noise":

            denom = 1 - m_t
            denom = torch.clamp(denom, min=1e-5)

            # 计算 x0
            predicted_original_sample = (sample - m_t * y - self.sqrt_sigma(timestep).view(-1, 1, 1,
                                                                                           1) * model_output) / denom
        else:
            predicted_original_sample = sample - model_output

        m_prev = self.m(prev_timestep).view(-1, 1, 1, 1)
        alpha_prev = 1 - m_prev

        prev_sample_mean = alpha_prev * predicted_original_sample + m_prev * y


        if self.step_index != self.num_inference_steps - 1:
            noise = randn_tensor(
                model_output.shape, generator=generator, device=model_output.device, dtype=sample.dtype
            )
            variance = self.sqrt_tilde_sigma(timestep, prev_timestep).view(-1, 1, 1, 1)
            prev_sample = prev_sample_mean + variance * noise
        else:
            prev_sample = prev_sample_mean

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample, predicted_original_sample)

        return BBDMSchedulerOutput(prev_sample=prev_sample, predicted_original_sample=predicted_original_sample)

    def add_noise(
            self,
            original_samples: torch.FloatTensor,
            ys: torch.FloatTensor,
            noise: torch.FloatTensor,
            timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        noisy_samples = (1 - self.m(timesteps)).view(-1, 1, 1, 1) * original_samples + self.m(timesteps).view(-1, 1, 1,
                                                                                                              1) * ys + self.sqrt_sigma(
            timesteps).view(-1, 1, 1, 1) * noise
        return noisy_samples

    def __len__(self):
        return self.config.num_train_timesteps



if __name__ == "__main__":
    pass
