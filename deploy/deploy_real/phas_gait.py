"""Base classes for command manager terms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from torch import Tensor
import torch


class CommandTermBase(ABC):
    """Base class for stateful command terms.

    Command terms can keep internal state and execute logic during the
    environment lifecycle. Subclasses should implement all lifecycle hooks.
    """

    def __init__(self):
        pass

    @abstractmethod
    def setup(self) -> None:
        """Setup hook called once during environment initialization."""

    @abstractmethod
    def reset(self, env_ids: Tensor | None) -> None:
        """Reset hook called whenever environments reset.

        Args:
            env_ids: Tensor of environment ids to reset, or None to reset all.
        """

    @abstractmethod
    def step(self) -> None:
        """Per-step hook called during simulation rollout."""


class LocomotionGait(CommandTermBase):
    """Stateful term that owns gait phase buffers and updates them each step."""

    gait_period: float
    gait_period_randomization_width: float

    def __init__(self, dt):
        self.dt = dt
        self.gait_period: float = 1.0
        self.gait_period_randomization_width: 0.2
        self.randomize_phase: bool = True
        self.stand_phase_value: float = torch.pi

        self.phase_offset: torch.Tensor | None = None
        self.phase: torch.Tensor | None = None
        self.gait_freq: torch.Tensor | None = None
        self.phase_dt: torch.Tensor | None = None
        self.mean_gait_freq: float = 1.0 / self.gait_period

    def setup(self) -> None:
        device = "cpu"
        num_envs = 1

        self.phase_offset = torch.zeros((num_envs, 2), dtype=torch.float32, device=device)
        self.phase = torch.zeros((num_envs, 2), dtype=torch.float32, device=device)
        self.gait_freq = torch.zeros((num_envs, 1), dtype=torch.float32, device=device)
        self.phase_dt = torch.zeros((num_envs, 1), dtype=torch.float32, device=device)

        self._initialize_indices(None, evaluating=True)

    def reset(self, env_ids: torch.Tensor | None) -> None:
        self._initialize_indices(env_ids, evaluating=True)

    def step(self, cmd, episode_length_buf) -> None:
        if self.phase is None or self.phase_offset is None or self.phase_dt is None:
            return

        phase_tp1 = episode_length_buf.unsqueeze(1) * self.phase_dt + self.phase_offset
        self.phase.copy_(torch.fmod(phase_tp1 + torch.pi, 2 * torch.pi) - torch.pi)

        command_tensor = torch.tensor(cmd, dtype=torch.float32)[None, :]

        stand_mask = torch.logical_and(
            torch.linalg.norm(command_tensor[:, :2], dim=1) < 0.01,
            torch.abs(command_tensor[:, 2]) < 0.01,
        )
        if stand_mask.any():
            self.phase[stand_mask] = torch.full(
                (int(stand_mask.sum().item()), 2), self.stand_phase_value, device="cpu"
            )

    def set_eval_mode(self, evaluating: bool) -> None:
        self._initialize_indices(None, evaluating=evaluating)

    def resample_frequency(self, env_ids: torch.Tensor) -> None:
        if self.gait_freq is None or self.phase_dt is None:
            return

        idx = self._ensure_index_tensor(env_ids)
        if idx.numel() == 0:
            return

        if True or self.gait_period_randomization_width <= 0.0:
            self.gait_freq[idx] = self.mean_gait_freq
        else:
            low = self.mean_gait_freq - self.gait_period_randomization_width
            high = self.mean_gait_freq + self.gait_period_randomization_width
            self.gait_freq[idx] = torch_rand_float(low, high, (idx.shape[0], 1), device="cpu")

        self.phase_dt[idx] = 2 * torch.pi * self.dt * self.gait_freq[idx]

    # ------------------------------------------------------------------ #
    # Internal utilities
    # ------------------------------------------------------------------ #

    def _initialize_indices(self, env_ids: torch.Tensor | None, *, evaluating: bool) -> None:
        if self.phase_offset is None or self.phase is None or self.gait_freq is None or self.phase_dt is None:
            return

        idx = self._ensure_index_tensor(env_ids)
        if idx.numel() == 0:
            return

        if evaluating:
            self.phase_offset[idx, 0] = 0.0
            self.phase_offset[idx, 1] = -torch.pi
        elif self.randomize_phase:
            self.phase_offset[idx, 0] = torch_rand_float(
                -torch.pi, torch.pi, (idx.shape[0], 1), device="cpu"
            ).squeeze(1)
            self.phase_offset[idx, 1] = torch.fmod(self.phase_offset[idx, 0] + 2 * torch.pi, 2 * torch.pi) - torch.pi
        else:
            self.phase_offset[idx] = 0.0

        self.phase[idx] = self.phase_offset[idx]
        self.resample_frequency(idx)

    def _ensure_index_tensor(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(1, device="cpu", dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device="cpu", dtype=torch.long)
        return torch.as_tensor(env_ids, device="cpu", dtype=torch.long)


def torch_rand_float(lower: float, upper: float, shape: tuple[int, int], device: str) -> torch.Tensor:
    """Generate random float tensor.

    Parameters
    ----------
    lower : float
        Lower bound
    upper : float
        Upper bound
    shape : tuple[int, int] | torch.Size
        Shape of output tensor. Can be a tuple or torch.Size object.
    device : str
        Device to place tensor on

    Returns
    -------
    torch.Tensor
        Random tensor of specified shape
    """
    return (upper - lower) * torch.rand(*shape, device=device) + lower
