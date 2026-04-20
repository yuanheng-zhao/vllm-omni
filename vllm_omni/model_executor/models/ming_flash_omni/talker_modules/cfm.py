# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
# Copyright (c) Ant Group. All rights reserved.
# Ported from:
# https://github.com/inclusionAI/Ming/blob/e58533db227031990c5a6864dcf5f08fb53ed0d2/talker_module/cfm.py

from __future__ import annotations

from queue import Queue
from threading import Lock

import torch
from torch import nn


def get_epss_timesteps(n, device, dtype):
    dt = 1 / 32
    predefined_timesteps = {
        5: [0, 2, 4, 8, 16, 32],
        6: [0, 2, 4, 6, 8, 16, 32],
        7: [0, 2, 4, 6, 8, 16, 24, 32],
        10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32],
        12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
        16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32],
    }
    t = predefined_timesteps.get(n, [])
    if not t:
        return torch.linspace(0, 1, n + 1, device=device, dtype=dtype)
    return dt * torch.tensor(t, device=device, dtype=dtype)


class CFM(nn.Module):
    """Conditional Flow Matching module for audio latent generation."""

    def __init__(self, model: nn.Module, steps: int = 10, sway_sampling_coef: float | None = -1.0):
        """
        Args:
            model: DiT used for the velocity prediction.
            steps: number of integration steps per sample call.
            sway_sampling_coef: coefficient used to skew the integration
                grid towards low-noise timesteps. Defaults to -1.0 which
                packs more steps near t=0, where prediction error is highest.
                Set to `None` to use the linear grid as-is.
        """
        super().__init__()
        self.model = model
        self.steps = steps
        self.sway_sampling_coef = sway_sampling_coef

    @torch.no_grad()
    def sample(
        self,
        llm_cond: torch.Tensor,
        lat_cond: torch.Tensor,
        y0: torch.Tensor,
        t: torch.Tensor,
        sde_args: torch.Tensor,
        sde_rnd: torch.Tensor,
    ):
        """Sample audio latent via ODE/SDE integration with CFG.

        Args:
            llm_cond: LLM hidden state (B, 1, hidden_size)
            lat_cond: latent history (B, his_patch_size, latent_dim)
            y0: initial noise (B, patch_size, latent_dim)
            t: timesteps from get_epss_timesteps
            sde_args: [cfg_strength, sigma, temperature]
            sde_rnd: random noise for SDE steps (steps, B, patch_size, latent_dim)
        """

        def fn(fn_t, x):
            pred_cfg = self.model.forward_with_cfg(x, fn_t, llm_cond, lat_cond, None)
            pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
            return pred + (pred - null_pred) * sde_args[0]

        if self.sway_sampling_coef is not None:
            t = t + self.sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        for step in range(self.steps):
            dt = t[step + 1] - t[step]
            y0 = y0 + fn(t[step], y0) * dt
            y0 = y0 + sde_args[1] * (sde_args[2] ** 0.5) * (dt.abs() ** 0.5) * sde_rnd[step]

        return y0


class CFMGraphExecutor:
    """CUDA graph-accelerated executor for CFM + Aggregator + StopHead pipeline."""

    def __init__(self, config, cfm, aggregator, stop_head: nn.Linear):
        self.config = config
        self.cfm = cfm
        self.aggregator = aggregator
        self.stop_head = stop_head
        self.initialized = False

        self.last_hidden_state_placeholder = None
        self.his_lat_placeholder = None
        self.randn_like_placeholder = None
        self.t_placeholder = None
        self.sde_args_placeholder = None
        self.sde_rnd_placeholder = None
        self.gen_lat_placeholder = None
        self.inputs_embeds_placeholder = None
        self.stop_out_placeholder = None
        self.graph = None

    def execute(
        self,
        input_tensor: torch.Tensor,
        his_lat: torch.Tensor,
        cfg_strength: float = 2.0,
        sigma: float = 0.25,
        temperature: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bat_size, his_patch_size, z_dim = his_lat.shape
        randn_tensor = torch.randn(
            (bat_size, self.config.patch_size, z_dim), device=input_tensor.device, dtype=input_tensor.dtype
        )
        t = get_epss_timesteps(self.config.steps, device=input_tensor.device, dtype=input_tensor.dtype)
        sde_rnd = torch.randn(
            (self.config.steps, *randn_tensor.shape), device=input_tensor.device, dtype=input_tensor.dtype
        )

        if not self.initialized:
            self._initialize_graph(input_tensor, his_lat, randn_tensor, sde_rnd)

        self.last_hidden_state_placeholder.copy_(input_tensor)
        self.his_lat_placeholder.copy_(his_lat)
        self.randn_like_placeholder.copy_(randn_tensor)
        self.t_placeholder.copy_(t)
        self.sde_args_placeholder[0] = cfg_strength
        self.sde_args_placeholder[1] = sigma
        self.sde_args_placeholder[2] = temperature
        self.sde_rnd_placeholder.copy_(sde_rnd)

        self.graph.replay()

        gen_lat = torch.empty_like(self.gen_lat_placeholder)
        gen_lat.copy_(self.gen_lat_placeholder)

        inputs_embeds = torch.empty_like(self.inputs_embeds_placeholder)
        inputs_embeds.copy_(self.inputs_embeds_placeholder)

        stop_out = torch.empty_like(self.stop_out_placeholder)
        stop_out.copy_(self.stop_out_placeholder)

        return gen_lat, inputs_embeds, stop_out

    def _initialize_graph(
        self,
        input_tensor: torch.Tensor,
        his_lat: torch.Tensor,
        randn_tensor: torch.Tensor,
        sde_rnd: torch.Tensor,
    ) -> None:
        self.last_hidden_state_placeholder = torch.empty_like(input_tensor)
        self.his_lat_placeholder = torch.empty_like(his_lat)
        self.randn_like_placeholder = torch.empty_like(randn_tensor)
        self.t_placeholder = get_epss_timesteps(self.config.steps, device=input_tensor.device, dtype=input_tensor.dtype)
        self.sde_args_placeholder = torch.empty(3, device=input_tensor.device, dtype=input_tensor.dtype)
        self.sde_rnd_placeholder = torch.empty_like(sde_rnd)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.gen_lat_placeholder = self.cfm.sample(
                self.last_hidden_state_placeholder,
                self.his_lat_placeholder,
                self.randn_like_placeholder,
                self.t_placeholder,
                self.sde_args_placeholder,
                self.sde_rnd_placeholder,
            )
            self.inputs_embeds_placeholder = self.aggregator(self.gen_lat_placeholder)
            self.stop_out_placeholder = self.stop_head(self.last_hidden_state_placeholder[:, -1, :]).softmax(dim=-1)

        self.initialized = True


class CFMGraphExecutorPool:
    """Thread-safe pool of CFMGraphExecutors for concurrent inference."""

    def __init__(self, config, cfm, aggregator, stop_head: nn.Linear, pool_size: int = 1):
        self.config = config
        self.cfm = cfm
        self.aggregator = aggregator
        self.stop_head = stop_head
        self.pool_size = pool_size
        self.pool = Queue(maxsize=pool_size)
        self.lock = Lock()

        for _ in range(pool_size):
            executor = CFMGraphExecutor(config, cfm, aggregator, stop_head)
            self.pool.put(executor)

    def acquire(self) -> CFMGraphExecutor:
        return self.pool.get()

    def release(self, executor: CFMGraphExecutor) -> None:
        self.pool.put(executor)

    def execute(
        self,
        input_tensor: torch.Tensor,
        his_lat: torch.Tensor,
        cfg_strength: float = 2.0,
        sigma: float = 0.25,
        temperature: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        executor = self.acquire()
        try:
            return executor.execute(
                input_tensor, his_lat, cfg_strength=cfg_strength, sigma=sigma, temperature=temperature
            )
        finally:
            self.release(executor)
