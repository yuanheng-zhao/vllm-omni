# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
# Adapted from Ming repository modeling_bailing_talker.py
# https://github.com/inclusionAI/Ming

from __future__ import annotations

from queue import Queue
from threading import Lock

import torch

from .cfm import get_epss_timesteps


class CFMGraphExecutor:
    """CUDA graph-accelerated executor for CFM + Aggregator + StopHead pipeline."""

    def __init__(self, config, cfm, aggregator, stop_head):
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

    def execute(self, input_tensor, his_lat, cfg_strength=2.0, sigma=0.25, temperature=0.0):
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

    def _initialize_graph(self, input_tensor, his_lat, randn_tensor, sde_rnd):
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

    def __init__(self, config, cfm, aggregator, stop_head, pool_size=1):
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

    def acquire(self):
        return self.pool.get()

    def release(self, executor):
        if isinstance(executor, CFMGraphExecutor):
            self.pool.put(executor)

    def execute(self, input_tensor, his_lat, cfg_strength=2.0, sigma=0.25, temperature=0.0):
        executor = self.acquire()
        try:
            gen_lat, inputs_embeds, stop_out = executor.execute(
                input_tensor, his_lat, cfg_strength=cfg_strength, sigma=sigma, temperature=temperature
            )
        finally:
            self.release(executor)
            return gen_lat, inputs_embeds, stop_out
