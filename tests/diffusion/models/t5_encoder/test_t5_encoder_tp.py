# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from transformers import T5Config
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

from vllm_omni.diffusion.models.t5_encoder.t5_encoder import (
    T5DenseGatedActDense,
    T5EncoderModel,
    T5SelfAttention,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_T5_MODULE = "vllm_omni.diffusion.models.t5_encoder.t5_encoder"

SMALL_T5_CONFIG = dict(
    d_model=64,
    d_kv=8,
    d_ff=128,
    num_heads=8,
    num_layers=2,
    vocab_size=256,
    relative_attention_num_buckets=32,
    relative_attention_max_distance=128,
    is_gated_act=True,
    dense_act_fn="gelu_new",
    layer_norm_epsilon=1e-6,
    feed_forward_proj="gated-gelu",
)


@pytest.fixture(scope="module")
def t5_config() -> T5Config:
    return T5Config(**SMALL_T5_CONFIG)


@pytest.fixture(scope="function", autouse=True)
def setup_tp_group(monkeypatch, mocker):
    """Set up TP=2, rank=0, VllmConfig, and mock activation for all tests."""
    device_config = DeviceConfig(device="cpu")

    # TP world size
    monkeypatch.setattr("vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(f"{_T5_MODULE}.get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(
        "vllm.model_executor.layers.vocab_parallel_embedding.get_tensor_model_parallel_world_size",
        lambda: 2,
    )

    monkeypatch.setattr(f"{_T5_MODULE}.get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        "vllm.model_executor.layers.vocab_parallel_embedding.get_tensor_model_parallel_rank",
        lambda: 0,
    )

    # TP group
    mock_tp_group = mocker.MagicMock()
    mock_tp_group.world_size = 2
    mocker.patch("vllm.distributed.parallel_state.get_tp_group", return_value=mock_tp_group)

    monkeypatch.setattr(f"{_T5_MODULE}.get_act_fn", lambda _: torch.nn.GELU())

    with set_current_vllm_config(VllmConfig(device_config=device_config)):
        yield


class TestT5SelfAttentionWeightLoading:
    """Verify QKV fusion weight loading under TP=2"""

    def test_qkv_fusion_weight_shapes(self, t5_config):
        attn = T5SelfAttention(t5_config, has_relative_attention_bias=True)

        q_weight = torch.randn(t5_config.num_heads * t5_config.d_kv, t5_config.d_model)
        k_weight = torch.randn(t5_config.num_heads * t5_config.d_kv, t5_config.d_model)
        v_weight = torch.randn(t5_config.num_heads * t5_config.d_kv, t5_config.d_model)

        loaded = attn.load_weights(
            [
                ("q.weight", q_weight),
                ("k.weight", k_weight),
                ("v.weight", v_weight),
            ]
        )

        assert len(loaded) > 0, "Should load weights"

        # TP=2: qkv_proj output = 3 * (n_heads/2) * d_kv = 96
        expected_output_dim = 3 * (t5_config.num_heads // 2) * t5_config.d_kv
        assert attn.qkv_proj.weight.shape == (expected_output_dim, t5_config.d_model), (
            f"Expected ({expected_output_dim}, {t5_config.d_model}), got {attn.qkv_proj.weight.shape}"
        )

    def test_output_projection_shape(self, t5_config):
        attn = T5SelfAttention(t5_config, has_relative_attention_bias=False)

        o_weight = torch.randn(t5_config.d_model, t5_config.num_heads * t5_config.d_kv)
        loaded = attn.load_weights([("o.weight", o_weight)])

        assert "o.weight" in loaded
        expected_input_dim = (t5_config.num_heads * t5_config.d_kv) // 2
        assert attn.o.weight.shape == (t5_config.d_model, expected_input_dim)

    def test_relative_attention_bias_loaded(self, t5_config):
        attn = T5SelfAttention(t5_config, has_relative_attention_bias=True)

        bias_weight = torch.randn(t5_config.relative_attention_num_buckets, t5_config.num_heads)
        loaded = attn.load_weights(
            [
                ("relative_attention_bias.weight", bias_weight),
            ]
        )

        assert "relative_attention_bias.weight" in loaded
        assert attn.relative_attention_bias.weight.shape == (
            t5_config.relative_attention_num_buckets,
            t5_config.num_heads,
        )


class TestRelativePositionBiasTPSlicing:
    """Verify compute_bias slices heads correctly per TP rank."""

    def test_compute_bias_shape(self, t5_config):
        attn = T5SelfAttention(t5_config, has_relative_attention_bias=True)

        seq_len = 6
        bias = attn.compute_bias(seq_len, seq_len, device=torch.device("cpu"))

        local_heads = t5_config.num_heads // 2
        assert bias.shape == (1, local_heads, seq_len, seq_len)

    def test_all_ranks_cover_all_heads(self, t5_config, monkeypatch):
        seq_len = 4

        biases = []
        ref_weight = None
        for rank in range(2):
            monkeypatch.setattr(f"{_T5_MODULE}.get_tensor_model_parallel_rank", lambda r=rank: r)
            attn = T5SelfAttention(t5_config, has_relative_attention_bias=True)
            if rank > 0:
                attn.relative_attention_bias.weight.data.copy_(ref_weight)
            else:
                ref_weight = attn.relative_attention_bias.weight.data.clone()
            biases.append(attn.compute_bias(seq_len, seq_len, device=torch.device("cpu")))

        full_bias = torch.cat(biases, dim=1)
        assert full_bias.shape == (1, t5_config.num_heads, seq_len, seq_len)


class TestT5DenseGatedActDenseWeightLoading:
    """Verify wi_0/wi_1 fusion into MergedColumnParallelLinear under TP=2."""

    def test_wi_fusion_weight_shapes(self, t5_config):
        ffn = T5DenseGatedActDense(t5_config)

        wi_0_weight = torch.randn(t5_config.d_ff, t5_config.d_model)
        wi_1_weight = torch.randn(t5_config.d_ff, t5_config.d_model)

        loaded = ffn.load_weights(
            [
                ("wi_0.weight", wi_0_weight),
                ("wi_1.weight", wi_1_weight),
            ]
        )

        assert len(loaded) > 0

        # MergedColumnParallelLinear with TP=2: d_ff/2 * 2 shards = d_ff
        expected_output_dim = t5_config.d_ff
        assert ffn.wi.weight.shape == (expected_output_dim, t5_config.d_model), (
            f"Expected ({expected_output_dim}, {t5_config.d_model}), got {ffn.wi.weight.shape}"
        )

    def test_wo_shape(self, t5_config):
        ffn = T5DenseGatedActDense(t5_config)

        wo_weight = torch.randn(t5_config.d_model, t5_config.d_ff)
        loaded = ffn.load_weights([("wo.weight", wo_weight)])

        assert "wo.weight" in loaded
        expected_input_dim = t5_config.d_ff // 2
        assert ffn.wo.weight.shape == (t5_config.d_model, expected_input_dim)


class TestT5EncoderModelWeightLoading:
    """Test weight loading at the top-level T5EncoderModel."""

    def test_model_instantiation(self, t5_config):
        model = T5EncoderModel(t5_config, prefix="text_encoder")

        assert model.config is t5_config
        assert model.encoder is not None
        assert len(model.encoder.block) == t5_config.num_layers

    def test_embedding_shape(self, t5_config):
        model = T5EncoderModel(t5_config, prefix="text_encoder")

        assert model.shared.embedding_dim == t5_config.d_model

    def test_embed_input_ids(self, t5_config, monkeypatch):
        # Verify method and output shape
        model = T5EncoderModel(t5_config, prefix="text_encoder")

        # Mock all-reduce to be identity (no actual TP communication)
        monkeypatch.setattr(
            "vllm.model_executor.layers.vocab_parallel_embedding.tensor_model_parallel_all_reduce",
            lambda x: x,
        )

        input_ids = torch.randint(0, t5_config.vocab_size, (2, 8))
        embeddings = model.embed_input_ids(input_ids)

        assert embeddings.shape == (2, 8, t5_config.d_model)

    def test_qkv_weights_loaded_through_blocks(self):
        # Verify that HF-style separate Q/K/V weights can be loaded through
        # the block hierarchy
        config = T5Config(**{**SMALL_T5_CONFIG, "num_layers": 1})
        model = T5EncoderModel(config, prefix="text_encoder")

        inner_dim = config.num_heads * config.d_kv

        attn = model.encoder.block[0].layer[0].SelfAttention
        loaded = attn.load_weights(
            [
                ("q.weight", torch.randn(inner_dim, config.d_model)),
                ("k.weight", torch.randn(inner_dim, config.d_model)),
                ("v.weight", torch.randn(inner_dim, config.d_model)),
            ]
        )

        assert len(loaded) > 0
        expected_qkv_dim = 3 * (config.num_heads // 2) * config.d_kv
        assert attn.qkv_proj.weight.shape == (expected_qkv_dim, config.d_model)


class TestTPConstraints:
    """Verify that invalid TP configurations raise clear errors."""

    def test_num_heads_not_divisible_by_tp(self):
        config = T5Config(**{**SMALL_T5_CONFIG, "num_heads": 7})
        with pytest.raises(AssertionError, match=r"num_heads.*must be divisible by tp_size"):
            T5SelfAttention(config)

    def test_num_heads_divisible_by_tp(self, t5_config):
        attn = T5SelfAttention(t5_config)
        assert attn.n_heads_per_partition == 4
