"""Property tests for text-conditioned attention (cross-attention, layer, transformer).

These check behaviour, not implementation: shapes, that the text actually matters,
that padding is ignored, that the gate starts closed, and that every parameter
learns. Small dims, fixed seeds, CPU: the whole file runs in seconds.

    uv run pytest tests/test_text_conditioning.py -q
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from jax.sharding import AxisType

from dreamer.models import (
    LAYER_SPACE,
    LAYER_SPACE_TEXT,
    LAYER_TIME,
    BlockCausalTextLayer,
    BlockCausalTextTransformer,
    SpaceTextCrossAttention,
    build_text_layer_schedule,
)
from dreamer.parallel import MeshRules

B, T, S, L, D = 2, 3, 16, 5, 64
N_HEADS, N_KV = 8, 2

# Params are created with sharding annotations, so a (one-device) mesh must exist.
jax.set_mesh(
    jax.make_mesh((1, 1), ("data", "model"), axis_types=(AxisType.Auto, AxisType.Auto))
)
RULES = MeshRules(attn="model", mlp="model", data="data")


def _normal(seed, shape):
    return jax.random.normal(jax.random.key(seed), shape)


@pytest.fixture
def video():
    return _normal(0, (B, T, S, D))


@pytest.fixture
def texts():
    return _normal(1, (B, L, D)), _normal(2, (B, L, D))


@pytest.fixture
def text_mask():
    # sample 0 has 3 real words, sample 1 has 4
    return jnp.array([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=bool)


def _dead_params(grads):
    return [
        jax.tree_util.keystr(path)
        for path, g in jax.tree_util.tree_leaves_with_path(nnx.state(grads))
        if not jnp.any(g != 0)
    ]


# ---------------------------------------------------------------- cross-attention


class TestSpaceTextCrossAttention:
    @pytest.fixture
    def attn(self):
        return SpaceTextCrossAttention(
            dim=D, num_heads=N_HEADS, num_kv_heads=N_KV, mesh_rules=RULES, rngs=nnx.Rngs(0)
        )

    def test_output_has_video_shape_even_though_L_differs_from_S(
        self, attn, video, texts, text_mask
    ):
        out, cache, w = attn(video, texts[0], text_mask=text_mask, return_weights=True)
        assert out.shape == (B, T, S, D)
        assert w.shape == (B, T, N_HEADS, S, L)
        assert cache is None

    def test_text_changes_output(self, attn, video, texts, text_mask):
        out1 = attn(video, texts[0], text_mask=text_mask)[0]
        out2 = attn(video, texts[1], text_mask=text_mask)[0]
        assert not jnp.allclose(out1, out2)

    def test_padded_words_are_ignored(self, attn, video, texts, text_mask):
        text = texts[0]
        garbage_in_pads = text.at[0, 3:].set(99.0).at[1, 4:].set(-99.0)
        out1 = attn(video, text, text_mask=text_mask)[0]
        out2 = attn(video, garbage_in_pads, text_mask=text_mask)[0]
        assert jnp.allclose(out1, out2, atol=1e-5)

    def test_weights_are_a_distribution_over_real_words(
        self, attn, video, texts, text_mask
    ):
        _, _, w = attn(video, texts[0], text_mask=text_mask, return_weights=True)
        assert jnp.allclose(w.sum(-1), 1.0, atol=1e-5)
        assert jnp.allclose(w[0, ..., 3:], 0.0)
        assert jnp.allclose(w[1, ..., 4:], 0.0)

    def test_every_param_gets_gradient(self, attn, video, texts, text_mask):
        grads = nnx.grad(lambda m: m(video, texts[0], text_mask=text_mask)[0].sum())(
            attn
        )
        assert _dead_params(grads) == []


# ------------------------------------------------------------------------ schedule


def test_default_schedule_interleaves_space_text_and_time():
    assert build_text_layer_schedule(8) == (
        LAYER_SPACE, LAYER_SPACE_TEXT, LAYER_SPACE, LAYER_TIME,
        LAYER_SPACE, LAYER_SPACE_TEXT, LAYER_SPACE, LAYER_TIME,
    )  # fmt: skip


def test_schedule_rejects_bad_depth():
    with pytest.raises(ValueError):
        build_text_layer_schedule(0)


# --------------------------------------------------------------------------- layer


def _layer(kind, **kw):
    return BlockCausalTextLayer(
        dim=D,
        num_heads=N_HEADS,
        num_kv_heads=N_KV,
        layer_kind=kind,
        mesh_rules=RULES,
        rngs=nnx.Rngs(0),
        **kw,
    )


@pytest.mark.parametrize("kind", [LAYER_SPACE, LAYER_SPACE_TEXT, LAYER_TIME])
def test_layer_keeps_shape(kind, video, texts, text_mask):
    out, _, _ = _layer(kind)(video, texts[0], text_mask=text_mask)
    assert out.shape == video.shape


def test_layer_rejects_unknown_kind():
    with pytest.raises(ValueError):
        _layer("spcae")


def test_space_text_layer_requires_text(video):
    with pytest.raises(ValueError):
        _layer(LAYER_SPACE_TEXT)(video, None)


def test_closed_gate_makes_cross_attention_a_noop_at_init(video, texts, text_mask):
    layer = _layer(LAYER_SPACE_TEXT)
    out1 = layer(video, texts[0], text_mask=text_mask)[0]
    out2 = layer(video, texts[1], text_mask=text_mask)[0]
    assert jnp.allclose(out1, out2)


def test_closed_gate_still_receives_gradient(video, texts, text_mask):
    layer = _layer(LAYER_SPACE_TEXT)
    grads = nnx.grad(lambda m: m(video, texts[0], text_mask=text_mask)[0].sum())(layer)
    assert jnp.any(grads.text_gate.value != 0)


def test_opened_gate_lets_text_through(video, texts, text_mask):
    layer = _layer(LAYER_SPACE_TEXT)
    layer.text_gate.value = jnp.ones_like(layer.text_gate.value)
    out1 = layer(video, texts[0], text_mask=text_mask)[0]
    out2 = layer(video, texts[1], text_mask=text_mask)[0]
    assert not jnp.allclose(out1, out2)


# --------------------------------------------------------------------- transformer


def _transformer(**kw):
    defaults = dict(
        d_model=D,
        n_heads=N_HEADS,
        n_kv_heads=N_KV,
        depth=8,
        mesh_rules=RULES,
        rngs=nnx.Rngs(0),
    )
    return BlockCausalTextTransformer(**{**defaults, **kw})


def test_transformer_projects_text_of_a_different_width(video, text_mask):
    model = _transformer(text_dim=32)
    text = _normal(3, (B, L, 32))
    out, _, weights = model(video, text, text_mask=text_mask, return_weights=True)
    assert out.shape == video.shape
    assert len(weights) == 8
    for kind, w in zip(model.layer_schedule, weights):
        if kind == LAYER_SPACE_TEXT:
            assert w.shape == (B, T, N_HEADS, S, L)


def test_transformer_without_gate_uses_text_and_trains_everything(
    video, texts, text_mask
):
    model = _transformer(use_text_gate=False)
    out1 = model(video, texts[0], text_mask=text_mask)[0]
    out2 = model(video, texts[1], text_mask=text_mask)[0]
    assert not jnp.allclose(out1, out2)

    grads = nnx.grad(lambda m: m(video, texts[0], text_mask=text_mask)[0].sum())(model)
    assert _dead_params(grads) == []


def test_transformer_jit_matches_eager(video, texts, text_mask):
    model = _transformer(use_text_gate=False)
    eager = model(video, texts[0], text_mask=text_mask)[0]
    jitted = nnx.jit(lambda m, x, t, k: m(x, t, text_mask=k)[0])(
        model, video, texts[0], text_mask
    )
    assert jnp.allclose(eager, jitted, atol=1e-5)


def test_transformer_rejects_schedule_without_text_layer():
    with pytest.raises(ValueError):
        _transformer(depth=2, layer_schedule=(LAYER_SPACE, LAYER_TIME))


@pytest.mark.parametrize(
    "text_shape, mask_shape",
    [((B + 1, L, D), None), ((B, L), None), ((B, L, D), (B, L + 1))],
)
def test_transformer_rejects_bad_text_shapes(video, text_shape, mask_shape):
    text = jnp.zeros(text_shape)
    mask = None if mask_shape is None else jnp.ones(mask_shape, dtype=bool)
    with pytest.raises(ValueError):
        _transformer()(video, text, text_mask=mask)


def test_flops_count_text_layers_as_S_by_L():
    model = _transformer()
    with_text = model.estimate_attention_flops(B, T, S, seq_text=L)
    without_text = model.estimate_attention_flops(B, T, S, seq_text=0)
    n_text = model.layer_schedule.count(LAYER_SPACE_TEXT)
    assert with_text - without_text == 12 * n_text * D * S * L * B * T
