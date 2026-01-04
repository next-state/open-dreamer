from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import imageio.v2 as imageio
from einops import rearrange
import coinrun_data.dataloader as coinrun_loader
from .configs import DatasetConfig


# ============================================================
# Constants
# ============================================================

# Action space (categorical):
#   0: up, 1: down, 2: left, 3: right, 4: initial (null)
# Directions are in (dy, dx) order for image coordinates.
# Use numpy for module-level constants to avoid JAX init in Grain worker subprocesses.
ACTION_DELTAS_YX = np.array(
    [[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int32
)
NULL_ACTION = np.int32(4)


# ============================================================
# Shared painting helpers
# ============================================================


def _paint_squares_batch(
    init_background_color: jnp.ndarray,  # (B, C) uint8
    init_foreground_color: jnp.ndarray,  # (B, C) uint8
    positions: jnp.ndarray,  # (B, T, 2) int32 (y, x)
    square_sizes: jnp.ndarray,  # (B,) int32
    height: jnp.ndarray | int,  # scalar int32 or Python int
    width: jnp.ndarray | int,   # scalar int32 or Python int
    channels: jnp.ndarray | int,  # scalar int32 or Python int
) -> jnp.ndarray:
    """
    Paint axis-aligned squares onto a batch of frames.

    Args:
        init_background_color: (B, C) uint8 background colors.
        positions: (B, T, 2) int32 top-left (y, x) per frame.
        square_sizes: (B,) int32 side length per sample.
        height, width, channels: spatial dimensions (can be Python int or JAX scalar).

    Returns:
        video: (B, T, H, W, C) uint8
    """
    B = positions.shape[0]
    T = positions.shape[1]
    # Convert to JAX scalars if needed (works for both Python int and JAX array)
    H = height
    W = width
    C = channels

    ys = jnp.arange(H)
    xs = jnp.arange(W)

    # initialize backgrounds
    video = (
        jnp.ones((B, T, H, W, C), dtype=jnp.uint8)
        * init_background_color[:, None, None, None, :]
    )

    def paint_one(frame, y, x, color, k):
        """Paint one square of size (k,k) at (y,x) on a single frame."""
        ymask = (ys >= y) & (ys < y + k)
        xmask = (xs >= x) & (xs < x + k)
        mask = (ymask[:, None] & xmask[None, :])[..., None]
        return jnp.where(mask, color[None, None, :], frame)

    # Vectorize over time and batch
    paint_over_time = jax.vmap(paint_one, in_axes=(0, 0, 0, None, None))
    paint_over_batch = jax.vmap(paint_over_time, in_axes=(0, 0, 0, 0, 0))

    y_idx = positions[..., 0]
    x_idx = positions[..., 1]
    # Paint squares using the foreground color on top of the background.
    video = paint_over_batch(video, y_idx, x_idx, init_foreground_color, square_sizes)
    return video


# JIT-compiled version with static height, width, channels
_paint_squares_batch_jit = jax.jit(
    _paint_squares_batch,
    static_argnames=("height", "width", "channels"),
)


# ============================================================
# Frame generation for action-conditioned videos
# ============================================================

@partial(
    jax.jit,
    static_argnames=[
        "batch_size",
        "time_steps",
        "height",
        "width",
        "channels",
        "pixels_per_step",
    ],
)
def generate_batch(
    init_pos: jnp.ndarray,               # (B, 2) int32 — initial square top-left (y, x)
    init_background_color: jnp.ndarray,  # (B, C) uint8 — per-sample background color
    init_foreground_color: jnp.ndarray,  # (B, C) uint8 — per-sample square color
    square_sizes: jnp.ndarray,           # (B,) int32  — side length per sample
    actions: jnp.ndarray,                # (B, T-1) int32 ∈ {0,1,2,3}
    batch_size: int,
    time_steps: int,
    height: int,
    width: int,
    channels: int,
    pixels_per_step: int = 1,
):
    """
    Simulate a batch of action-conditioned bouncing-square videos.

    Args:
        init_pos: (B,2) top-left corner (y,x) for each sample
        init_background_color: (B,C) RGB background colors, uint8
        init_foreground_color: (B,C) RGB foreground colors, uint8
        square_sizes: (B,) per-sample square side lengths
        actions: (B,T-1) categorical directions in {0:up,1:down,2:left,3:right}
        batch_size, time_steps, height, width, channels: scene dimensions
        pixels_per_step: number of pixels moved per action step

    Returns:
        video:   (B,T,H,W,C) float32 in [0,1]
        actions: (B,T-1) int32
        rewards: (B,T) float32 - negative pixel distance from square center to
            image center (higher reward = closer to center).
            r_0 is dummy (NaN), r_t (t>=1) is reward from action a_t taken from s_{t-1}
    """
    H, W = height, width
    k_b = jnp.clip(square_sizes, 1, jnp.minimum(H, W))  # ensure squares fit
    deltas = ACTION_DELTAS_YX * pixels_per_step         # motion step table

    # --- Integrate positions over time for one sample ---
    def integrate_one(p0, k, acts):
        """Integrate position sequence given actions for one sample."""
        max_y = H - k
        max_x = W - k

        def step(pos, a):
            dy, dx = deltas[a]
            y, x = pos

            # update position
            y_next = y + dy
            x_next = x + dx

            # reflect off vertical boundaries
            y_next = jnp.where(y_next < 0, -y_next, y_next)
            y_next = jnp.where(y_next > max_y, 2 * max_y - y_next, y_next)

            # reflect off horizontal boundaries
            x_next = jnp.where(x_next < 0, -x_next, x_next)
            x_next = jnp.where(x_next > max_x, 2 * max_x - x_next, x_next)

            nxt = jnp.stack([y_next, x_next])
            return nxt, nxt

        # scan over (T-1) actions, accumulate all positions
        _, pos_seq = jax.lax.scan(step, p0, acts)
        positions = jnp.concatenate([p0[None, :], pos_seq], 0)  # include initial
        return positions

    # Vectorize over batch dimension
    positions = jax.vmap(integrate_one)(init_pos, k_b, actions)  # (B,T,2)

    # --- Compute rewards: distance from square center to image center ---
    # According to Dreamer convention:
    # - r_0 is dummy (no action taken yet)
    # - r_t (for t >= 1) is the reward from taking action a_t from state s_{t-1}
    # So we compute rewards only for positions[1:] (states after actions)
    # Image center coordinates
    image_center_y = H / 2.0
    image_center_x = W / 2.0
    image_center = jnp.array([image_center_y, image_center_x])
    
    # Square centers: positions are top-left corners, so add half the square size
    # positions: (B, T, 2) where last dim is (y, x)
    # k_b: (B,) -> need to expand to (B, T, 1) for broadcasting
    k_b_expanded = k_b[:, None, None]  # (B, 1, 1)
    square_centers = positions.astype(jnp.float32) + k_b_expanded / 2.0  # (B, T, 2)
    
    # Compute pixel distance: sqrt((y_center - H/2)^2 + (x_center - W/2)^2)
    # Only compute rewards for timesteps [1, T] (skip initial state at t=0)
    distances = square_centers[:, 1:, :] - image_center  # (B, T-1, 2)
    # Use negative distance as reward so that higher return corresponds to
    # staying closer to the image center (hovering near center).
    rewards_t1 = -jnp.linalg.norm(distances, axis=-1)  # (B, T-1)
    
    # Prepend dummy reward at t=0 (use NaN so any accidental use will be obvious)
    dummy_reward = jnp.full((batch_size, 1), jnp.nan, dtype=jnp.float32)
    rewards = jnp.concatenate([dummy_reward, rewards_t1], axis=1)  # (B, T)

    # --- Paint frames based on positions ---
    ys = jnp.arange(H)
    xs = jnp.arange(W)

    # initialize backgrounds
    video = (
        jnp.ones((batch_size, time_steps, H, W, channels), dtype=jnp.uint8)
        * init_background_color[:, None, None, None, :]
    )

    def paint_one(frame, y, x, color, k):
        """Paint one square of size (k,k) at (y,x) on a single frame."""
        ymask = (ys >= y) & (ys < y + k)
        xmask = (xs >= x) & (xs < x + k)
        mask = (ymask[:, None] & xmask[None, :])[..., None]
        return jnp.where(mask, color[None, None, :], frame)

    # Vectorize over time and batch
    paint_over_time = jax.vmap(paint_one, in_axes=(0, 0, 0, None, None))
    paint_over_batch = jax.vmap(paint_over_time, in_axes=(0, 0, 0, 0, 0))

    y_idx = positions[..., 0]
    x_idx = positions[..., 1]
    video = paint_over_batch(video, y_idx, x_idx, init_foreground_color, k_b)
    return (video.astype(jnp.float32) / 255.0, actions, rewards)


# ============================================================
# JIT-friendly iterator with stochastic policy
# ============================================================

def make_bouncing_square_iterator(
    batch_size: int,
    time_steps: int,
    height: int,
    width: int,
    channels: int,
    pixels_per_step: int = 1,
    size_min: int = 5,
    size_max: int = 12,
    hold_min: int = 3,
    hold_max: int = 8,
    fg_min_color: int = 0,
    fg_max_color: int = 255,
    bg_min_color: int = 0,
    bg_max_color: int = 255,
):
    """
    Construct a JIT-friendly data iterator for stochastic action-conditioned videos.

    The iterator implements a *commit-then-switch* policy:
      - The agent commits to its current direction for a random number of steps
        in [hold_min, hold_max].
      - When that interval expires, it switches to a different direction,
        chosen uniformly from the remaining 3.

    Returns:
        next_fn(key) -> (new_key, (video, actions, rewards))

    Args:
        batch_size: number of videos per batch
        time_steps: number of frames per video
        height, width, channels: spatial dimensions
        pixels_per_step: step size for each move
        size_min, size_max: range of square sizes
        hold_min, hold_max: range of commit durations
        fg_min_color, fg_max_color: range of foreground colors
        bg_min_color, bg_max_color: range of background colors
    """
    # Wrap renderer with static shape args
    gen = jax.jit(
        lambda pos, bg, fg, sizes, acts: generate_batch(
            pos, bg, fg, sizes, acts,
            batch_size=batch_size,
            time_steps=time_steps,
            height=height,
            width=width,
            channels=channels,
            pixels_per_step=pixels_per_step,
        )
    )

    Tm1 = time_steps - 1

    # --- Define per-sample stochastic policy ---
    def _sample_actions_for_one(key):
        """
        Generate (T-1,) integer action sequence using the commit-then-switch policy.

        Args:
            key: PRNGKey for one sample

        Returns:
            actions_t: (T-1,) int32 in {0,1,2,3}
        """
        # Split RNGs for reproducible scan keys
        step_keys = jax.random.split(key, Tm1 * 2).reshape(Tm1, 2, 2)

        # Initialize with remaining=0 to force a new direction at t=0
        init_action = jnp.int32(0)
        init_remaining = jnp.int32(0)

        def one_step(carry, keys_pair):
            cur_action, remaining = carry
            k_a, k_h = keys_pair

            # sample a new direction (different from current)
            r = jax.random.randint(k_a, (), 0, 3, dtype=jnp.int32)
            new_action = r + (r >= cur_action)  # skip over current

            # sample new hold length in [hold_min, hold_max]
            new_hold = jax.random.randint(k_h, (), hold_min, hold_max + 1, dtype=jnp.int32)

            need_new = (remaining <= 0)
            cur_action = jnp.where(need_new, new_action, cur_action)
            remaining = jnp.where(need_new, new_hold, remaining)

            # emit current action, decrement counter
            action_out = cur_action
            next_remaining = remaining - 1
            return (cur_action, next_remaining), action_out

        (_, _), actions_t = jax.lax.scan(one_step, (init_action, init_remaining), step_keys)
        return actions_t

    # Vectorize across batch
    sample_actions_batch = jax.vmap(_sample_actions_for_one)

    # --- Define main iterator step ---
    @jax.jit
    def next(key):
        """
        One iterator step.

        Samples random initial positions, colors, sizes, and actions,
        then calls `generate_action_batch` to produce a full video batch.

        Args:
            key: PRNGKey

        Returns:
            new_key: updated PRNGKey
            (video, actions, rewards):
                video   (B,T,H,W,C) float32 in [0,1]
                actions (B,T) int32 in {0,1,2,3,4} - a_0 is dummy (4), a_t (t>=1) are actual actions
                rewards (B,T) float32 - r_0 is dummy (NaN), r_t (t>=1) is reward from action a_t
        """
        key, sub = jax.random.split(key)
        k_pos, k_bg, k_fg, k_size, k_act = jax.random.split(sub, 5)

        # sample initial states
        init_pos = jax.random.randint(
            k_pos, (batch_size, 2),
            minval=jnp.array([0, 0]),
            maxval=jnp.array([height, width]),
            dtype=jnp.int32,
        )
        init_bg = jax.random.randint(
            k_bg, (batch_size, channels),
            bg_min_color, bg_max_color + 1, dtype=jnp.uint8,
        )
        init_fg = jax.random.randint(
            k_fg, (batch_size, channels),
            fg_min_color, fg_max_color + 1, dtype=jnp.uint8,
        )
        sizes = jax.random.randint(
            k_size, (batch_size,),
            size_min, size_max + 1,
            dtype=jnp.int32,
        )

        # sample actions using stochastic policy
        act_keys = jax.random.split(k_act, batch_size)
        actions = sample_actions_batch(act_keys)  # (B,T-1)

        # generate video and rewards
        video, _, rewards = gen(init_pos, init_bg, init_fg, sizes, actions)
        # prepend an empty action (value = 4) at t=0
        actions = jnp.concatenate([jnp.full((batch_size,1), 4, dtype=actions.dtype), actions], axis=1)
        return key, (video, actions, rewards)

    return next



def make_iterator(cfg: DatasetConfig, seq_len: int | None = None):
    """
    Factory function to create a data iterator based on config.

    Args:
        cfg: Dataset configuration
        seq_len: Override sequence length (if None, uses cfg.T)
    """
    T = seq_len if seq_len is not None else cfg.T

    if cfg.source == "bouncing_square":
        if cfg.diversify_data:
            fg_min, fg_max = 0, 255
            bg_min, bg_max = 0, 255
        else:
            fg_min, fg_max = 128, 128
            bg_min, bg_max = 255, 255

        return make_bouncing_square_iterator(
            batch_size=cfg.B,
            time_steps=T,
            height=cfg.H,
            width=cfg.W,
            channels=cfg.C,
            pixels_per_step=cfg.pixels_per_step,
            size_min=cfg.size_min,
            size_max=cfg.size_max,
            hold_min=cfg.hold_min,
            hold_max=cfg.hold_max,
            fg_min_color=fg_min,
            fg_max_color=fg_max,
            bg_min_color=bg_min,
            bg_max_color=bg_max,
        )
    elif cfg.source == "custom":
        return coinrun_loader.get_dataloader(
            array_record_paths=cfg.array_record_path,
            seq_len=T,
            global_batch_size=cfg.B,
            image_h=cfg.H,
            image_w=cfg.W,
            image_c=cfg.C,
            num_workers=22,
            print_filter_warnings=False,
        )
    else:
        raise ValueError(f"Unknown dataset source: {cfg.source}")



# ============================================================
# Simple batched environment API (reset / step)
# ============================================================


def env_reset(
    key: jax.Array,
    *,
    batch_size: int,
    height: int,
    width: int,
    channels: int,
    pixels_per_step: int = 1,
    size_min: int = 5,
    size_max: int = 12,
    fg_min_color: int = 0,
    fg_max_color: int = 255,
    bg_min_color: int = 0,
    bg_max_color: int = 255,
):
    """
    Batched environment reset for the bouncing-square world.

    Returns:
        env_state: dict PyTree containing static env parameters and per-env state.
        obs0:      (B, H, W, C) float32 in [0,1]
        a0:        (B,) int32 null actions (value = 4)
        r0:        (B,) float32 dummy rewards (NaN)
    """
    B = batch_size
    H, W, C = height, width, channels

    k_pos, k_bg, k_fg, k_size = jax.random.split(key, 4)

    # sample initial states
    init_pos = jax.random.randint(
        k_pos,
        (B, 2),
        minval=jnp.array([0, 0]),
        maxval=jnp.array([H, W]),
        dtype=jnp.int32,
    )
    init_bg = jax.random.randint(
        k_bg,
        (B, C),
        bg_min_color,
        bg_max_color + 1,
        dtype=jnp.uint8,
    )
    init_fg = jax.random.randint(
        k_fg,
        (B, C),
        fg_min_color,
        fg_max_color + 1,
        dtype=jnp.uint8,
    )
    sizes = jax.random.randint(
        k_size,
        (B,),
        size_min,
        size_max + 1,
        dtype=jnp.int32,
    )
    # ensure squares fit
    k_b = jnp.clip(sizes, 1, jnp.minimum(H, W))

    # positions for the initial frame only, shape (B, 1, 2)
    positions0 = init_pos[:, None, :]
    # Use JIT-compiled version with static args (H, W, C are Python ints)
    video0_uint8 = _paint_squares_batch_jit(
        init_background_color=init_bg,
        init_foreground_color=init_fg,
        positions=positions0,
        square_sizes=k_b,
        height=H,
        width=W,
        channels=C,
    )
    obs0 = video0_uint8[:, 0].astype(jnp.float32) / 255.0  # (B, H, W, C)

    a0 = jnp.full((B,), NULL_ACTION, dtype=jnp.int32)
    r0 = jnp.full((B,), jnp.nan, dtype=jnp.float32)

    env_state = {
        "pos": init_pos,
        "square_size": k_b,
        "bg_color": init_bg,
        "fg_color": init_fg,
        "height": H,  # Store as Python int (static)
        "width": W,   # Store as Python int (static)
        "channels": C,  # Store as Python int (static)
        "pixels_per_step": pixels_per_step,  # Store as Python int (static)
    }

    return env_state, obs0, a0, r0


def env_step(env_state: dict, actions: jnp.ndarray, *, height: int, width: int, channels: int):
    """
    Batched environment step.

    Args:
        env_state: dict from `env_reset`.
        actions:   (B,) int32 in {0,1,2,3}
        height, width, channels: static spatial dimensions (must match env_state values)

    Returns:
        env_state_next: updated state dict
        obs_next:       (B, H, W, C) float32 in [0,1]
        rewards_next:   (B,) float32 reward from taking `actions`
        dones_next:     (B,) bool (currently always False)
    """
    pos = env_state["pos"]  # (B, 2)
    k_b = env_state["square_size"]  # (B,)
    H = height  # Use static arg
    W = width   # Use static arg
    C = channels  # Use static arg
    pixels_per_step = env_state["pixels_per_step"]

    B = pos.shape[0]

    # Motion deltas per action
    deltas = ACTION_DELTAS_YX * pixels_per_step  # (4, 2)
    dy_dx = deltas[actions]  # (B, 2)
    dy = dy_dx[:, 0]
    dx = dy_dx[:, 1]

    y = pos[:, 0]
    x = pos[:, 1]

    # Update positions with reflection at boundaries
    max_y = H - k_b
    max_x = W - k_b

    y_next = y + dy
    x_next = x + dx

    y_next = jnp.where(y_next < 0, -y_next, y_next)
    y_next = jnp.where(y_next > max_y, 2 * max_y - y_next, y_next)

    x_next = jnp.where(x_next < 0, -x_next, x_next)
    x_next = jnp.where(x_next > max_x, 2 * max_x - x_next, x_next)

    pos_next = jnp.stack([y_next, x_next], axis=-1)  # (B, 2)

    # Reward: distance from square center to image center.
    H_f = jnp.asarray(H, dtype=jnp.float32)
    W_f = jnp.asarray(W, dtype=jnp.float32)
    image_center = jnp.array([H_f / 2.0, W_f / 2.0], dtype=jnp.float32)

    square_centers = pos_next.astype(jnp.float32) + k_b[:, None] / 2.0  # (B, 2)
    distances = square_centers - image_center  # (B, 2)
    rewards_next = -jnp.linalg.norm(distances, axis=-1)  # (B,)

    # Render next observation
    positions1 = pos_next[:, None, :]  # (B, 1, 2)
    # Use JIT-compiled version with static args (H, W, C are Python ints from env_state)
    video1_uint8 = _paint_squares_batch_jit(
        init_background_color=env_state["bg_color"],
        init_foreground_color=env_state["fg_color"],
        positions=positions1,
        square_sizes=k_b,
        height=H,
        width=W,
        channels=C,
    )
    obs_next = video1_uint8[:, 0].astype(jnp.float32) / 255.0  # (B, H, W, C)

    dones_next = jnp.zeros((B,), dtype=bool)

    env_state_next = dict(env_state)
    env_state_next["pos"] = pos_next

    return env_state_next, obs_next, rewards_next, dones_next


def make_env_reset_fn(
    *,
    batch_size: int,
    height: int,
    width: int,
    channels: int,
    pixels_per_step: int = 1,
    size_min: int = 5,
    size_max: int = 12,
    fg_min_color: int = 0,
    fg_max_color: int = 255,
    bg_min_color: int = 0,
    bg_max_color: int = 255,
):
    """
    Convenience wrapper that returns a JITted `reset(key)` function
    with static geometry and sampling hyperparameters bound.
    """

    @partial(
        jax.jit,
        static_argnames=(
            "batch_size",
            "height",
            "width",
            "channels",
            "pixels_per_step",
            "size_min",
            "size_max",
            "fg_min_color",
            "fg_max_color",
            "bg_min_color",
            "bg_max_color",
        ),
    )
    def _reset(
        key,
        batch_size=batch_size,
        height=height,
        width=width,
        channels=channels,
        pixels_per_step=pixels_per_step,
        size_min=size_min,
        size_max=size_max,
        fg_min_color=fg_min_color,
        fg_max_color=fg_max_color,
        bg_min_color=bg_min_color,
        bg_max_color=bg_max_color,
    ):
        return env_reset(
            key,
            batch_size=batch_size,
            height=height,
            width=width,
            channels=channels,
            pixels_per_step=pixels_per_step,
            size_min=size_min,
            size_max=size_max,
            fg_min_color=fg_min_color,
            fg_max_color=fg_max_color,
            bg_min_color=bg_min_color,
            bg_max_color=bg_max_color,
        )

    return _reset


def make_env_step_fn(*, height: int, width: int, channels: int):
    """
    Convenience wrapper that returns a JITted `step(env_state, actions)` function
    with static geometry dimensions bound.
    """

    @partial(
        jax.jit,
        static_argnames=("height", "width", "channels"),
    )
    def _step(env_state, actions, height=height, width=width, channels=channels):
        return env_step(env_state, actions, height=height, width=width, channels=channels)

    return _step

def patchify(x: jnp.ndarray, patch: int) -> jnp.ndarray:
    """
    x: (B, H, W, C)  ->  patches: (B, N, D)
      where N = (H/patch)*(W/patch), D = patch*patch*C
    """
    patches = rearrange(x, "... (hp p1) (wp p2) c -> ... (hp wp) (p1 p2 c)", p1=patch, p2=patch)
    return patches

def unpatchify(patches: jnp.ndarray, patch: int, H: int, W: int) -> jnp.ndarray:
    """
    patches: (B, N, D)  ->  x: (B, H, W, C)
      where N = (H/patch)*(W/patch), D = patch*patch*C
    """
    image = rearrange(patches, "... (hp wp) (p1 p2 c) -> ... (hp p1) (wp p2) c", hp=H//patch, wp=W//patch, p1=patch, p2=patch)
    return image




# ============================================================
# Demo / visualization
# ============================================================

def test_iterator():
    """
    Example usage of make_action_iterator.

    Generates a batch of videos where each sample moves stochastically,
    holding a direction for a random duration before switching.
    """
    key = jax.random.PRNGKey(0)
    B, T = 6, 48
    H, W, C = 64, 64, 3

    next_step = make_bouncing_square_iterator(
        batch_size=B,
        time_steps=T,
        height=H,
        width=W,
        channels=C,
        pixels_per_step=2,
        size_min=6,
        size_max=14,
        hold_min=4,
        hold_max=9,
    )

    key, (video, actions, rewards) = next_step(key)
    print("video", video.shape, video.dtype, "actions", actions.shape, actions.dtype, "rewards", rewards.shape, rewards.dtype)
    print("rewards[0] (should be NaN):", rewards[0, 0])
    print("rewards[0] is NaN:", jnp.isnan(rewards[0, 0]))
    print("rewards[1:] range:", rewards[:, 1:].min(), "to", rewards[:, 1:].max())
    print("rewards[:, 1:] mean:", rewards[:, 1:].mean())

    # Concatenate all samples horizontally per frame for visualization
    def render_frame(_, frame):
        grid = jnp.concatenate(frame, axis=1)
        return (), grid

    _, frames = jax.lax.scan(render_frame, (), video.transpose(1, 0, 2, 3, 4))
    imageio.mimsave(
        "stochastic_policy_iter.gif",
        jnp.asarray(frames * 255.0, dtype=jnp.uint8),
        fps=8, loop=1000,
    )
    print("Saved stochastic_policy_iter.gif")


def test_env_reset_draws_foreground_square():
    """
    Simple sanity check: env_reset should render at least one foreground pixel
    that differs from the background color for each sample.
    """
    key = jax.random.PRNGKey(0)
    B, H, W, C = 2, 32, 32, 3

    env_state, obs0, a0, r0 = env_reset(
        key,
        batch_size=B,
        height=H,
        width=W,
        channels=C,
        pixels_per_step=1,
        size_min=8,
        size_max=8,
        fg_min_color=255,
        fg_max_color=255,
        bg_min_color=0,
        bg_max_color=255,
    )

    assert obs0.shape == (B, H, W, C)
    assert a0.shape == (B,)
    assert r0.shape == (B,)

    # Convert observations back to uint8 for comparison.
    obs0_u8 = jnp.asarray(jnp.round(obs0 * 255.0), dtype=jnp.uint8)
    bg_colors = env_state["bg_color"]  # (B, C) uint8

    # For each sample, ensure at least one pixel differs from the background color.
    same_as_bg = obs0_u8 == bg_colors[:, None, None, :]
    all_bg_per_sample = jnp.all(same_as_bg, axis=(1, 2, 3))
    # We expect there to be at least one foreground pixel for every sample.
    assert bool(jnp.all(~all_bg_per_sample))
    print("test_env_reset_draws_foreground_square passed.")


def test_env_step_updates_position_and_image():
    """
    Simple sanity check: env_step should change positions, change images, and
    produce non-positive rewards (negative distance) with correct shapes.
    """
    key = jax.random.PRNGKey(1)
    B, H, W, C = 2, 32, 32, 3

    env_state, obs0, _, _ = env_reset(
        key,
        batch_size=B,
        height=H,
        width=W,
        channels=C,
        pixels_per_step=1,
        size_min=8,
        size_max=8,
        fg_min_color=255,
        fg_max_color=255,
        bg_min_color=0,
        bg_max_color=255,
    )

    # Take a simple deterministic action (all "down" = 1).
    actions = jnp.full((B,), 1, dtype=jnp.int32)
    env_state_next, obs_next, rewards_next, dones_next = env_step(
        env_state,
        actions,
        height=H,
        width=W,
        channels=C,
    )

    # Positions should change for at least one batch element.
    pos_changed = jnp.any(env_state_next["pos"] != env_state["pos"])
    assert bool(pos_changed)

    # Observations should differ from the initial frame for at least one pixel.
    frames_differ = jnp.any(obs_next != obs0)
    assert bool(frames_differ)

    # Rewards are non-positive (negative distance) and have the right shape.
    assert rewards_next.shape == (B,)
    assert bool(jnp.all(rewards_next <= 0.0))

    # Dones are all False for now.
    assert dones_next.shape == (B,)
    assert bool(jnp.all(dones_next == False))
    print("test_env_step_updates_position_and_image passed.")


def test_make_env_reset_step_fn_jittable():
    """
    Sanity check that the convenience wrappers for JIT-compiled reset/step
    functions run and return correctly-shaped outputs.
    """
    key = jax.random.PRNGKey(2)
    B, H, W, C = 2, 16, 16, 3

    env_reset_fn = make_env_reset_fn(
        batch_size=B,
        height=H,
        width=W,
        channels=C,
        pixels_per_step=1,
        size_min=4,
        size_max=8,
        fg_min_color=64,
        fg_max_color=255,
        bg_min_color=0,
        bg_max_color=255,
    )
    env_step_fn = make_env_step_fn(
        height=H,
        width=W,
        channels=C,
    )

    env_state, obs0, a0, r0 = env_reset_fn(key)
    assert obs0.shape == (B, H, W, C)
    assert a0.shape == (B,)
    assert r0.shape == (B,)

    # Run a few steps with random actions to ensure the compiled fn executes.
    step_keys = jax.random.split(key, 3)
    env_state_t = env_state
    obs_t = obs0
    for k in step_keys:
        actions = jax.random.randint(k, (B,), 0, 4, dtype=jnp.int32)
        env_state_t, obs_t, rew_t, done_t = env_step_fn(env_state_t, actions)
        assert obs_t.shape == (B, H, W, C)
        assert rew_t.shape == (B,)
        assert done_t.shape == (B,)
    print("test_make_env_reset_step_fn_jittable passed.")


def demo_env_api_rollout():
    """
    Small manual demo for the reset/step env API.

    Runs a short rollout with random actions and saves a GIF where each row
    corresponds to one environment in the batch.
    """
    key = jax.random.PRNGKey(0)
    B, T = 4, 32
    H, W, C = 32, 32, 3

    reset_fn = make_env_reset_fn(
        batch_size=B,
        height=H,
        width=W,
        channels=C,
        pixels_per_step=2,
        size_min=6,
        size_max=10,
        fg_min_color=128,
        fg_max_color=255,
        bg_min_color=0,
        bg_max_color=255,
    )
    step_fn = make_env_step_fn(
        height=H,
        width=W,
        channels=C,
    )

    key, sub = jax.random.split(key)
    env_state, obs0, _, _ = reset_fn(sub)

    def rollout_step(carry, t):
        env_state_t = carry
        k_t = jax.random.fold_in(key, t)
        actions_t = jax.random.randint(k_t, (B,), 0, 4, dtype=jnp.int32)
        env_state_next, obs_next, _, _ = step_fn(env_state_t, actions_t)
        return env_state_next, obs_next

    _, obs_seq = jax.lax.scan(
        rollout_step,
        env_state,
        jnp.arange(T),
    )  # (T, B, H, W, C)

    obs_seq = jnp.concatenate([obs0[None, ...], obs_seq], axis=0)  # (T+1, B, H, W, C)

    # Stack each batch horizontally per frame for visualization.
    def render_frame(frame_bt_hwc):
        return jnp.concatenate(frame_bt_hwc, axis=1)

    frames = jax.vmap(render_frame)(obs_seq)  # (T+1, H, B*W, C)
    frames_u8 = jnp.asarray(frames * 255.0, dtype=jnp.uint8)

    imageio.mimsave(
        "env_api_rollout.gif",
        frames_u8,
        fps=8,
        loop=1000,
    )
    print("Saved env_api_rollout.gif")


if __name__ == "__main__":
    # print("Running test_iterator() ...")
    # test_iterator()
    print("Running test_env_reset_draws_foreground_square() ...")
    test_env_reset_draws_foreground_square()
    print("Running test_env_step_updates_position_and_image() ...")
    test_env_step_updates_position_and_image()
    print("Running test_make_env_reset_step_fn_jittable() ...")
    test_make_env_reset_step_fn_jittable()
    print("Running demo_env_api_rollout() ...")
    demo_env_api_rollout()
