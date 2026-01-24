import logging

import hydra
import jax
import jax.numpy as jnp
from flax import nnx
from omegaconf import OmegaConf
from tqdm import tqdm
from einops import rearrange, repeat

from dreamer.configs import DynamicsConfig
from dreamer.data import make_dual_iterators
from dreamer.logging import build_logger
from dreamer.models import Dynamics, Tokenizer
from dreamer.types import Actions, create_noop_action_like
from dreamer.parallel import build_parallel
from dreamer.scaling import ScalingContext
from dreamer.training import run_evaluation, shortcut_forcing_step
from dreamer.checkpointing import (
    DynamicsCheckpointBundle,
    TokenizerCheckpointBundle,
    build_checkpoint_manager,
)
from dreamer.utils import (
    count_parameters_by_component,
    setup_training_directories,
    build_lr_schedule,
    build_optimizer,
)

# Suppress absl info logs
logging.getLogger('absl').setLevel(logging.WARNING)


# ---------------------------
# Training Step
# ---------------------------

@nnx.jit(static_argnames=("packing_factor", "k_max", "B_img", "T", "categorical_action_dim"))
def encode_and_train_step(
    tokenizer: Tokenizer,
    dynamics: Dynamics,
    optimizer: nnx.Optimizer,
    videos: jnp.ndarray,      # Full batch (B, T, H, W, C)
    actions: Actions,         # Full batch (B, T, ...)
    *,
    tokenizer_key: jax.Array,
    master_key: jax.Array,
    step: int,
    packing_factor: int,
    B_img: int,               # Number of samples to treat as images
    T: int,
    categorical_action_dim: int,
    k_max: int,
    context_length: int | None,  # None = use is_causal, int = sliding window with local_window_size
):
    rngs = nnx.Rngs(mae=tokenizer_key)

    # TODO: handle edge cases B_img = 0 or B_img = B

    # Split batch:
    # Images (T=1)
    images = rearrange(videos[:B_img], 'B_img T H W C -> (B_img T) 1 H W C')
    noop_actions = create_noop_action_like(actions[:B_img], categorical_action_dim)
    noop_actions = jax.tree.map(lambda x: repeat(x, 'B_img 1 ... -> (B_img T) 1 ...', T=T), noop_actions)

    # Videos (T=T)
    videos_batch = videos[B_img:]
    actions_batch = jax.tree.map(lambda x: x[B_img:], actions)

    # Encode both batches
    latents_img, _ = tokenizer.encode(images, packing_factor=packing_factor, deterministic=True, rngs=rngs)
    latents_vid, _ = tokenizer.encode(videos_batch, packing_factor=packing_factor, deterministic=True, rngs=rngs)

    # Compute B_self values
    B_self_img = B_img * T // 2
    B_vid = videos.shape[0] - B_img
    B_self_vid = B_vid // 2

    # Training step
    step_key = jax.random.fold_in(master_key, step)
    key_img, key_vid = jax.random.split(step_key, 2)

    def loss_fn(model: Dynamics, latents, actions, rng, B_self, ctx_len):
        losses, aux = shortcut_forcing_step(
            dynamics_model=model,
            actions=actions,
            latents=latents,
            rng=rng,
            k_max=k_max,
            B_self=B_self,
            context_length=ctx_len,
            task_embeddings=None,  # Not used in dynamics pretraining
        )
        return losses['total'], aux

    # Compute gradients for images
    (loss_img, aux_img), grads_img = nnx.value_and_grad(loss_fn, has_aux=True)(
        dynamics, latents_img, noop_actions, key_img, B_self_img, None  # Images: T=1, no context_length
    )

    # Compute gradients for videos
    (loss_vid, aux_vid), grads_vid = nnx.value_and_grad(loss_fn, has_aux=True)(
        dynamics, latents_vid, actions_batch, key_vid, B_self_vid, context_length
    )

    # Aggregate gradients
    B_img_actual = latents_img.shape[0]
    B_vid_actual = latents_vid.shape[0]
    total_B = B_img_actual + B_vid_actual
    combined_grads = jax.tree.map(
        lambda g1, g2: (g1 * B_img_actual + g2 * B_vid_actual) / total_B,
        grads_img, grads_vid
    )

    # Update model with optimizer
    optimizer.update(dynamics, combined_grads)

    return aux_img, aux_vid

# ---------------------------
# Main
# ---------------------------

def run(cfg: DynamicsConfig):
    # Setup
    run_dir, ckpt_dir, vis_dir = setup_training_directories(cfg)

    # Logging
    logger = build_logger(
        logger_cfg=cfg.logger,
        config=OmegaConf.to_container(cfg, resolve=True),
        dir=str(run_dir),
    )

    # Parallelism
    mesh, data_sharding, mesh_rules = build_parallel(cfg.parallel_strategy)

    with logger, jax.set_mesh(mesh):
        key = jax.random.PRNGKey(cfg.seed)
        rng, init_key = jax.random.split(key)

        # Load pretrained tokenizer
        tokenizer_bundle = TokenizerCheckpointBundle.from_pretrained(cfg.tokenizer_ckpt, mesh_rules=mesh_rules)
        tokenizer = tokenizer_bundle.tokenizer
        tokenizer_cfg = tokenizer.cfg

        # Initialize dynamics
        dynamics = Dynamics(cfg.dynamics, mesh_rules=mesh_rules, rngs=nnx.Rngs(init_key))
        param_counts = count_parameters_by_component(dynamics)
        print(f"Parameter counts: {param_counts['total']:,}")

        # Scaling context (handles iso-FLOPs/tokens-per-param modes + CSV output)
        n_spatial = tokenizer_cfg.encoder.n_latents // cfg.dynamics.packing_factor
        scaling = ScalingContext.create(
            cfg=cfg,
            param_count=param_counts["total"],
            flops_per_step=dynamics.estimate_flops(batch_size=cfg.dataset.B, seq_length=cfg.dataset.T, n_spatial=n_spatial),
            data_tokens_per_step=cfg.dataset.B * cfg.dataset.T * (n_spatial + 1),  # spatial + action
            total_tokens_per_step=cfg.dataset.B * cfg.dataset.T * (3 + n_spatial + cfg.dynamics.n_register),  # action + signal + step + spatial + register
            logger=logger,
            run_dir=run_dir,
        )

        # Build learning rate schedule
        lr_schedule = build_lr_schedule(cfg.lr_schedule)

        # Build optimizer
        optimizer = build_optimizer(cfg.optimizer, dynamics, lr_schedule, d_model=cfg.dynamics.d_model)

        # Create checkpoint bundle (includes frozen tokenizer for self-contained checkpoints)
        bundle = DynamicsCheckpointBundle(
            dynamics=dynamics,
            tokenizer=tokenizer,
            dynamics_optimizer=optimizer,
        )

        # Data iterators
        short_T = cfg.short_T
        long_T = cfg.long_T
        short_dataloader, long_dataloader = make_dual_iterators(cfg.dataset, short_T=short_T, long_T=long_T)
        short_iterator = iter(short_dataloader)
        long_iterator = iter(long_dataloader)

        with build_checkpoint_manager(
            cfg.ckpt, ckpt_dir,
            item_names=DynamicsCheckpointBundle.get_item_names(
                iterator_names=("short_dataloader_state", "long_dataloader_state")
            )
        ) as checkpoint_manager:
            # Resume from checkpoint
            iterators = {"short_dataloader_state": short_iterator, "long_dataloader_state": long_iterator}
            start_step, bundle, iterators, rng = bundle.restore(
                checkpoint_manager, iterators, rng
            )
            short_iterator = iterators["short_dataloader_state"]
            long_iterator = iterators["long_dataloader_state"]

            scaling.start_training()

            # Training loop
            pbar = tqdm(range(start_step, cfg.max_steps), initial=start_step, total=cfg.max_steps)
            for step in pbar:
                if step >= cfg.max_steps:
                    break

                rng, dispatch_key, tokenizer_key, master_key = jax.random.split(rng, num=4)

                use_long = float(jax.random.uniform(dispatch_key)) < cfg.long_batch_ratio
                if use_long:
                    batch = next(long_iterator)
                    T = long_T
                    context_length = cfg.dynamics.context_length
                else:
                    batch = next(short_iterator)
                    T = short_T
                    context_length = None  # Use default causal attention

                # Shard batch data
                videos = jax.device_put(batch["videos"], data_sharding)
                actions = jax.device_put(batch["actions"], data_sharding)

                # Training step
                B_img = int(cfg.dataset.B * cfg.image_fraction)

                aux_img, aux_vid = encode_and_train_step(
                    bundle.tokenizer, bundle.dynamics, bundle.dynamics_optimizer,
                    videos, actions,
                    tokenizer_key=tokenizer_key,
                    master_key=master_key,
                    step=step,
                    packing_factor=cfg.dynamics.packing_factor,
                    B_img=B_img,
                    T=T,
                    categorical_action_dim=cfg.dataset.categorical_action_dim,
                    k_max=cfg.dynamics.k_max,
                    context_length=context_length,
                )

                # Logging
                if logger.should_log(step):
                    metrics_img_cpu = jax.device_get(aux_img)
                    metrics_vid_cpu = jax.device_get(aux_vid)
                    scaling.on_step(step, metrics_vid_cpu)
                    logger.log(
                        step,
                        metrics={
                            "image/flow_mse": metrics_img_cpu["flow_mse"],
                            "image/boot_mse": metrics_img_cpu["bootstrap_mse"],
                            "video/flow_mse": metrics_vid_cpu["flow_mse"],
                            "video/boot_mse": metrics_vid_cpu["bootstrap_mse"],
                            "lr": lr_schedule(step),
                            "batch_type": 1.0 if use_long else 0.0,
                            **scaling.get_step_metrics(step),
                        },
                        pbar=pbar,
                    )

                # Checkpointing
                iterators = {"short_dataloader_state": short_iterator, "long_dataloader_state": long_iterator}
                bundle.maybe_save(checkpoint_manager, step, iterators, rng)

                # Periodic lightweight AR eval
                if cfg.write_video_every and (step % cfg.write_video_every == 0) and step > 0:
                    # Use subset of batch for visualization
                    val_videos = batch["videos"][:4]
                    val_actions = batch["actions"][:4]

                    run_evaluation(
                        cfg, step, bundle.tokenizer, bundle.dynamics,
                        val_videos, val_actions, vis_dir, rng, logger
                    )

            scaling.finalize()


@hydra.main(version_base=None, config_path="../configs", config_name="dynamics")
def main(cfg: DynamicsConfig):
    run(cfg)


if __name__ == "__main__":
    main()
