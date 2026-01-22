import logging

import hydra
import jax
import jax.numpy as jnp
from flax import nnx
from omegaconf import OmegaConf
from tqdm import tqdm
from functools import partial
from einops import rearrange, repeat

from dreamer.configs import DynamicsConfig
from dreamer.data import make_iterator
from dreamer.logging import build_logger
from dreamer.models import Dynamics, Tokenizer
from dreamer.types import Actions, get_noop_action_like
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

@nnx.jit(static_argnames=("packing_factor", "k_max", "B_self"))
def encode_and_train_step(
    tokenizer: Tokenizer,
    dynamics: Dynamics,
    optimizer: nnx.Optimizer,
    videos: jnp.ndarray,
    actions: Actions,
    *,
    tokenizer_key: jax.Array,
    master_key: jax.Array,
    step: int,
    packing_factor: int,
    B_self: int,
    k_max: int,
):
    # Phase 1: Encode videos to latents
    rngs = nnx.Rngs(mae=tokenizer_key)
    latents, _ = tokenizer.encode(
        videos,
        packing_factor=packing_factor,
        deterministic=True,
        rngs=rngs
    )

    # Phase 2: Training step
    metrics = train_step(
        dynamics, optimizer, latents, actions,
        B_self=B_self, k_max=k_max, master_key=master_key, step=step
    )

    return metrics


@nnx.jit(static_argnames=("k_max", "B_self"))
def train_step(
    dynamics: Dynamics,
    optimizer: nnx.Optimizer,
    latents: jnp.ndarray,
    actions: Actions,
    *,
    B_self: int,
    k_max: int,
    master_key: jax.Array,
    step: int
):
    # Generate step-specific key
    step_key = jax.random.fold_in(master_key, step)

    def loss_and_aux(dynamics_model: Dynamics):
        """Loss function that takes the model and returns (loss, aux)."""
        losses, aux = shortcut_forcing_step(
            dynamics_model=dynamics_model,
            actions=actions,
            latents=latents,
            rng=step_key,
            k_max=k_max,
            B_self=B_self,
            task_embeddings=None,  # Not used in dynamics pretraining
        )
        return losses['total'], aux

    (loss_val, metrics), grads = nnx.value_and_grad(loss_and_aux, has_aux=True)(dynamics)

    # Update model with optimizer
    optimizer.update(dynamics, grads)

    return metrics

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

        # Data iterator
        train_dataloader = make_iterator(cfg.dataset)
        train_iterator = iter(train_dataloader)  # type: ignore

        # Train step helper
        encode_and_train_step_partial = partial(encode_and_train_step,
            bundle.tokenizer, bundle.dynamics, bundle.dynamics_optimizer,
            packing_factor=cfg.dynamics.packing_factor,
            k_max=cfg.dynamics.k_max,
        )

        with build_checkpoint_manager(
            cfg.ckpt, ckpt_dir,
            item_names=DynamicsCheckpointBundle.get_item_names()
        ) as checkpoint_manager:
            # Resume from checkpoint
            start_step, bundle, train_iterator, rng = bundle.restore(
                checkpoint_manager, train_iterator, rng
            )

            scaling.start_training()

            # Training loop
            pbar = tqdm(enumerate(train_iterator, start=start_step), initial=start_step, total=cfg.max_steps)
            for step, batch in pbar:
                if step >= cfg.max_steps:
                    break

                # Shard batch data
                rng, tokenizer_key, master_key = jax.random.split(rng, num=3)
                videos = jax.device_put(batch["videos"], data_sharding)
                actions = jax.device_put(batch["actions"], data_sharding)

                # Training step 1: treat cfg.image_fraction of the samples as separate images
                B_img = int(cfg.dataset.B * cfg.image_fraction)
                images = rearrange(videos[:B_img], 'B_img T H W C -> (B_img T) 1 H W C')
                noop_actions = get_noop_action_like(actions[:B_img], cfg.dataset)
                noop_actions = jax.tree.map(lambda x: repeat(x, 'B_img 1 ... -> (B_img T) 1 ...', T=cfg.dataset.T), noop_actions)
                aux_img = encode_and_train_step_partial(images, noop_actions, tokenizer_key=tokenizer_key, master_key=master_key, step=step, B_self=B_img * cfg.dataset.T // 2)

                # Training step 2: treat remaining samples as videos
                B_vid = cfg.dataset.B - B_img
                videos = videos[B_img:]   # (B_vid, T, H, W, C)
                actions = actions[B_img:] # (B_vid, T, ...)
                aux_vid = encode_and_train_step_partial(videos, actions, tokenizer_key=tokenizer_key, master_key=master_key, step=step, B_self=B_vid // 2)

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
                            **scaling.get_step_metrics(step),
                        },
                        pbar=pbar,
                    )

                # Checkpointing
                bundle.maybe_save(checkpoint_manager, step, train_iterator, rng)

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
