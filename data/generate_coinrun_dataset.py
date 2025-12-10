"""
Generates a dataset of random-action CoinRun episodes.
Episodes are saved individually as memory-mapped files for efficient loading.
"""

from dataclasses import dataclass

import numpy as np
from procgen import ProcgenGym3Env
import tyro
import json
import os
import pickle
from gym3 import types_np
from array_record.python.array_record_module import ArrayRecordWriter


def save_chunks(file_idx, chunks_per_file, output_dir, obs_chunks, act_chunks=None):
    os.makedirs(output_dir, exist_ok=True)

    metadata = []
    while len(obs_chunks) >= chunks_per_file:
        chunk_batch = obs_chunks[:chunks_per_file]
        obs_chunks = obs_chunks[chunks_per_file:]
        act_chunk_batch = None
        if act_chunks:
            act_chunk_batch = act_chunks[:chunks_per_file]
            act_chunks = act_chunks[chunks_per_file:]
        episode_path = os.path.join(output_dir, f"data_{file_idx:04d}.array_record")
        writer = ArrayRecordWriter(str(episode_path), "group_size:1")
        seq_lens = []
        for idx, chunk in enumerate(chunk_batch):
            seq_len = chunk.shape[0]
            seq_lens.append(seq_len)
            chunk_record = {
                "raw_video": chunk.tobytes(),
                "sequence_length": seq_len,
            }
            if act_chunk_batch:
                assert len(chunk) == len(
                    act_chunk_batch[idx]
                ), f"Observation data length and action sequence length do not match: {len(chunk)} != {len(act_chunk_batch[idx])}"
                chunk_record["actions"] = act_chunk_batch[idx]
            writer.write(pickle.dumps(chunk_record))
        writer.close()
        file_idx += 1
        metadata.append(
            {
                "path": episode_path,
                "num_chunks": len(chunk_batch),
                "avg_seq_len": np.mean(seq_lens),
            }
        )
        print(f"Created {episode_path} with {len(chunk_batch)} video chunks")

    return metadata, file_idx, obs_chunks, act_chunks


@dataclass
class Args:
    num_episodes_train: int = 10000
    num_episodes_val: int = 500
    num_episodes_test: int = 500
    output_dir: str = "datasets/coinrun_episodes"
    min_episode_length: int = 1000
    max_episode_length: int = 1000
    chunk_size: int = 160
    chunks_per_file: int = 100
    seed: int = 0


args = tyro.cli(Args)
assert (
    args.max_episode_length >= args.min_episode_length
), "Maximum episode length must be greater than or equal to minimum episode length."

if args.min_episode_length < args.chunk_size:
    print(
        "Warning: Minimum episode length is smaller than chunk size. Note that episodes shorter than the chunk size will be discarded."
    )


# --- Generate episodes ---
def generate_episodes(num_episodes, split):
    episode_idx = 0
    episode_metadata = []
    obs_chunks = []
    act_chunks = []
    file_idx = 0
    output_dir_split = os.path.join(args.output_dir, split)
    while episode_idx < num_episodes:
        seed = np.random.randint(0, 10000)
        env = ProcgenGym3Env(num=1, env_name="coinrun", start_level=seed)

        observations_seq = []
        actions_seq = []
        episode_obs_chunks = []
        episode_act_chunks = []

        # --- Run episode ---
        step_t = 0
        first_obs = True
        for step_t in range(args.max_episode_length):
            _, obs, first = env.observe()
            action = types_np.sample(env.ac_space, bshape=(env.num,))
            env.act(action)
            observations_seq.append(obs["rgb"])
            actions_seq.append(action)
            if len(observations_seq) == args.chunk_size:
                episode_obs_chunks.append(observations_seq)
                episode_act_chunks.append(actions_seq)
                observations_seq = []
                actions_seq = []
            if first and not first_obs:
                break
            first_obs = False

        # --- Save episode ---
        if step_t + 1 >= args.min_episode_length:
            if observations_seq:
                if len(observations_seq) < args.chunk_size:
                    print(
                        f"Warning: Inconsistent chunk_sizes. Episode has {len(observations_seq)} frames, "
                        f"which is smaller than the requested chunk_size: {args.chunk_size}. "
                        "This might lead to performance degradation during training."
                    )
                episode_obs_chunks.append(observations_seq)
                episode_act_chunks.append(actions_seq)

            obs_chunks_data = [
                np.concatenate(seq, axis=0).astype(np.uint8)
                for seq in episode_obs_chunks
            ]
            act_chunks_data = [
                np.concatenate(act, axis=0) for act in episode_act_chunks
            ]
            obs_chunks.extend(obs_chunks_data)
            act_chunks.extend(act_chunks_data)

            ep_metadata, file_idx, obs_chunks, act_chunks = save_chunks(
                file_idx, args.chunks_per_file, output_dir_split, obs_chunks, act_chunks
            )
            episode_metadata.extend(ep_metadata)

            print(f"Episode {episode_idx} completed, length: {step_t + 1}.")
            episode_idx += 1
        else:
            print(f"Episode too short ({step_t + 1}), resampling...")

    if len(obs_chunks) > 0:
        print(
            f"Warning: Dropping {len(obs_chunks)} chunks for consistent number of chunks per file.",
            "Consider changing the chunk_size and chunks_per_file parameters to prevent data-loss.",
        )

    print(f"Done generating {split} split")
    return episode_metadata


def get_action_space():
    env = ProcgenGym3Env(num=1, env_name="coinrun", start_level=0)
    return env.ac_space.eltype.n


def main():
    # Set random seed and create dataset directories
    np.random.seed(args.seed)
    # --- Generate episodes ---
    train_episode_metadata = generate_episodes(args.num_episodes_train, "train")
    val_episode_metadata = generate_episodes(args.num_episodes_val, "val")
    test_episode_metadata = generate_episodes(args.num_episodes_test, "test")

    # --- Save metadata ---
    metadata = {
        "env": "coinrun",
        "num_actions": get_action_space(),
        "num_episodes_train": args.num_episodes_train,
        "num_episodes_val": args.num_episodes_val,
        "num_episodes_test": args.num_episodes_test,
        "avg_episode_len_train": np.mean(
            [ep["avg_seq_len"] for ep in train_episode_metadata]
        ),
        "avg_episode_len_val": np.mean(
            [ep["avg_seq_len"] for ep in val_episode_metadata]
        ),
        "avg_episode_len_test": np.mean(
            [ep["avg_seq_len"] for ep in test_episode_metadata]
        ),
        "episode_metadata_train": train_episode_metadata,
        "episode_metadata_val": val_episode_metadata,
        "episode_metadata_test": test_episode_metadata,
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)

    print(f"Done generating dataset.")


if __name__ == "__main__":
    main()