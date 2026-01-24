#!/usr/bin/env python3
"""Test dataloader throughput with different configurations."""
import sys
import time
import argparse

sys.path.insert(0, '.')

import grain
from dreamer.data import EpisodeLengthFilter, MinecraftVPTProcessEpisodeAndSlice
from dreamer.configs import DatasetConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=64)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--worker_buffer', type=int, default=4)
    parser.add_argument('--prefetch', type=int, default=8)
    parser.add_argument('--shards', type=int, default=100)
    parser.add_argument('--data_path', type=str, default='/home/ubuntu/minecraft-vpt/arrayrecords-mp4')
    parser.add_argument('--batches', type=int, default=100)
    args = parser.parse_args()

    cfg = DatasetConfig(
        name='minecraft_vpt',
        array_record_path=args.data_path,
        index_max=args.shards,
        B=args.batch,
        T=16,
        H=360,
        W=640,
        C=3,
    )

    array_record_paths = [f'{cfg.array_record_path}/shard-{i:05d}.array_record' for i in range(cfg.index_max)]
    source = grain.sources.ArrayRecordDataSource(array_record_paths)
    sampler = grain.samplers.IndexSampler(
        num_records=len(source),
        shard_options=grain.sharding.ShardByJaxProcess(drop_remainder=True),
        shuffle=True,
        num_epochs=None,
        seed=42,
    )
    operations = [
        EpisodeLengthFilter(seq_len=cfg.T, print_filter_warnings=False),
        MinecraftVPTProcessEpisodeAndSlice(seq_len=cfg.T),
        grain.transforms.Batch(batch_size=cfg.B, drop_remainder=True),
    ]

    print(f'workers={args.workers}, batch={args.batch}, worker_buffer={args.worker_buffer}, prefetch={args.prefetch}', flush=True)
    dataloader = grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=operations,
        worker_count=args.workers,
        worker_buffer_size=args.worker_buffer,
        read_options=grain.ReadOptions(prefetch_buffer_size=args.prefetch, num_threads=4),
    )

    start = time.time()
    for i, batch in enumerate(dataloader):
        if i == 0:
            print(f'shape: {batch["videos"].shape}', flush=True)
        if i >= args.batches - 1:
            break
    elapsed = time.time() - start
    samples = args.batches * args.batch
    print(f'{samples} samples in {elapsed:.2f}s = {samples/elapsed:.1f} samples/sec', flush=True)


if __name__ == '__main__':
    main()
