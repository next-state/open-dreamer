"""Read i-th record from ArrayRecord shards."""

import argparse
import pickle

import grain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=int, help="Record index (maps to shard number)")
    parser.add_argument("--data-dir", default="/home/minecraft-vpt/contractor_demos/8xx_Jun_29/")
    args = parser.parse_args()

    # Each shard has 1 record, so index == shard number
    path = f"{args.data_dir}/shard-{args.index:05d}.array_record"
    source = grain.sources.ArrayRecordDataSource([path])
    data = pickle.loads(source[0])

    print(f"Record {args.index}")
    for k, v in data.items():
        if isinstance(v, bytes):
            print(f"  {k}: bytes[{len(v)}]")
        elif isinstance(v, list):
            print(f"  {k}: {type(v).__name__}[{len(v)}]")
        elif isinstance(v, tuple):
            print(f"  {k}: {type(v).__name__}[{len(v)}]")
            print(v)
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
