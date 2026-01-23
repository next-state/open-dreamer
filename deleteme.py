from dreamer.data import make_iterator
from dreamer.configs import DatasetConfig

if __name__ == '__main__':
    cfg = DatasetConfig(
        name="minecraft_vpt",
        array_record_path="/scratch/vpt_arrayrecord",
        index_max=100,  # start small
        B=4,
        T=16,
    )
    print(f"dataset {cfg}")
    loader = make_iterator(cfg, use_decord=True, num_workers=4)
    print(f"loader {loader}")
    batch = next(iter(loader))
    print(batch["videos"].shape)   # (4, 16, 360, 640, 3)
    print(batch["actions"].shape)  # (4, 16, 22)
