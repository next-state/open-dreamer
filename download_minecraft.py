import os
from huggingface_hub import snapshot_download

# Get the token you set in your environment

snapshot_download(
    repo_id="open-world-agents/vpt-owamcap",
    repo_type="dataset",
    local_dir="/scratch/vpt",
    #token=token,
    max_workers=16,
    # Note: max_workers=8 is the default, but you can increase it 
    # if you have a fast connection, e.g., max_workers=16
)