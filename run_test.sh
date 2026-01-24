#!/bin/bash
ulimit -n 65536
source .venv/bin/activate
ulimit -n
python test_dataloader.py "$@"
