import subprocess
import time
import os
import signal

env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

p = subprocess.Popen(
    ["uv", "run", "scripts/train_tokenizer.py"],
    env=env,
    start_new_session=True,
)

time.sleep(3600)

# Graceful shutdown
os.killpg(p.pid, signal.SIGTERM)

try:
    p.wait(timeout=60)
except subprocess.TimeoutExpired:
    os.killpg(p.pid, signal.SIGKILL)
    p.wait()

subprocess.run(
    ["uv", "run", "scripts/train_dynamics.py"],
    env=env,
)
