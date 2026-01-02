import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from functools import partial

# 1. Setup simulated 2-device mesh
devices = jax.devices()
if len(devices) < 2:
    print("Warning: Simulating 2 devices on host for demonstration")
    mesh = Mesh(devices * 2, axis_names=('data',))
else:
    mesh = Mesh(devices[:2], axis_names=('data',))

# 2. Define Specs
replicated_spec = NamedSharding(mesh, P())       # Key is everywhere
sharded_spec    = NamedSharding(mesh, P('data')) # Output is split

# 3. Define the function (Pure Python first)
def generate_sharded_noise_impl(key):
    # Global shape is (4,). 
    # Device 0 is responsible for indices [0, 1]
    # Device 1 is responsible for indices [2, 3]
    return jax.random.normal(key, shape=(4,))

# 4. Apply JIT with explicit arguments (Fixes the TypeError)
# We tell JAX: "Input is replicated, but Output MUST be split across devices"
generate_sharded_noise = jax.jit(
    generate_sharded_noise_impl, 
    out_shardings=sharded_spec
)

# --- TEST ---
key = jax.random.key(42)
replicated_key = jax.device_put(key, replicated_spec)

# Run the jitted function
output_sharded = generate_sharded_noise(replicated_key)

print("\n=== CORRECTED PROOF: Replicated Key -> Sharded Output ===")
d0_jax = output_sharded.addressable_data(0)
d1_jax = output_sharded.addressable_data(1)

# Move to CPU for comparison
d0_cpu = np.array(d0_jax)
d1_cpu = np.array(d1_jax)

print(f"Device 0 Chunk: {d0_cpu}")
print(f"Device 1 Chunk: {d1_cpu}")

# Comparison
if not np.allclose(d0_cpu, d1_cpu):
    print("✅ SUCCESS: Device 0 and Device 1 hold DIFFERENT noise segments.")
    print("   (JAX automatically split the single key into unique noise per shard)")
else:
    print("❌ FAIL: Identical noise.")