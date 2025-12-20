import jax
import jax.numpy as jnp
from dreamer.generation import DenoiseSchedule


scheduler = DenoiseSchedule.init(4, 256, 0.9)

print(scheduler)


all_taus = jnp.arange(0,scheduler.k_max+1)

tau_jump = all_taus[scheduler.tau_idx_ctx]
tau_value = jnp.linspace(0,1,scheduler.k_max+1)

print(tau_jump, scheduler.tau_idx_ctx, tau_value[tau_jump])