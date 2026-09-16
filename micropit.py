"""microPIT: micro Physics-Informed Transformer

Based on microgpt_jax.py; original scalar microgpt: @karpathy.

Task: noisy past positions -> future positions, for a previously unseen throw.
The model does NOT receive the initial velocity or future observations.
Physics is a soft finite-difference loss, not a built-in ballistic solver.
Set physics_weight = 0.0 for the same model without the physics loss.

Run:
    python micropit.py

Requires JAX, NumPy, and Matplotlib.
"""

import os
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg') # save figures even on a server without a display
import matplotlib.pyplot as plt

# Shared physical settings for data generation, training, and evaluation.
g = 9.8   # m/s^2; supplied gravitational acceleration
dt = 0.05 # s between samples
vx0_range = (2.0, 8.0)  # m/s; horizontal initial velocity range for training data
vy0_range = (8.0, 12.0) # m/s; vertical initial velocity range for training data

# Generate synthetic projectile observations.
# Noise is a measurement error, not a force.
def make_projectile_data(
    n_trajectories: int = 1000,
    n_samples: int = 40,
    noise_std: float = 0.05,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Return true positions and observed positions. noise_std is in meters."""

    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) * dt

    # Vary the initial velocity for each throw. The initial position is (0, 1) m.
    vx0 = rng.uniform(*vx0_range, size=n_trajectories)
    vy0 = rng.uniform(*vy0_range, size=n_trajectories)

    x = vx0[:, None] * t[None, :]
    y = 1.0 + vy0[:, None] * t[None, :] - 0.5 * g * t[None, :] ** 2
    clean = np.stack([x, y], axis=-1)       # (n_trajectories, n_timesteps, 2)

    # Add independent measurement noise for each coordinate, time step, and throw.
    noise = rng.normal(0.0, noise_std, size=clean.shape)
    observed = clean + noise

    return {
        "time": t,                          # (n_samples,), s
        "clean": clean,                     # (N, n_samples, 2), m
        "observed": observed,               # (N, n_samples, 2), m
    }

# Let there be a Dataset `train` (`docs` in the original microgpt).
# Independent throws, split BEFORE cutting windows.
# Initial position (0, 1) m; initial velocities use the shared ranges.
# NOTE: Ground collisions are not modeled; trajectories may extend below y = 0.
noise_std = 0.05 # m; measurement noise, not a random force
train = make_projectile_data(1000, noise_std=noise_std, seed=42)
# `clean` is available only because this is synthetic supervised training.
print(f"num trajectories: {len(train['clean'])}")

# Number of components per time step.
n_channel = 2        # x and y

# Convert positions from SI units to dimensionless values using a precomputed scale.
# Use the RMS of training observations (m) as the shared scale.
position_scale = float(np.sqrt(np.mean(train['observed'] ** 2)))

# Initialize the parameters, to store the knowledge of the model
n_layer = 1     # depth of the transformer neural network (number of layers)
n_embd = 16     # width of the network (embedding dimension)
n_head = 4      # number of attention heads
if n_layer < 1 or n_head <= 0 or n_embd <= 0 or n_embd % n_head:
    raise ValueError('n_layer >= 1; n_embd must be positive and divisible by a positive n_head')
head_dim = n_embd // n_head # derived dimension of each head
block_size = 14  # number of PAST positions supplied at each prediction
rollout_steps = 4       # predict this many future points inside EACH training loss
physics_weight = 0.01  # dimensionless; 0.0 disables physics, keeping all else the same

if block_size < 2 or rollout_steps < 3 or block_size + rollout_steps > len(train['time']):
    raise ValueError(f'Require block_size >= 2, rollout_steps >= 3, and their sum <= {len(train['time'])}')

model_rng = np.random.default_rng(42) # Let there be order among chaos
matrix = lambda nout, nin, std=0.08: jnp.array(model_rng.normal(0.0, std, size=(nout, nin)))
# wte projects input positions; lm_head predicts the change in position.
state_dict = {'wte': matrix(n_embd, n_channel), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(n_channel, n_embd)}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
params = state_dict # same dictionary, not a copy; JAX differentiates its arrays directly
print(f"num params: {sum(p.size for p in params.values())}")

# Define the model architecture: past positions -> next positions (dimensionless).
def linear(x, w):
    return x @ w.T

def softmax(logits):
    return jax.nn.softmax(logits, axis=-1)

def rmsnorm(x):
    ms = jnp.mean(x * x, axis=-1, keepdims=True)
    scale = (ms + 1e-5) ** -0.5
    return x * scale

@jax.jit
def gpt(state_dict, tokens):
    # tokens: (T, 2), containing dimensionless positions.
    pos_id = jnp.arange(len(tokens))
    tok_emb = linear(tokens, state_dict['wte']) # continuous input projection
    pos_emb = state_dict['wpe'][pos_id] # position within the current window
    x = tok_emb + pos_emb
    x = rmsnorm(x)

    for li in range(n_layer):
        # 1) Multi-head Attention block
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[:, hs:hs+head_dim]
            k_h = k[:, hs:hs+head_dim]
            v_h = v[:, hs:hs+head_dim]
            attn_logits = q_h @ k_h.T / head_dim**0.5
            # Causal mask: row t can see columns 0..t, never future positions.
            attn_logits = jnp.where(pos_id[:, None] >= pos_id[None, :], attn_logits, -jnp.inf)
            attn_weights = softmax(attn_logits)
            head_out = attn_weights @ v_h
            x_attn.append(head_out)
        x_attn = jnp.concatenate(x_attn, axis=-1)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = x + x_residual
        # 2) MLP block
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = jax.nn.relu(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = x + x_residual

    # A residual forecast: next position = current position + LEARNED change.
    # No g, analytic trajectory, or known initial velocity is used here.
    prediction = tokens + linear(x, state_dict['lm_head'])
    return prediction

def rollout(state_dict, tokens, steps):
    """Roll out several predictions, without inserting future observed/true positions."""
    sample = []
    for _ in range(steps):
        prediction = gpt(state_dict, tokens)[-1]
        sample.append(prediction)
        tokens = jnp.concatenate([tokens[1:], prediction[None, :]], axis=0)
    return jnp.stack(sample) # (steps, 2), all model-generated

def loss_fn(state_dict, tokens, targets, physics_weight):
    # (rollout_steps, 2), all model-generated
    prediction = rollout(state_dict, tokens, rollout_steps)
    # Supervised data loss: predict CLEAN positions from a NOISY past window.
    data_loss = jnp.mean((prediction - targets) ** 2)
    # Physical residual on predicted points ONLY; no noisy seed points mixed in.
    acceleration = (prediction[2:] - 2 * prediction[1:-1] + prediction[:-2]) * position_scale / dt**2
    physics_loss = jnp.mean(((acceleration - jnp.array([0.0, -g])) / g) ** 2)
    loss = data_loss + physics_weight * physics_loss
    return loss, (data_loss, physics_loss)

loss_and_grad = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

# Let there be Adam, the blessed optimizer and its buffers
learning_rate, beta1, beta2, eps_adam = 0.003, 0.85, 0.99, 1e-8
m = {i: jnp.zeros_like(p) for i, p in params.items()}
v = {i: jnp.zeros_like(p) for i, p in params.items()}

# Repeat in sequence
num_steps = 2000 # training updates, NOT epochs
# random training-window selection; separate from data generation
rng = np.random.default_rng(42)
order = np.arange(len(train['clean']))
rng.shuffle(order)
data_losses = []
for step in range(num_steps):

    # Take one throw and one random window; clean targets are strictly AFTER it.
    trajectory_idx = order[step % len(order)]
    start = int(rng.integers(0, len(train['time']) - block_size - rollout_steps + 1))
    tokens = jnp.array(train['observed'][trajectory_idx, start:start+block_size] / position_scale)
    targets = jnp.array(train['clean'][trajectory_idx, start+block_size:start+block_size+rollout_steps] / position_scale)

    # Differentiate through ALL rollout steps; no stop_gradient/detach.
    (loss, (data_loss, physics_loss)), grads = loss_and_grad(state_dict, tokens, targets, physics_weight)

    # Adam optimizer update: update the model parameters based on the corresponding gradients
    lr_t = learning_rate * (1 - step / num_steps)
    for i, p in params.items():
        m[i] = beta1 * m[i] + (1 - beta1) * grads[i]
        v[i] = beta2 * v[i] + (1 - beta2) * grads[i] ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        params[i] = p - lr_t * m_hat / (v_hat ** 0.5 + eps_adam)

    loss, data_loss, physics_loss = map(float, (loss, data_loss, physics_loss))
    data_losses.append(data_loss)
    if (step + 1) % 100 == 0 or step == 0:
        print(f"step {step+1:4d} / {num_steps:4d} | loss {loss:.6f} | data {data_loss:.6f} | physics {physics_loss:.6f}")

# Inference: forecast future positions for an unseen throw.
# Initial velocity for a NEW synthetic test case.
initial_velocity = np.array([6.0, 10.0]) # m/s; edit this to try another throw
if not (vx0_range[0] <= initial_velocity[0] <= vx0_range[1]
        and vy0_range[0] <= initial_velocity[1] <= vy0_range[1]):
    print('Warning: this initial velocity is outside the training range.')
time = train['time']
future_time = time[block_size:]
clean = np.stack([initial_velocity[0] * time, 1.0 + initial_velocity[1] * time - 0.5 * g * time**2], axis=-1)
observed = clean[:block_size] + np.random.default_rng(2026).normal(0.0, noise_std, size=(block_size, n_channel))
print(f'\n--- inference (only the first block_size (= {block_size}) observations are supplied, forecasting {len(future_time)} time steps) ---')
print(f"observed: {time[0]:.2f}..{time[block_size-1]:.2f} s; forecast: {time[block_size]:.2f}..{time[-1]:.2f} s")

# Roll out future positions from one throw's observed prefix.
tokens = jnp.array(observed / position_scale)
sample = rollout(state_dict, tokens, len(future_time))
prediction = np.array(sample) * position_scale # back to m
# Compute some metrics.
error = prediction - clean[block_size:]
acceleration = (prediction[2:] - 2 * prediction[1:-1] + prediction[:-2]) / dt**2
residual = acceleration - np.array([0.0, -g])
position_rmse = np.sqrt(np.mean(np.sum(error**2, axis=-1)))
final_position_rmse = np.linalg.norm(error[-1])
acceleration_rmse = np.sqrt(np.mean(np.sum(residual**2, axis=-1)))
print(f"estimated: position RMSE {position_rmse:.4f} m | final {final_position_rmse:.4f} m | acceleration residual {acceleration_rmse:.4f} m/s^2")

# Save figures for the training loss and estimated trajectory.
output_dir = f'micropit_results/physics_{physics_weight:g}'
os.makedirs(output_dir, exist_ok=True)

# 1) Training progress: unweighted data loss.
# A trailing moving mean reduces variation between the randomly selected windows.
width = min(100, len(data_losses))
averaged = np.convolve(data_losses, np.ones(width) / width, mode='valid')
fig, ax = plt.subplots(figsize=(8.4, 5.0), layout='constrained')
ax.semilogy(np.arange(width, num_steps + 1), averaged, marker='.' if len(averaged) == 1 else None,
            label=f'microPIT, physics_weight={physics_weight:g}')
ax.set(xlabel='Training update', ylabel='Dimensionless data loss',
       title=f'Training data loss ({width}-update moving mean)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.25)
fig.savefig(os.path.join(output_dir, 'training_loss.png'), dpi=180)
plt.close(fig)

# 2) Trajectory forecast: compare the model's prediction to the true trajectory and a reference least-squares fit.
# Known-g least-squares fit as a reference for the model forecast.
# Fit r0 and v0 from observed past positions only; no true velocity/future is used.
acceleration = np.array([0.0, -g])
design = np.stack([np.ones(block_size), time[:block_size]], axis=-1)
adjusted = observed - 0.5 * time[:block_size, None]**2 * acceleration
coefficients = np.linalg.lstsq(design, adjusted, rcond=None)[0]
future_design = np.stack([np.ones(len(future_time)), future_time], axis=-1)
fit = future_design @ coefficients + 0.5 * future_time[:, None]**2 * acceleration

fig, ax = plt.subplots(figsize=(8.4, 5.3), layout='constrained')
ax.plot(clean[:, 0], clean[:, 1], '--', label='True trajectory')
ax.scatter(observed[:, 0], observed[:, 1], s=24, label='Observed sequence (model input)')
ax.plot(prediction[:, 0], prediction[:, 1], 'o-', markersize=3,
        label=f'Forecast by microPIT, physics_weight={physics_weight:g}')
ax.plot(fit[:, 0], fit[:, 1], ':', label='Reference from Known-g least-squares fit')
vx0, vy0 = initial_velocity
ax.set(xlabel='x (m)', ylabel='y (m)',
       title=f'Initial velocity: ({vx0:g}, {vy0:g}) m/s\n'
             f'Observed: {time[0]:.2f}-{time[block_size-1]:.2f} s; '
             f'Forecast: {future_time[0]:.2f}-{future_time[-1]:.2f} s')
ax.set_aspect('equal', adjustable='datalim')
ax.legend(fontsize=9, loc='best')
ax.grid(True, alpha=0.25)
fig.savefig(os.path.join(output_dir, 'trajectory.png'), dpi=180)
plt.close(fig)

print(f"Saved {output_dir}/training_loss.png and {output_dir}/trajectory.png")
