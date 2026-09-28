# microPIT: micro Physics-Informed Transformer

Based on microgpt_jax.py; original scalar microgpt: @karpathy.

A minimal implementation of a physics-informed transformer for projectile motion prediction.

## Scripts

- [microgpt.py](microgpt.py): Original microgpt implementation by @karpathy.
- [microgpt_jax.py](microgpt_jax.py): JAX rewrite of `microgpt.py`.
- [micropit.py](micropit.py): An example implementation of a physics-informed transformer for projectile motion prediction.

## Usage

```bash
# Optional: Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS

# Install dependencies: jax, numpy, and matplotlib
pip install -r requirements.txt

# Run the physics-informed transformer
python micropit.py
```

## Example Output

Modify `physics_weight` in `micropit.py` to see the effect of physics-informed loss on the predicted trajectory.

> NOTE: Value below may not be the optimal setting.

| Physics OFF<br>(physics_weight = 0.00) | Physics ON<br>(physics_weight = 0.025) |
|-------------|------------|
| ![Physics OFF](docs/fig/trajectory_0.00.png) | ![Physics ON](docs/fig/trajectory_0.025.png) |