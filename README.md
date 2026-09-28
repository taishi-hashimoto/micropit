# microPIT: micro Physics-Informed Transformer

Based on [microgpt by Andrej Karpathy](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95).

A minimal implementation of a physics-informed transformer for projectile motion prediction.

## Scripts

- [microgpt.py](microgpt.py): Original microgpt implementation by @karpathy.
- [microgpt_jax.py](microgpt_jax.py): JAX rewrite of `microgpt.py`.
- [micropit.py](micropit.py): An example implementation of a physics-informed transformer for projectile motion prediction.

## Disclaimer

- This is a toy example for my learning purposes only.
- The physics model here is a minimal implementation of the projectile motion with no air resistance nor ground collision. 

## Usage

```bash
# Optional: Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS

# Install dependencies: jax, numpy, and matplotlib
pip install -r requirements.txt

# Run the original microgpt implementation
python microgpt.py

# Run the JAX rewrite of microgpt
python microgpt_jax.py

# Run the physics-informed transformer example
python micropit.py
```

Modify `acceleration_weight` in `micropit.py` to see the effect of the acceleration loss, which encourages acceleration `(0, -g)`, on the predicted trajectory.

`velocity_weight` adds a supervised velocity loss using differences between consecutive predicted positions and consecutive clean target positions, divided by `dt`.
Velocities are normalized by `sqrt(g * position_scale)` before computing the MSE.
Set `velocity_weight = 0.0` and `acceleration_weight = 0.0` to disable phyisics.
Output folders include both weights so comparisons are saved separately.

The default training budget is 10,000 updates, configured by `num_steps` in `micropit.py`.

## Example Output

> NOTE: Values below may not be optimal.

| Physics OFF | Physics ON |
|-------------|------------|
| ![Physics OFF](docs/fig/trajectory_0.00.png) | ![Physics ON](docs/fig/trajectory_0.025.png) |
| acceleration_weight = 0.00<br>velocity_weight = 0.00 | acceleration_weight = 0.025<br>velocity_weight = 0.025 |

## License

Licensed under the [MIT License](LICENSE).
