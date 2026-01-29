# Mixed Flow Density Estimator on S²

This repository demonstrates density estimation on the sphere using cross‑coupled piecewise linear flows implemented in pure NumPy/SciPy.

## Overview
We revisit the normalizing flow architecture for spherical data and introduce a more expressive model that can capture multimodal distributions along longitude. The key ideas are:
- **Conditional splines** for longitude and latitude whose parameters depend on the other coordinate.
- A **circular shift** for the periodic longitude to ensure invertibility.
- **Stacking multiple layers** to model multimodal densities.

The script `mixed_sphere_flow.py` generates a synthetic dataset with two clusters on the equator, trains the cross‑coupled flow via maximum likelihood and visualises the learned density against an empirical kernel density estimate.

## Files
- **mixed_sphere_flow.py** – main training script implementing the cross‑coupled flow.
- **analysis.txt** – a short explanation of why cross‑coupling is necessary.
- **training_log.csv** – negative log‑likelihood per optimisation iteration.
- **heatmap_comparison.png** – Mollweide projection comparing the learned density to a KDE.

## Requirements
- Python 3.8+
- NumPy
- SciPy
- Matplotlib

Install the dependencies with:

```bash
pip install numpy scipy matplotlib
```

## Usage
Run the training script from the repository root:

```bash
python mixed_sphere_flow.py
```

This will write `training_log.csv`, `heatmap_comparison.png` and `analysis.txt` to the working directory. The heatmap should show two modes near longitude 0 and π along the equator when the model has converged.
