"""
Mixed Normalizing Flows for Density Estimation on the Sphere (S²)
----------------------------------------------------------------------

This script implements a simple density estimator on the unit sphere using
a mixed-flow architecture inspired by Rezende et al.'s *Normalizing Flows
on Tori and Spheres*.  The key idea is to separately model the two
angular coordinates of the sphere:

* **Longitude (ϕ)** is periodic on the interval \([−π, π]\).  To
  capture its periodic nature we build a circular flow by first
  applying a phase shift and then passing the angle through a
  monotonic piece‑wise linear spline.  The spline is defined on the
  unit interval \([0, 1]\) with periodic boundary conditions.

* **Latitude (θ̃)** is bounded on \([−π/2, π/2]\) but not periodic.
  We model its distribution with a monotonic piece‑wise linear spline
  defined on \([0, 1]\).  Unlike the longitude, the latitude flow
  does not apply any circular shift.

Our flows map samples from a base distribution – the uniform
distribution on \([0,1]^2\) – to the data distribution expressed in
terms of normalised spherical coordinates.  The flow parameters (the
spline slopes and the longitude shift) are learned by maximising the
log‑likelihood of a synthetic dataset of points on the sphere.  A
Jacobian correction term \(\log(\sin\theta)\) is subtracted to
account for the change of variables between the flat coordinate
\((ϕ,θ)\) and the surface measure on the sphere \(\mathbb{S}^2\).

The implementation relies only on NumPy and SciPy and thus avoids
external dependencies such as PyTorch or Zuko.  Nevertheless, the
design closely follows Zuko's normalising‑flow abstractions and
demonstrates how separate transformations are needed for the periodic
and bounded directions on the sphere.

The script produces three outputs in the working directory:

* ``training_log.csv``: records the negative log‑likelihood (NLL)
  after each optimisation callback.
* ``heatmap_comparison.png``: a two‑panel figure comparing the
  density learned by the mixed flow with a kernel density estimate
  (KDE) of the data distribution on a Mollweide projection.
* ``analysis.txt``: a short explanation of why mixed flows are
  necessary on the sphere.

To execute the script simply run it with a recent version of Python.

"""

import csv
import math
import numpy as np
from typing import Tuple

import matplotlib
# Use Agg backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def generate_synthetic_data(n: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic dataset on S² consisting of two clusters.

    Each cluster is centred on the equator (θ̃≈0) but differs in
    longitude: one near ϕ=0 and the other near ϕ=π.  The latitudinal
    coordinate θ̃ is sampled from a narrow normal distribution and
    clipped to remain within the valid range.  Points are returned in
    normalised angular coordinates (ϕ_norm, θ_norm) such that
    ϕ_norm=(ϕ+π)/(2π)∈[0,1] and θ_norm=θ/π∈[0,1], where θ=θ̃+π/2.

    Args:
        n: total number of points to generate (must be even).
        seed: seed for the random number generator.

    Returns:
        A tuple ``(phi_norm, theta_norm)`` of shape ``(n,)``.
    """
    assert n % 2 == 0, "n must be even to allocate points equally between clusters"
    rng = np.random.default_rng(seed)

    # Longitude clusters: one around 0, one around π
    half = n // 2
    phi1 = rng.normal(loc=0.0, scale=0.3, size=half)
    phi2 = rng.normal(loc=math.pi, scale=0.3, size=half)
    phi = np.concatenate((phi1, phi2))
    # wrap into (−π, π]
    phi = (phi + math.pi) % (2 * math.pi) - math.pi

    # Latitude (θ̃) clusters: both around 0
    theta_tilde1 = rng.normal(loc=0.0, scale=0.2, size=half)
    theta_tilde2 = rng.normal(loc=0.0, scale=0.2, size=half)
    theta_tilde = np.concatenate((theta_tilde1, theta_tilde2))
    # clip to valid range to avoid sin(θ)=0 exactly
    eps = 1e-4
    theta_tilde = np.clip(theta_tilde, -math.pi / 2 + eps, math.pi / 2 - eps)
    # convert to polar angle θ = θ̃ + π/2
    theta = theta_tilde + math.pi / 2

    # normalised coordinates
    phi_norm = (phi + math.pi) / (2 * math.pi)
    theta_norm = theta / math.pi
    return phi_norm, theta_norm


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute a numerically stable softmax."""
    x = np.asarray(x)
    x_max = x.max()
    exps = np.exp(x - x_max)
    return exps / exps.sum()


def invert_piecewise(y: np.ndarray, slopes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Invert a monotonic piece‑wise linear spline.

    The spline maps ``u`` in ``[0,1]`` to ``y`` in ``[0,1]`` via
    ``y = sum_{j<i} (s_j / K) + (s_i / K) * t``, where ``t``
    parametrises the position within segment ``i`` and ``K`` is the
    number of segments.  This function performs the inverse mapping
    ``u = sum_{j<i} (1/K) + (1/K) * (y - cum_y[i]) / s_i`` and also
    returns the index of the segment used for each input.

    Args:
        y: array of shape ``(n,)`` with values in ``[0,1]``.
        slopes: array of shape ``(K,)`` giving segment slopes; must
            satisfy ``sum(slopes) / K == 1``.

    Returns:
        A tuple ``(u, idx)`` where ``u`` is the inverted coordinate and
        ``idx`` is the segment index per element.
    """
    K = len(slopes)
    # cumulative mass (y) at boundaries: length K+1
    cum = np.concatenate(([0.0], np.cumsum(slopes) / K))
    # identify the segment for each y
    # Find the segment for each y.  When y == 1 exactly the searchsorted
    # returns K which would be out of bounds, so clip to K-1.
    idx = np.searchsorted(cum[1:], y, side="right")
    idx = np.clip(idx, 0, K - 1)
    # slope and offset for each element
    s = slopes[idx]
    # start of segment in y
    y0 = cum[idx]
    # compute local coordinate t
    t = (y - y0) / (s / K)
    # invert mapping: u = (i + t) / K
    u = (idx + t) / K
    return u, idx


def forward_piecewise(u: np.ndarray, slopes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the forward monotonic piece‑wise linear spline.

    Given ``u`` in ``[0,1]`` and segment slopes ``slopes``, compute
    ``y`` such that for the segment ``i = floor(u*K)`` with local
    coordinate ``t = u*K - i``,
    ``y = sum_{j<i} (s_j / K) + (s_i / K) * t``.

    Args:
        u: array of shape ``(n,)`` with values in ``[0,1]``.
        slopes: array of shape ``(K,)`` giving segment slopes.

    Returns:
        A tuple ``(y, idx)`` where ``y`` is the transformed value and
        ``idx`` is the segment index per element.
    """
    K = len(slopes)
    # scale u by K to determine segment index and intra‑segment coordinate
    scaled = u * K
    # prevent values equal to K by clipping
    scaled = np.clip(scaled, 0.0, K - 1e-8)
    idx = scaled.astype(int)
    t = scaled - idx
    # cumulative mass (y) at segment boundaries
    cum = np.concatenate(([0.0], np.cumsum(slopes) / K))
    # start of segment in y
    y0 = cum[idx]
    s = slopes[idx]
    y = y0 + (s / K) * t
    return y, idx


class MixedFlow:
    """Mixed circular and bounded flow for modelling densities on the sphere.

    The flow maintains two sets of unnormalised slopes (for the
    longitude and latitude) and a shift parameter for the longitude.
    The slopes are transformed via a softmax to ensure positivity and
    normalisation.  The shift is passed through a sigmoid to stay in
    the unit interval.  Only the inverse transforms are required for
    likelihood evaluation; forward transforms are used for sampling.
    """

    def __init__(self, k_phi: int, k_theta: int):
        self.k_phi = k_phi
        self.k_theta = k_theta
        # initialise parameters to zeros for identity transform
        self.params = np.zeros(k_phi + 1 + k_theta)

    def _unpack_params(self, params: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        """Unpack the parameter vector into slopes and shift."""
        p_phi = params[: self.k_phi]
        shift_param = params[self.k_phi]
        p_theta = params[self.k_phi + 1 :]
        slopes_phi = softmax(p_phi) * self.k_phi
        slopes_theta = softmax(p_theta) * self.k_theta
        # map shift parameter to (0,1) via sigmoid
        shift = 1.0 / (1.0 + np.exp(-shift_param))
        return slopes_phi, shift, slopes_theta

    def inverse(self, phi_norm: np.ndarray, theta_norm: np.ndarray, params: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply the inverse transform to normalised data coordinates.

        Args:
            phi_norm: array of shape ``(n,)`` with values in ``[0,1]``.
            theta_norm: array of shape ``(n,)`` with values in ``[0,1]``.
            params: parameter vector containing unnormalised slopes and
                shift.

        Returns:
            A triple ``(u_phi, u_theta, logdet)`` where ``u_phi`` and
            ``u_theta`` are the base coordinates in ``[0,1]`` and
            ``logdet`` is an array of per‑sample log‑Jacobian
            determinants of the inverse transformation (i.e.\ 
            \(\log |\frac{du}{dx}|\)).
        """
        slopes_phi, shift, slopes_theta = self._unpack_params(params)
        # longitude: apply circular shift, invert spline, undo shift
        phi_shifted = (phi_norm + shift) % 1.0
        u_phi_shifted, idx_phi = invert_piecewise(phi_shifted, slopes_phi)
        u_phi = (u_phi_shifted - shift) % 1.0
        logdet_phi = -np.log(slopes_phi[idx_phi])
        # latitude: invert spline directly
        u_theta, idx_theta = invert_piecewise(theta_norm, slopes_theta)
        logdet_theta = -np.log(slopes_theta[idx_theta])
        logdet = logdet_phi + logdet_theta
        return u_phi, u_theta, logdet

    def forward(self, u_phi: np.ndarray, u_theta: np.ndarray, params: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply the forward transform to base coordinates.

        Args:
            u_phi: array of shape ``(n,)`` with values in ``[0,1]``.
            u_theta: array of shape ``(n,)`` with values in ``[0,1]``.
            params: parameter vector.

        Returns:
            A triple ``(phi_norm, theta_norm, logdet)`` where
            ``phi_norm`` and ``theta_norm`` are the transformed
            coordinates and ``logdet`` is the per‑sample log
            determinant of the forward transformation (\(\log |\frac{dx}{du}|\)).
        """
        slopes_phi, shift, slopes_theta = self._unpack_params(params)
        # longitude: apply shift, forward spline, undo shift
        u_phi_shifted = (u_phi + shift) % 1.0
        phi_shifted, idx_phi = forward_piecewise(u_phi_shifted, slopes_phi)
        phi_norm = (phi_shifted - shift) % 1.0
        logdet_phi = np.log(slopes_phi[idx_phi])
        # latitude: forward spline
        theta_norm, idx_theta = forward_piecewise(u_theta, slopes_theta)
        logdet_theta = np.log(slopes_theta[idx_theta])
        logdet = logdet_phi + logdet_theta
        return phi_norm, theta_norm, logdet


def train_flow(phi_norm: np.ndarray, theta_norm: np.ndarray, k_phi: int = 8, k_theta: int = 8, maxiter: int = 50) -> Tuple[MixedFlow, list]:
    """Train a mixed flow on the provided dataset.

    The parameters are optimised by minimising the negative
    log‑likelihood on the sphere using Powell's derivative‑free method.
    A callback records the NLL after each iteration.

    Args:
        phi_norm: normalised longitudes in ``[0,1]``.
        theta_norm: normalised polar angles in ``[0,1]``.
        k_phi: number of segments for the longitude spline.
        k_theta: number of segments for the latitude spline.
        maxiter: maximum number of optimisation iterations.

    Returns:
        The trained ``MixedFlow`` instance and a list of NLL values
        recorded during optimisation.
    """
    flow = MixedFlow(k_phi, k_theta)
    nll_history = []

    def objective(params: np.ndarray) -> float:
        """Compute the negative log‑likelihood for the current parameters."""
        # apply inverse transform to obtain base coordinates and log‑Jacobian
        u_phi, u_theta, logdet = flow.inverse(phi_norm, theta_norm, params)
        # base distribution is uniform on [0,1]^2 → log PDF = 0
        # compute log(sinθ) term: θ = π * theta_norm
        theta = theta_norm * math.pi
        logsin = np.log(np.sin(theta))
        # total log density on the sphere: log p = logdet − log(sinθ)
        logp = logdet - logsin
        # negative mean log likelihood
        return -np.mean(logp)

    def callback(params: np.ndarray) -> None:
        """Record the current NLL during optimisation."""
        nll_history.append(objective(params))

    # optimise using Powell's method – derivative‑free and robust for
    # small parameter counts; limit iterations for practicality
    result = minimize(
        objective,
        flow.params,
        method="Powell",
        callback=callback,
        options={"maxiter": maxiter, "disp": False},
    )
    flow.params = result.x
    return flow, nll_history


def evaluate_density(flow: MixedFlow, params: np.ndarray, phi: np.ndarray, theta_tilde: np.ndarray) -> np.ndarray:
    """Evaluate the learned density on a grid of angles.

    Args:
        flow: trained flow instance.
        params: parameter vector of the flow.
        phi: array of longitudes in radians in ``[−π, π]``.
        theta_tilde: array of shifted latitudes (θ̃) in radians in
            ``[−π/2, π/2]``.

    Returns:
        A 2D array of densities on the sphere corresponding to the
        meshgrid defined by ``phi`` and ``theta_tilde``.
    """
    # meshgrid for evaluation
    Phi, Theta_tilde = np.meshgrid(phi, theta_tilde)
    # convert to normalised coordinates
    phi_norm = (Phi + math.pi) / (2 * math.pi)
    # polar angle θ = θ̃ + π/2, then normalise
    theta_norm = (Theta_tilde + math.pi / 2) / math.pi
    # flatten for vectorised processing
    phi_norm_flat = phi_norm.ravel()
    theta_norm_flat = theta_norm.ravel()
    # apply inverse transformation
    _, _, logdet = flow.inverse(phi_norm_flat, theta_norm_flat, params)
    theta_vals = theta_norm_flat * math.pi
    # ensure sine is strictly positive to avoid log(0)
    sin_theta = np.sin(theta_vals)
    sin_theta[sin_theta <= 0] = 1e-8
    logsin = np.log(sin_theta)
    logp = logdet - logsin
    p = np.exp(logp)
    return p.reshape(phi_norm.shape)


def kernel_density_estimate(phi_data: np.ndarray, theta_tilde_data: np.ndarray, phi_grid: np.ndarray, theta_tilde_grid: np.ndarray, sigma_phi: float = 0.3, sigma_theta: float = 0.3) -> np.ndarray:
    """Compute an ad‑hoc kernel density estimate on the sphere.

    The KDE factorises into a periodic Gaussian in longitude and a
    Gaussian in latitude.  The periodic distance between longitudes
    accounts for wrap‑around at ±π.

    Args:
        phi_data: array of data longitudes in radians.
        theta_tilde_data: array of data latitudes in radians (shifted).
        phi_grid: 1D array of grid longitudes.
        theta_tilde_grid: 1D array of grid latitudes.
        sigma_phi: bandwidth for the longitude kernel.
        sigma_theta: bandwidth for the latitude kernel.

    Returns:
        A 2D array of KDE values on the meshgrid defined by the grid
        inputs.
    """
    # meshgrid for evaluation
    Phi, Theta_tilde = np.meshgrid(phi_grid, theta_tilde_grid)
    # flatten for convenience
    phi_flat = Phi.ravel()
    theta_flat = Theta_tilde.ravel()
    # compute periodic distances in longitude
    # broadcasting: for each grid point and data point
    diff_phi = phi_flat[:, None] - phi_data[None, :]
    # wrap difference to [-π, π]
    diff_phi = (diff_phi + math.pi) % (2 * math.pi) - math.pi
    # distances in latitude (θ̃)
    diff_theta = theta_flat[:, None] - theta_tilde_data[None, :]
    # Gaussian kernels
    kernel_phi = np.exp(-0.5 * (diff_phi / sigma_phi) ** 2)
    kernel_theta = np.exp(-0.5 * (diff_theta / sigma_theta) ** 2)
    weights = kernel_phi * kernel_theta
    kde = weights.mean(axis=1)
    # reshape to grid
    return kde.reshape(Phi.shape)


def save_training_log(nll_history: list, filename: str) -> None:
    """Save the negative log‑likelihood history to a CSV file."""
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["iteration", "nll"])
        for i, nll in enumerate(nll_history, start=1):
            writer.writerow([i, nll])


def create_heatmap_figure(flow: MixedFlow, params: np.ndarray, phi_data: np.ndarray, theta_tilde_data: np.ndarray, outfile: str) -> None:
    """Generate a two‑panel heatmap comparing flow and KDE densities."""
    # set up evaluation grid
    n_grid = 50
    eps = 1e-3  # avoid evaluating exactly at the poles
    phi_grid = np.linspace(-math.pi, math.pi, n_grid)
    theta_grid = np.linspace(-math.pi / 2 + eps, math.pi / 2 - eps, n_grid)
    # evaluate densities
    density_flow = evaluate_density(flow, params, phi_grid, theta_grid)
    density_kde = kernel_density_estimate(phi_data, theta_tilde_data, phi_grid, theta_grid)
    # normalise densities for visual comparability
    # replace NaNs or infs and normalise densities for visual comparability
    density_flow = np.nan_to_num(density_flow, nan=0.0, posinf=0.0, neginf=0.0)
    density_kde = np.nan_to_num(density_kde, nan=0.0, posinf=0.0, neginf=0.0)
    # normalise by maximum to obtain values in [0,1]
    flow_max = density_flow.max() if density_flow.max() > 0 else 1.0
    kde_max = density_kde.max() if density_kde.max() > 0 else 1.0
    density_flow_norm = density_flow / flow_max
    density_kde_norm = density_kde / kde_max

    # create figure with Mollweide projections
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={"projection": "mollweide"})
    titles = ["Mixed flow density", "KDE density"]
    for ax, density, title in zip(axes, [density_flow_norm, density_kde_norm], titles):
        # convert meshgrid to longitudes and latitudes in radians
        Phi_mesh, Theta_mesh = np.meshgrid(phi_grid, theta_grid)
        im = ax.pcolormesh(Phi_mesh, Theta_mesh, density, shading="auto", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("longitude ϕ")
        ax.set_ylabel("latitude θ̃")
        # grid lines for better readability
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
    fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.05, pad=0.04, label="normalised density")
    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close(fig)


def write_explanation(flow_file: str) -> None:
    """Write a short explanation of why mixed flows are necessary."""
    explanation = (
        "On the sphere the longitudinal and latitudinal coordinates have distinct geometries.\n"
        "Longitude (ϕ) wraps around, so its density must be periodic.  A flow that ignores this\n"
        "periodicity can accumulate probability mass at an arbitrary branch cut and miss the fact\n"
        "that ϕ=−π and ϕ=π represent the same physical direction.  By introducing a circular shift\n"
        "and a periodic spline we ensure that the transformation acts on a circle rather than a\n"
        "line segment.  Conversely, the latitude (θ̃) lies on a bounded interval and has no\n"
        "intrinsic wrap‑around.  Using a bounded monotonic spline allows the model to flexibly\n"
        "reshape the distribution within the allowed range without introducing artificial\n"
        "periodicity.  The resulting mixed flow therefore respects the topology of S²: one\n"
        "direction is treated as circular and the other as a finite segment."
    )
    with open(flow_file, "w") as f:
        f.write(explanation)


def main():
    # generate data
    n_samples = 2000
    phi_norm, theta_norm = generate_synthetic_data(n_samples, seed=42)
    # train flow
    flow, nll_history = train_flow(phi_norm, theta_norm, k_phi=8, k_theta=8, maxiter=50)
    # save training log
    save_training_log(nll_history, "training_log.csv")
    # convert data to radians for KDE evaluation
    phi_data = phi_norm * 2 * math.pi - math.pi
    theta_tilde_data = theta_norm * math.pi - math.pi / 2
    # generate heatmap comparison
    create_heatmap_figure(flow, flow.params, phi_data, theta_tilde_data, "heatmap_comparison.png")
    # write explanation
    write_explanation("analysis.txt")


if __name__ == "__main__":
    main()