"""Local-function families.

A local function maps the values of its ``k`` parents, an array of shape
``(n, k)``, to one value per row. A family object holds the family constants;
the parameters of one local function are a dictionary drawn from the target's
structure stream, so one family object serves every node of a target.

Both families are computed in float64.

Interface of a family:

``draw(rng, k)``            parameters of one local function
``calibrate(params, Z)``    set the parameters that depend on the parent distribution
``value(params, Z)``        function values, shape ``(n,)``
``grad(params, Z)``         partial derivatives with respect to the parents, shape ``(n, k)``
``describe()``              the constants that enter the target fingerprint
"""
from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi
ACCEPTANCE = "catastrophic_only"


def _leave_one_out_product(M: np.ndarray) -> np.ndarray:
    """For each column, the product of the other columns."""
    n, k = M.shape
    out = np.empty((n, k), dtype=M.dtype)
    for j in range(k):
        others = [i for i in range(k) if i != j]
        out[:, j] = M[:, others].prod(axis=1) if others else 1.0
    return out


class AdditiveMultiplicative:
    """The additive-multiplicative construction (paper Eq. 4).

        g(z) = sqrt((1 - rho) / k) * sum_i a_i(z_i) + sqrt(rho) * prod_i b_i(z_i)

    Each factor ``a_i`` and ``b_i`` is a sinusoid ``sin(pi * f * u + 2 * pi * phi)``
    with ``f ~ U[f_lo, f_hi]`` and ``phi ~ U[0, 1)``. The factor is centred and
    scaled to unit variance under the distribution of its parent, which is
    measured on the normalization pool of the target.
    """

    name = "additive_multiplicative"

    def __init__(self, rho: float, f_range):
        rho = float(rho)
        if not 0.0 <= rho <= 1.0:
            raise ValueError(f"rho must lie in [0, 1]; received {rho}")
        lo, hi = float(f_range[0]), float(f_range[1])
        if not 0.0 < lo <= hi:
            raise ValueError(f"f_range must satisfy 0 < lo <= hi; received {f_range}")
        self.rho = rho
        self.f_range = (lo, hi)

    def _weights(self, k: int) -> tuple[float, float]:
        return float(np.sqrt((1.0 - self.rho) / k)), float(np.sqrt(self.rho))

    def draw(self, rng: np.random.Generator, k: int) -> dict:
        if k < 2:
            raise ValueError(f"an additive-multiplicative function needs k >= 2; received {k}")
        lo, hi = self.f_range
        params = {}
        for bank in ("a", "b"):
            params[f"omega_{bank}"] = np.pi * rng.uniform(lo, hi, size=k)
            params[f"phi_{bank}"] = TWO_PI * rng.uniform(0.0, 1.0, size=k)
            params[f"mu_{bank}"] = np.full(k, np.nan)
            params[f"sd_{bank}"] = np.full(k, np.nan)
        return params

    def calibrate(self, params: dict, Z: np.ndarray) -> None:
        """Centre and scale each factor under the distribution of its parent."""
        for bank in ("a", "b"):
            raw = np.sin(params[f"omega_{bank}"] * Z + params[f"phi_{bank}"])
            mu = raw.mean(axis=0)
            sd = raw.std(axis=0)
            if float(sd.min()) < 1e-3:
                raise RuntimeError(f"degenerate factor: standard deviation {float(sd.min()):.3e} < 1e-3")
            params[f"mu_{bank}"] = mu
            params[f"sd_{bank}"] = sd

    def _factors(self, params: dict, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        A = (np.sin(params["omega_a"] * Z + params["phi_a"]) - params["mu_a"]) / params["sd_a"]
        B = (np.sin(params["omega_b"] * Z + params["phi_b"]) - params["mu_b"]) / params["sd_b"]
        return A, B

    def value(self, params: dict, Z: np.ndarray) -> np.ndarray:
        alpha, gamma = self._weights(Z.shape[1])
        A, B = self._factors(params, Z)
        return alpha * A.sum(axis=1) + gamma * B.prod(axis=1)

    def grad(self, params: dict, Z: np.ndarray) -> np.ndarray:
        alpha, gamma = self._weights(Z.shape[1])
        dA = params["omega_a"] * np.cos(params["omega_a"] * Z + params["phi_a"]) / params["sd_a"]
        dB = params["omega_b"] * np.cos(params["omega_b"] * Z + params["phi_b"]) / params["sd_b"]
        _, B = self._factors(params, Z)
        return alpha * dA + gamma * dB * _leave_one_out_product(B)

    def describe(self) -> dict:
        return {"rho": self.rho, "f_range": list(self.f_range), "variant": "sinusoidal_cycles"}


class RandomFourier:
    """Random Fourier functions (paper Eq. 5).

        g(z) = (1 / sqrt(Q)) * sum_q a_q * sin(omega_q . z + phi_q)

    with ``a_q ~ U[0.7, 1]``, ``phi_q ~ U[0, 2 pi)`` and
    ``omega_q ~ N(0, sigma_omega^2 I_k / k)``.
    """

    name = "random_fourier"

    def __init__(self, sigma_omega: float, Q: int = 8):
        sigma_omega = float(sigma_omega)
        if sigma_omega <= 0.0:
            raise ValueError(f"sigma_omega must be positive; received {sigma_omega}")
        Q = int(Q)
        if Q < 1:
            raise ValueError(f"Q must be a positive integer; received {Q}")
        self.sigma_omega = sigma_omega
        self.Q = Q

    def draw(self, rng: np.random.Generator, k: int) -> dict:
        return {
            "amp": rng.uniform(0.7, 1.0, size=self.Q),
            "omega": rng.normal(0.0, self.sigma_omega / np.sqrt(k), size=(self.Q, k)),
            "phi": rng.uniform(0.0, TWO_PI, size=(self.Q,)),
        }

    def calibrate(self, params: dict, Z: np.ndarray) -> None:
        return None

    def value(self, params: dict, Z: np.ndarray) -> np.ndarray:
        angles = Z @ params["omega"].T + params["phi"]
        return (params["amp"] * np.sin(angles)).sum(axis=1) / np.sqrt(self.Q)

    def grad(self, params: dict, Z: np.ndarray) -> np.ndarray:
        angles = Z @ params["omega"].T + params["phi"]
        return ((params["amp"] * np.cos(angles)) @ params["omega"]) / np.sqrt(self.Q)

    def describe(self) -> dict:
        return {"sigma_omega": self.sigma_omega, "Q": self.Q, "acceptance": ACCEPTANCE}
