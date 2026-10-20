"""Node laws (target-design.md §4): anova plain/shaped, fourier, relu, tri.

A law maps one node's parent values (n, k) to node values (n,). A law instance
holds the family-level knobs (rho, sigma_shape, sigma_omega, ...); per-node
coefficients live in a params dict drawn from the structural stream, so one law
object serves every node of a target.

Law contract:
  draw(rng, k) -> params          per-node coefficients
  value(params, Z) -> (n,)        node values, float64
  grad(params, Z) -> (n, k)       partial derivatives w.r.t. the parents
  calibrate(params, Z, rng)       pool hook (shape standardization, median bias)
  activation_fraction(params, h)  band statistic, or None when not applicable
  standardize_nodes               True -> node outputs are pool-standardized (§7 mode 3)
  needs_pool                      True -> drawing requires the calibration pool
  scale_leaves                    True -> uniform_pm1 leaves enter as sqrt(3) * x
  exact                           True -> node moments are exact by construction (§7 mode 1)
"""
from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi
FOURIER_Q = 8       # §4.2: fixed across the k sweep so rank(W) <= Q never binds

ACCEPTANCE_CATASTROPHIC = "catastrophic_only"


def _acceptance_is_catastrophic(cfg) -> bool:
    """Read the cell's target-acceptance rule.

    ``acceptance: catastrophic_only`` is the v0.5 rule: sequential
    normalization handles ordinary scale variation, so a draw is redrawn only
    when its raw calibration-pool standard deviation falls below the
    catastrophic floor. Absent, the legacy rule applies -- which is what keeps
    historical Fourier configurations reproducing their frozen fingerprints.
    """
    value = cfg.get("acceptance")
    if value is None:
        return False
    if value != ACCEPTANCE_CATASTROPHIC:
        raise ValueError(
            f"unknown acceptance rule {value!r}; expected "
            f"{ACCEPTANCE_CATASTROPHIC!r} or no value for the legacy rule")
    return True
DEFAULT_FREQUENCIES = (1,)


def _leave_one_out_product(M: np.ndarray) -> np.ndarray:
    """Per-column product of the remaining columns of ``M``."""
    n, k = M.shape
    out = np.empty((n, k), dtype=M.dtype)
    for j in range(k):
        others = [i for i in range(k) if i != j]
        out[:, j] = M[:, others].prod(axis=1) if others else 1.0
    return out


class AnovaLaw:
    """§4.1 default law: g = sqrt((1-rho)/k) sum_i a_i(z_i) + sqrt(rho) prod_i b_i(z_i).

    plain (sigma_shape = 0): a_i(u) = s_i u, b_i(u) = s'_i u -- multilinear,
    exact; the mechanistic/analytic primary.
    shaped (sigma_shape > 0): a_i, b_i = sin(omega u + phi), standardized on the
    calibration pool -- audited-MC approximation of the population construction;
    the empirical primary at the frozen setting sigma_shape = 1 (a setting, not
    a difficulty axis -- §4.1).
    sinusoidal-cycles (f_range set; CSFRBench v0.5): a_i, b_i =
    sin(2*pi*(f*u/2 + phi)) with f ~ U[f_lo, f_hi] in cycles per length-2
    interval of the parent axis and phi ~ U[0, 1), centered and standardized
    under the actual (induced) parent distribution on the calibration pool.
    Node outputs are additionally pool-standardized (sequential normalization),
    and the sd < 0.1 redraw is replaced by the catastrophic floor only.
    """

    name = "anova"
    standardize_nodes = False
    catastrophic_only = False
    scale_leaves = True

    def __init__(self, rho: float, sigma_shape: float = 0.0, f_range=None):
        rho = float(rho)
        sigma_shape = float(sigma_shape)
        if not 0.0 <= rho <= 1.0:
            raise ValueError(f"anova rho must lie in [0, 1]; received {rho}")
        if sigma_shape < 0.0:
            raise ValueError(f"anova sigma_shape must be nonnegative; received {sigma_shape}")
        if f_range is not None:
            if sigma_shape != 0.0:
                raise ValueError("f_range and sigma_shape are mutually exclusive")
            lo, hi = (float(f_range[0]), float(f_range[1]))
            if not 0.0 < lo <= hi:
                raise ValueError(f"f_range must satisfy 0 < lo <= hi; received {f_range}")
            f_range = (lo, hi)
            self.standardize_nodes = True       # sequential normalization (v0.5 §7)
            self.catastrophic_only = True       # no sd < 0.1 redraw; 1e-3 floor only
        self.rho = rho
        self.sigma_shape = sigma_shape
        self.f_range = f_range

    @property
    def sinusoidal(self) -> bool:
        return self.f_range is not None

    @property
    def plain(self) -> bool:
        return self.sigma_shape == 0.0 and not self.sinusoidal

    @property
    def exact(self) -> bool:
        return self.plain

    @property
    def needs_pool(self) -> bool:
        return not self.plain

    def _weights(self, k: int) -> tuple[float, float]:
        return float(np.sqrt((1.0 - self.rho) / k)), float(np.sqrt(self.rho))

    def draw(self, rng: np.random.Generator, k: int) -> dict:
        if k < 2:
            raise ValueError(f"anova nodes require k >= 2; received {k}")
        if self.plain:
            return {
                "sign_a": rng.choice((-1.0, 1.0), size=k),
                "sign_b": rng.choice((-1.0, 1.0), size=k),
            }
        params = {}
        for bank in ("a", "b"):
            if self.sinusoidal:
                lo, hi = self.f_range
                # sin(2*pi*(f*u/2 + phi)) = sin(pi*f*u + 2*pi*phi): store the
                # angular coefficient pi*f and the angular phase, so value/grad
                # share one code path with the legacy shaped variant.
                params[f"omega_{bank}"] = np.pi * rng.uniform(lo, hi, size=k)
                params[f"phi_{bank}"] = TWO_PI * rng.uniform(0.0, 1.0, size=k)
            else:
                params[f"omega_{bank}"] = self.sigma_shape * rng.uniform(0.75, 1.25, size=k)
                params[f"phi_{bank}"] = rng.uniform(0.0, TWO_PI, size=k)
            params[f"mu_{bank}"] = np.full(k, np.nan)
            params[f"sd_{bank}"] = np.full(k, np.nan)
        return params

    def calibrate(self, params: dict, Z: np.ndarray, rng: np.random.Generator) -> None:
        """Standardize each shape under its parent's pool marginal (shaped only)."""
        if self.plain:
            return
        for bank in ("a", "b"):
            raw = np.sin(params[f"omega_{bank}"] * Z + params[f"phi_{bank}"])
            mu = raw.mean(axis=0)
            sd = raw.std(axis=0)
            if float(sd.min()) < 1e-3:
                raise RuntimeError(
                    f"degenerate shape: pool sd {float(sd.min()):.3e} < 1e-3"
                )
            params[f"mu_{bank}"] = mu
            params[f"sd_{bank}"] = sd

    def _banks(self, params: dict, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.plain:
            return params["sign_a"] * Z, params["sign_b"] * Z
        A = (np.sin(params["omega_a"] * Z + params["phi_a"]) - params["mu_a"]) / params["sd_a"]
        B = (np.sin(params["omega_b"] * Z + params["phi_b"]) - params["mu_b"]) / params["sd_b"]
        return A, B

    def value(self, params: dict, Z: np.ndarray) -> np.ndarray:
        alpha, gamma = self._weights(Z.shape[1])
        A, B = self._banks(params, Z)
        return alpha * A.sum(axis=1) + gamma * B.prod(axis=1)

    def grad(self, params: dict, Z: np.ndarray) -> np.ndarray:
        alpha, gamma = self._weights(Z.shape[1])
        if self.plain:
            dA = np.broadcast_to(params["sign_a"], Z.shape)
            dB = np.broadcast_to(params["sign_b"], Z.shape)
            _, B = self._banks(params, Z)
        else:
            dA = params["omega_a"] * np.cos(params["omega_a"] * Z + params["phi_a"]) / params["sd_a"]
            dB = params["omega_b"] * np.cos(params["omega_b"] * Z + params["phi_b"]) / params["sd_b"]
            _, B = self._banks(params, Z)
        return alpha * dA + gamma * dB * _leave_one_out_product(B)

    def activation_fraction(self, params: dict, h: np.ndarray) -> None:
        return None

    def describe(self) -> dict:
        if self.sinusoidal:
            return {"rho": self.rho, "f_range": list(self.f_range),
                    "variant": "sinusoidal_cycles"}
        return {"rho": self.rho, "sigma_shape": self.sigma_shape,
                "variant": "plain" if self.plain else "shaped"}


def anova_law_from_cell(cell) -> AnovaLaw:
    """The anova law at a cell's (rho, sigma_shape) knobs (§10 defaults)."""
    return AnovaLaw(float(cell.get("rho", 0.5)), float(cell.get("sigma_shape", 0.0)))


def functional_anova_law_from_cell(cell) -> AnovaLaw:
    """The v0.5 functional-ANOVA law with sinusoidal factors: (rho, f_range)."""
    f_range = cell.get("f_range")
    if f_range is None:
        raise ValueError(
            "functional-ANOVA sinusoidal families require an explicit f_range "
            "(cycles per length-2 parent interval), e.g. f_range: [0.25, 0.40]"
        )
    return AnovaLaw(float(cell.get("rho", 0.5)), 0.0, f_range=tuple(f_range))


class FourierLaw:
    """§4.2 validation law: Q = 8 random-Fourier-feature teacher, radial omega scale.

    Generic nonseparable multivariate surfaces with realized intrinsic
    dimensionality audited (fourier_rank + intrinsic_dimension). Frozen scope:
    the tree_fourier family holds k = 2; an arity study must recalibrate Q
    jointly with k (rank(Omega) <= Q, and Q moves realized guidance).
    """

    name = "fourier"
    standardize_nodes = True
    catastrophic_only = False
    scale_leaves = True
    needs_pool = True
    exact = False

    def __init__(self, sigma_omega: float = 0.85, Q: int | None = None,
                 catastrophic_only: bool = False):
        sigma_omega = float(sigma_omega)
        if sigma_omega <= 0.0:
            raise ValueError(f"fourier sigma_omega must be positive; received {sigma_omega}")
        self.sigma_omega = sigma_omega
        Q = FOURIER_Q if Q is None else int(Q)
        if Q < 1:
            raise ValueError(f"fourier Q must be a positive integer; received {Q}")
        self.Q = Q
        # v0.5 acceptance: sequential normalization already removes ordinary
        # scale variation, so only the catastrophic floor redraws. Legacy
        # configurations leave this False and keep the sd < 0.1 rule, which
        # holds their fingerprints fixed.
        self.catastrophic_only = bool(catastrophic_only)

    def draw(self, rng: np.random.Generator, k: int) -> dict:
        return {
            "amp": rng.uniform(0.7, 1.0, size=self.Q),
            "omega": rng.normal(0.0, self.sigma_omega / np.sqrt(k), size=(self.Q, k)),
            "phi": rng.uniform(0.0, TWO_PI, size=(self.Q,)),
        }

    def calibrate(self, params: dict, Z: np.ndarray, rng: np.random.Generator) -> None:
        return None

    def value(self, params: dict, Z: np.ndarray) -> np.ndarray:
        angles = Z @ params["omega"].T + params["phi"]
        return (params["amp"] * np.sin(angles)).sum(axis=1) / np.sqrt(self.Q)

    def grad(self, params: dict, Z: np.ndarray) -> np.ndarray:
        angles = Z @ params["omega"].T + params["phi"]
        return ((params["amp"] * np.cos(angles)) @ params["omega"]) / np.sqrt(self.Q)

    def activation_fraction(self, params: dict, h: np.ndarray) -> None:
        return None

    def describe(self) -> dict:
        out = {"sigma_omega": self.sigma_omega, "Q": self.Q}
        # The acceptance rule is part of target identity: the same cell under
        # the two rules can yield different functions, so the fingerprint has
        # to separate them. Emitted only in the non-default case, which leaves
        # every legacy fingerprint exactly as it was.
        if self.catastrophic_only:
            out["acceptance"] = ACCEPTANCE_CATASTROPHIC
        return out


class ReluLaw:
    """§4.3 diagnostic law: signed ReLU ridge, median-centered, band-calibrated."""

    name = "relu"
    standardize_nodes = True
    scale_leaves = True
    needs_pool = True
    exact = False

    def draw(self, rng: np.random.Generator, k: int) -> dict:
        signs = rng.choice((-1.0, 1.0), size=k)
        magnitudes = rng.uniform(0.5, 1.5, size=k)
        return {"a": signs * magnitudes / np.sqrt(k), "b": np.float64(0.0)}

    def calibrate(self, params: dict, Z: np.ndarray, rng: np.random.Generator) -> None:
        """Center the preactivation on its pool median."""
        params["b"] = np.float64(-np.median(Z @ params["a"]))

    def value(self, params: dict, Z: np.ndarray) -> np.ndarray:
        return np.maximum(Z @ params["a"] + params["b"], 0.0)

    def grad(self, params: dict, Z: np.ndarray) -> np.ndarray:
        active = (Z @ params["a"] + params["b"]) > 0.0
        return active[:, None] * np.broadcast_to(params["a"], Z.shape)

    def activation_fraction(self, params: dict, h: np.ndarray) -> float:
        return float((h > 0.0).mean())

    def describe(self) -> dict:
        return {"beta": 1, "ridge": True}


def triangle_wave(t, frequency):
    """Amplitude-1 triangle wave completing ``frequency`` periods per unit-2 interval."""
    t = np.asarray(t, dtype=np.float64)
    return 1.0 - np.abs(np.mod(2.0 * np.asarray(frequency) * t + 1.0, 4.0) - 2.0)


def triangle_slope(t, frequency):
    """Almost-everywhere derivative of ``triangle_wave`` w.r.t. its argument."""
    t = np.asarray(t, dtype=np.float64)
    f = np.asarray(frequency, dtype=np.float64)
    return -np.sign(np.mod(2.0 * f * t + 1.0, 4.0) - 2.0) * 2.0 * f


def validate_frequencies(frequencies) -> tuple[int, ...]:
    """Check node frequencies against the measure-preservation requirement."""
    values = list(frequencies)
    if not values:
        raise ValueError("tri frequencies must not be empty")
    result = []
    for value in values:
        # One parent sweeps length 2; a whole number of periods over it keeps z_v uniform.
        if float(value) != int(value) or int(value) < 1:
            raise ValueError(
                f"tri frequencies must be positive integers; received {value!r}"
            )
        result.append(int(value))
    return tuple(result)


class TriLaw:
    """§4.4 stress law: triangle wave of a signed parent sum -- exactly U[-1, 1] nodes."""

    name = "tri"
    standardize_nodes = False
    scale_leaves = False       # tri nodes consume U[-1, 1] parents directly
    needs_pool = False
    exact = True

    def __init__(self, frequencies=DEFAULT_FREQUENCIES, random_signs=True, random_phases=True):
        self.frequencies = validate_frequencies(frequencies)
        self.random_signs = bool(random_signs)
        self.random_phases = bool(random_phases)

    def draw(self, rng: np.random.Generator, k: int) -> dict:
        return {
            "sigma": (rng.choice((-1.0, 1.0), size=k) if self.random_signs
                      else np.ones(k)),
            "phi": (np.float64(rng.uniform(-1.0, 1.0)) if self.random_phases
                    else np.float64(0.0)),
            "f": np.int64(rng.choice(self.frequencies)),
        }

    def calibrate(self, params: dict, Z: np.ndarray, rng: np.random.Generator) -> None:
        return None

    def _argument(self, params: dict, Z: np.ndarray) -> np.ndarray:
        return Z @ params["sigma"] + params["phi"]

    def value(self, params: dict, Z: np.ndarray) -> np.ndarray:
        return triangle_wave(self._argument(params, Z), params["f"])

    def grad(self, params: dict, Z: np.ndarray) -> np.ndarray:
        slope = triangle_slope(self._argument(params, Z), params["f"])
        return slope[:, None] * np.broadcast_to(params["sigma"], Z.shape)

    def activation_fraction(self, params: dict, h: np.ndarray) -> None:
        return None

    def describe(self) -> dict:
        return {"beta": 1, "frequencies": list(self.frequencies), "ridge": True}
