"""Task registry and corpus collector for the public stable-worldmodel suite.

Corpora are collected by driving ``reset``/``step`` directly -- never through
``World.collect()``, whose 0.1.1 implementation drops the reset observation and
rotates the action column (upstream issue #293, fixed by PR #294). The audit in
``audit_data.py`` verifies the alignment property that matters here directly on
the stored arrays, so the claim does not rest on reading upstream code.

Stored schema, one NPZ per task:
  obs      float32 [sum_e L_e, obs_dim]   raw states, episodes concatenated
  act      float32 [sum_e (L_e-1), act_dim]
  ep_ptr   int64   [n_eps+1]              row offsets into obs
  act_ptr  int64   [n_eps+1]              row offsets into act
plus <task>.meta.json (dims, seeds, policy, per-episode records, sha256).

Exploration policy (one generic rule for every task, frozen in the contract):
an AR(1) walk inside the action box,
    a_t = clip(rho * a_{t-1} + sqrt(1 - rho^2) * u_t),   u_t ~ Uniform(box),
with the per-episode correlation ``rho`` cycling through (0.0, 0.5, 0.9) by
episode index. No task-specific policy, no expert, no reward shaping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
DATA = STUDY / "data"
SEED0 = 20260831

RHOS = (0.0, 0.5, 0.9)

#: One collection geometry for every task (frozen): short episodes, many of
#: them. 1,200 episodes x 100 steps = 120,000 transitions from 1,200 distinct
#: initial conditions, which keeps small training budgets from being drawn out
#: of a handful of neighbouring frames. h_max for rollout evaluation is 32, so
#: every episode is long enough to host a full evaluation window.
COLLECT_HORIZON = 100
COLLECT_EPISODES = 1200


def derive(*parts) -> int:
    joined = "|".join(str(p) for p in parts)
    return int.from_bytes(hashlib.blake2b(joined.encode(), digest_size=4).digest(), "big")


# ---------------------------------------------------------------- registry

# task key -> (env id, extra gym.make kwargs, episode horizon, feature spec)
# feature spec: angles = raw indices expanded to (cos, sin); quats = 4-blocks
# renormalized every rollout step; circles = (i, i+1) pairs already on the unit
# circle and renormalized. Everything else passes through untouched.
TASKS: dict[str, dict] = {
    "pusht": {"env_id": "swm/PushT-v1", "kwargs": {}, "horizon": 200,
              "angles": (4,), "quats": (), "circles": ()},
    "ogbcube": {"env_id": "swm/OGBCube-v0",
                "kwargs": {"mode": "data_collection", "terminate_at_goal": False},
                "horizon": 200, "angles": (), "quats": (22,),
                "circles": ((15, 16), (26, 27))},
    "ogbscene": {"env_id": "swm/OGBScene-v0",
                 "kwargs": {"mode": "data_collection", "terminate_at_goal": False},
                 "horizon": 200, "angles": (), "quats": (), "circles": ()},
    "ogbmaze": {"env_id": "swm/OGBMaze-v0", "kwargs": {},
                "horizon": 200, "angles": (), "quats": (), "circles": ()},
    "acrobot": {"env_id": "swm/AcrobotDMControl-v0", "kwargs": {}, "horizon": 250,
                "angles": (), "quats": (), "circles": ()},
    "humanoid": {"env_id": "swm/HumanoidDMControl-v0", "kwargs": {}, "horizon": 250,
                 "angles": (), "quats": (), "circles": ()},
    "walker": {"env_id": "swm/WalkerDMControl-v0", "kwargs": {}, "horizon": 250,
               "angles": (), "quats": (), "circles": ()},
    "cheetah": {"env_id": "swm/CheetahDMControl-v0", "kwargs": {}, "horizon": 250,
                "angles": (), "quats": (), "circles": ()},
    "hopper": {"env_id": "swm/HopperDMControl-v0", "kwargs": {}, "horizon": 250,
               "angles": (), "quats": (), "circles": ()},
    "quadruped": {"env_id": "swm/QuadrupedDMControl-v0", "kwargs": {}, "horizon": 250,
                  "angles": (), "quats": (), "circles": ()},
    "finger": {"env_id": "swm/FingerDMControl-v0", "kwargs": {}, "horizon": 250,
               "angles": (), "quats": (), "circles": ()},
    "reacher": {"env_id": "swm/ReacherDMControl-v0", "kwargs": {}, "horizon": 250,
                "angles": (), "quats": (), "circles": ()},
    "cartpole": {"env_id": "swm/CartpoleDMControl-v0", "kwargs": {}, "horizon": 250,
                 "angles": (), "quats": (), "circles": ()},
    "pendulum": {"env_id": "swm/PendulumDMControl-v0", "kwargs": {}, "horizon": 250,
                 "angles": (), "quats": (), "circles": ()},
    "ballincup": {"env_id": "swm/BallInCupDMControl-v0", "kwargs": {}, "horizon": 250,
                  "angles": (), "quats": (), "circles": ()},
    "manipulator": {"env_id": "swm/ManipulatorDMControl-v0", "kwargs": {},
                    "horizon": 250, "angles": (), "quats": (), "circles": ()},
    "tworoom": {"env_id": "swm/TwoRoom-v1", "kwargs": {}, "horizon": 100,
                "angles": (), "quats": (), "circles": ()},
    "pointmaze": {"env_id": "swm/SimplePointMaze-v0", "kwargs": {}, "horizon": 100,
                  "angles": (), "quats": (), "circles": ()},
    "piecewise": {"env_id": "swm/Piecewise-v0", "kwargs": {}, "horizon": 100,
                  "angles": (), "quats": (), "circles": ()},
    "rocket": {"env_id": "swm/PFRocketLanding-v0", "kwargs": {}, "horizon": 200,
               "angles": (), "quats": (), "circles": ()},
    "pendulumctl": {"env_id": "swm/PendulumControl-v1", "kwargs": {}, "horizon": 200,
                    "angles": (), "quats": (), "circles": ()},
    "mountaincar": {"env_id": "swm/MountainCarContinuousControl-v0", "kwargs": {},
                    "horizon": 200, "angles": (), "quats": (), "circles": ()},
}


def spec(task: str) -> dict:
    return TASKS[task]


# ---------------------------------------------------------------- features

def _feature_blocks(task: str, d: int):
    s = spec(task)
    angles, quats = set(s["angles"]), set(s["quats"])
    circles = {c[0] for c in s["circles"]}
    i = 0
    while i < d:
        if i in angles:
            yield ("angle", i, 1, 2); i += 1
        elif i in quats:
            yield ("quat", i, 4, 4); i += 4
        elif i in circles:
            yield ("circle", i, 2, 2); i += 2
        else:
            yield ("pass", i, 1, 1); i += 1


def featurize(task: str, states: np.ndarray) -> np.ndarray:
    cols = []
    for kind, start, raw_len, _ in _feature_blocks(task, states.shape[-1]):
        if kind == "angle":
            cols.append(np.cos(states[..., start:start + 1]))
            cols.append(np.sin(states[..., start:start + 1]))
        else:
            cols.append(states[..., start:start + raw_len])
    return np.concatenate(cols, axis=-1).astype(np.float64)


def feature_projectors(task: str, d: int):
    groups, offset = [], 0
    for kind, _, _, feat_len in _feature_blocks(task, d):
        if kind in ("angle", "circle"):
            groups.append(("circle", (offset, offset + 1)))
        elif kind == "quat":
            groups.append(("quat", tuple(range(offset, offset + 4))))
        offset += feat_len
    return groups


def project_features_(feats: np.ndarray, groups) -> None:
    for _kind, idx in groups:
        cols = list(idx)
        block = feats[..., cols]
        norm = np.maximum(np.sqrt((block ** 2).sum(-1, keepdims=True)), 1e-8)
        feats[..., cols] = block / norm


def feature_names(task: str, d: int) -> list[str]:
    names = []
    for kind, start, raw_len, _ in _feature_blocks(task, d):
        if kind == "angle":
            names += [f"cos s{start}", f"sin s{start}"]
        elif kind == "quat":
            names += [f"quat{start}[{k}]" for k in range(4)]
        elif kind == "circle":
            names += [f"circ s{start}", f"circ s{start + 1}"]
        else:
            names.append(f"s{start}")
    return names


# ---------------------------------------------------------------- collection

def state_of(obs) -> np.ndarray:
    if isinstance(obs, dict):
        return np.asarray(obs["state"], dtype=np.float64).ravel()
    return np.asarray(obs, dtype=np.float64).ravel()


class ARPolicy:
    """AR(1) walk inside the action box; the study's single generic explorer."""

    def __init__(self, low, high, rho, rng):
        self.low, self.high = np.asarray(low, float), np.asarray(high, float)
        self.rho, self.rng = float(rho), rng
        self.a = self._draw()

    def _draw(self):
        return self.rng.uniform(self.low, self.high)

    def __call__(self):
        mid, half = (self.high + self.low) / 2, (self.high - self.low) / 2
        u = (self._draw() - mid) / np.maximum(half, 1e-12)
        cur = (self.a - mid) / np.maximum(half, 1e-12)
        nxt = self.rho * cur + np.sqrt(max(1.0 - self.rho ** 2, 0.0)) * u
        self.a = np.clip(mid + nxt * half, self.low, self.high)
        return self.a


def coerce_variation(env, values: dict) -> dict:
    """Cast requested variation values to each leaf space's dtype and shape."""
    space = env.unwrapped.variation_space
    out = {}
    for key, value in values.items():
        leaf = space
        for part in key.split("."):
            leaf = leaf[part]
        out[key] = (int(value) if hasattr(leaf, "n")
                    else np.asarray(value, dtype=leaf.dtype).reshape(leaf.shape))
    return out


def collect(task: str, n_eps: int, horizon: int | None = None,
            out_name: str | None = None, variation: dict | None = None) -> Path:
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    s = spec(task)
    name = out_name or task
    out_npz = DATA / f"{name}.npz"
    if out_npz.exists():
        print(f"[collect] {out_npz} exists, skipping")
        return out_npz
    horizon = horizon or COLLECT_HORIZON
    env = gym.make(s["env_id"], **s["kwargs"])
    low = np.asarray(env.action_space.low, float).ravel()
    high = np.asarray(env.action_space.high, float).ravel()
    dtype = env.action_space.dtype

    obs_rows, act_rows, ep_ptr, act_ptr, ep_meta = [], [], [0], [0], []
    t0 = time.time()
    try:
        for e in range(n_eps):
            seed = derive(SEED0, "swm", task, "collect", e)
            rng = np.random.default_rng(seed)
            if "OGB" in s["env_id"]:
                np.random.seed(seed)   # OGB task sampling reads the global RNG
            options = ({"variation_values": coerce_variation(env, variation)}
                       if variation else None)
            obs, info = env.reset(seed=seed, options=options)
            policy = ARPolicy(low, high, RHOS[e % len(RHOS)], rng)
            state = state_of(obs)
            states, actions = [state], []
            for _t in range(horizon):
                act = policy().astype(dtype)
                obs, _reward, term, trunc, info = env.step(act)
                states.append(state_of(obs))
                actions.append(np.asarray(act, dtype=np.float64).ravel())
                if bool(np.any(term)) or bool(np.any(trunc)):
                    break
            obs_rows.append(np.asarray(states, dtype=np.float32))
            act_rows.append(np.asarray(actions, dtype=np.float32))
            ep_ptr.append(ep_ptr[-1] + len(states))
            act_ptr.append(act_ptr[-1] + len(actions))
            ep_meta.append({"seed": int(seed), "len": len(actions),
                            "rho": RHOS[e % len(RHOS)]})
            if (e + 1) % 50 == 0:
                print(f"[collect] {name}: {e + 1}/{n_eps} eps "
                      f"({time.time() - t0:.0f}s, {act_ptr[-1]} transitions)", flush=True)
    finally:
        env.close()

    obs = np.concatenate(obs_rows)
    act = np.concatenate(act_rows)
    DATA.mkdir(parents=True, exist_ok=True)
    tmp = out_npz.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, obs=obs, act=act,
                        ep_ptr=np.asarray(ep_ptr, np.int64),
                        act_ptr=np.asarray(act_ptr, np.int64))
    os.replace(tmp, out_npz)
    meta = {
        "task": task, "env_id": s["env_id"], "kwargs": s["kwargs"],
        "n_eps": n_eps, "horizon": horizon, "policy": f"AR(1) rho in {RHOS}",
        "variation": variation,
        "obs_dim": int(obs.shape[1]), "act_dim": int(act.shape[1]),
        "n_transitions": int(act.shape[0]), "seed0": SEED0,
        "swm_version": "0.1.1",
        "sha256": hashlib.sha256(out_npz.read_bytes()).hexdigest(),
        "episodes": ep_meta,
    }
    (DATA / f"{name}.meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[collect] wrote {out_npz}: {obs.shape[0]} rows, "
          f"{act.shape[0]} transitions, obs_dim {meta['obs_dim']}, "
          f"act_dim {meta['act_dim']} ({time.time() - t0:.0f}s)")
    return out_npz


def load_corpus(task: str, name: str | None = None) -> dict:
    name = name or task
    with np.load(DATA / f"{name}.npz") as z:
        corpus = {k: z[k] for k in z.files}
    corpus["meta"] = json.loads((DATA / f"{name}.meta.json").read_text())
    return corpus


def episode_slices(corpus):
    optr, aptr = corpus["ep_ptr"], corpus["act_ptr"]
    return [((int(optr[i]), int(optr[i + 1])), (int(aptr[i]), int(aptr[i + 1])))
            for i in range(len(optr) - 1)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--eps", type=int, default=COLLECT_EPISODES)
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--name", default=None)
    a = p.parse_args()
    collect(a.task, a.eps, a.horizon, a.name)


if __name__ == "__main__":
    main()
