"""The evaluation settings of CSFRBench v1.5.

Two settings share the target instances, the methods and the protocol:

``primary``     every input is active: ``d = 8`` (``d = 16`` on the depth-5 tree).
``nuisance``    the same target instances with inert inputs added: ``d = 32``
                (24 inert inputs; 16 on the depth-5 tree).

A setting holds eight workloads. Each workload fixes a target family, the
ambient dimension, the target instances, the backbone, the training budget, the
grid of training-set sizes, the methods and their frozen search spaces.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import targets

TRIALS = 32                 # hyperparameter configurations per method, target instance and training-set size
SEARCH_SEED = 20260826      # the seed of the search; with the keys below it fixes every configuration
SEARCH_KEY = "m17"          # a key of the seed of the search; it has no other meaning
SIGMA_SQ = 0.01

#: Target instances of the evaluation. The search spaces were frozen on the instances 6016-6019, which are
#: disjoint from the evaluation instances.
INSTANCES = tuple(range(6000, 6016))
INSTANCES_SAMPLED = tuple(range(6000, 6016)) + tuple(range(6020, 6036))
DEVELOPMENT_INSTANCES = (6016, 6017, 6018, 6019)


@dataclass(frozen=True)
class Profile:
    """Backbone, training budget and grid of training-set sizes."""
    name: str
    width: int
    hidden: int
    max_steps: int
    eval_every: int
    sizes: tuple[int, ...]

    @property
    def n_max(self) -> int:
        return max(self.sizes)


COMMON = Profile("32x4", 32, 4, 64000, 1000, tuple(2 ** e for e in range(9, 16)))          # n = 2^9 .. 2^15
DEEP = Profile("64x8", 64, 8, 128000, 2000, tuple(2 ** e for e in range(11, 18)))          # n = 2^11 .. 2^17


@dataclass(frozen=True)
class Arm:
    """A method with its fixed arguments, as it appears in the tables of the paper."""
    name: str                   # identifier in this package
    paper_name: str
    method: str                 # registered method
    fixed: dict = field(default_factory=dict)       # hyperparameters that are not searched
    trial_key: str = ""         # key of the arm in the seed of the search
    record_name: str = ""       # name of the arm in the records of the paper's experiments
    graph_informed: bool = False


ARMS: dict[str, Arm] = {a.name: a for a in (
    Arm("lewi", "LeWi", "lewi", trial_key="lewi-v1", record_name="lewi-v1-rgm"),
    Arm("dense", "Dense", "dense", trial_key="noreg", record_name="noreg"),
    Arm("l1", "L1", "l1", trial_key="l1", record_name="l1"),
    Arm("l2", "L2", "l2", trial_key="l2", record_name="l2"),
    Arm("l0", "L0", "l0", fixed={"p_open_init": 0.8318}, trial_key="l0hc", record_name="l0hc"),
    Arm("rigl", "RigL", "rigl", trial_key="rigl", record_name="rigl"),
    Arm("bimt", "BIMT", "bimt", trial_key="bimt", record_name="bimt"),
    Arm("fiedler", "Fiedler", "fiedler", trial_key="fiedler", record_name="fiedler"),
    Arm("positions", "Neuron positions", "positions", trial_key="erb_positions", record_name="erb_positions"),
    Arm("go0", "GraphOracle(0)", "graph_oracle", fixed={"levels": 0}, trial_key="support_oracle",
        record_name="support_oracle", graph_informed=True),
    Arm("go1", "GraphOracle(1)", "graph_oracle", fixed={"levels": 1}, trial_key="go1", record_name="go1",
        graph_informed=True),
    Arm("go2", "GraphOracle(2)", "graph_oracle", fixed={"levels": 2}, trial_key="go2", record_name="go2",
        graph_informed=True),
    Arm("go_all", "GraphOracle(all)", "graph_oracle", fixed={"levels": "all"}, trial_key="graph_oracle",
        record_name="graph_oracle", graph_informed=True),
)}
LEARNED = ("lewi", "dense", "l1", "l2", "l0", "rigl", "bimt", "fiedler", "positions")
#: searched hyperparameters of every arm beside the learning rate
METHOD_KEYS = {"lewi": set(), "dense": set(), "l1": {"lam"}, "l2": {"lam"}, "l0": {"lam"}, "rigl": {"density"},
               "bimt": {"lam"}, "fiedler": {"delta"}, "positions": set(), "go0": set(), "go1": set(), "go2": set(),
               "go_all": set()}
ARM_ORDER = (*LEARNED, "go0", "go1", "go2", "go_all")
ARM_OF_RECORD = {a.record_name: a.name for a in ARMS.values()}


@dataclass(frozen=True)
class Workload:
    name: str                   # identifier in this package
    paper_name: str
    setting: str
    family: str                 # target family (fixes the seeds of the target)
    preset: str
    d: int
    active: int                 # active inputs
    levels: int                 # nonlinear levels of the target graph
    profile: Profile
    instances: tuple[int, ...]
    arms: tuple[str, ...]
    space_set: str              # frozen search spaces of the workload
    record_cell: str            # name of the workload in the records of the paper's experiments
    sigma_sq: float = SIGMA_SQ

    def space_set_of(self, arm: str) -> str:
        """The set of frozen search spaces of an arm (neuron positions has its own set in the primary setting)."""
        if self.space_set == "p8b" and arm == "positions":
            return "pos8p"
        return self.space_set

    def target(self, instance: int) -> targets.Target:
        return targets.draw(self.family, self.preset, self.d, int(instance), sigma_sq=self.sigma_sq)


# name, paper name, family, preset, active inputs, nonlinear levels, GraphOracle rungs below the full one
_TABLE = (
    ("flat_modules", "Flat modules", "flat_functional_anova", "w4", 8, 1, ()),
    ("depth4_tree", "Depth-4 tree", "tree_functional_anova", "depth3", 8, 3, ("go1",)),
    ("reuse_dag", "Reuse DAG", "dag_functional_anova", "default", 8, 2, ("go1",)),
    ("sampled_dags", "Sampled depth-4 DAGs", "rand_functional_anova", "default", 8, 3, ("go1", "go2")),
    ("fourier_tree", "Fourier tree", "tree_fourier", "binary_depth3", 8, 3, ("go1",)),
    ("fourier_reuse_dag", "Fourier reuse DAG", "dag_fourier_v2", "default", 8, 2, ("go1",)),
    ("high_arity_node", "High-arity node", "dense_fourier", "frozen_v1", 8, 1, None),
    ("depth5_tree", "Depth-5 tree", "tree_functional_anova", "depth4", 16, 4, ("go1", "go2")),
)
_RECORD_CELL = {
    "primary": {"flat_modules": "flat8", "depth4_tree": "tree8", "reuse_dag": "dag8", "sampled_dags": "rand8",
                "fourier_tree": "ftree8", "fourier_reuse_dag": "fdag8", "high_arity_node": "dense8",
                "depth5_tree": "tree5d16"},
    "nuisance": {"flat_modules": "flat32s", "depth4_tree": "tree32s", "reuse_dag": "dag32s",
                 "sampled_dags": "rand32s", "fourier_tree": "ftree32s", "fourier_reuse_dag": "fdag32s",
                 "high_arity_node": "dense32s", "depth5_tree": "tree5d32"},
}


def _build() -> dict[str, dict[str, Workload]]:
    out: dict[str, dict[str, Workload]] = {"primary": {}, "nuisance": {}}
    for setting in out:
        for name, paper, family, preset, active, levels, rungs in _TABLE:
            deep = name == "depth5_tree"
            arms = list(LEARNED)
            if setting == "nuisance":
                arms.append("go0")          # with every input active GraphOracle(0) is Dense
            if rungs is not None:           # the high-arity node has one node over all inputs: no rung differs from Dense
                arms.extend(rungs)
                arms.append("go_all")
            if setting == "primary":
                d, space_set = active, ("p0" if deep else "p8b")
            else:
                d, space_set = 32, ("p32t" if deep else "p32s")
            out[setting][name] = Workload(
                name=name, paper_name=paper, setting=setting, family=family, preset=preset, d=d, active=active,
                levels=levels, profile=DEEP if deep else COMMON,
                instances=INSTANCES_SAMPLED if name == "sampled_dags" else INSTANCES,
                arms=tuple(a for a in ARM_ORDER if a in arms), space_set=space_set,
                record_cell=_RECORD_CELL[setting][name])
    return out


SETTINGS: dict[str, dict[str, Workload]] = _build()
SETTING_TITLE = {"primary": "Primary setting: every input is active",
                 "nuisance": "Nuisance-input setting: inert inputs added (d = 32)"}

#: Target families of Table 1 of the paper that are not part of a campaign. They serve the
#: local-function fitting study: the model receives exactly the arguments of the function (d = k), without noise.
LOCAL_FUNCTIONS = {
    "single_function": ("Single local function, additive-multiplicative", "node_functional_anova", "default", 2),
    "single_fourier": ("Single local function, Fourier", "tree_fourier", "single", 2),
    "high_arity_stress": ("High-arity node, additive-multiplicative (stress)", "dense_functional_anova", "stress8", 8),
    "high_arity_node": ("High-arity node, Fourier", "dense_fourier", "frozen_v1", 8),
}


def local_function(name: str, instance: int) -> targets.Target:
    """A single local function in isolation: ``d`` equals its arity and there is no noise."""
    _paper, family, preset, arity = LOCAL_FUNCTIONS[name]
    return targets.draw(family, preset, arity, int(instance), sigma_sq=0.0)


def workload(setting: str, name: str) -> Workload:
    if setting not in SETTINGS:
        raise KeyError(f"unknown setting {setting!r}; available: {sorted(SETTINGS)}")
    if name not in SETTINGS[setting]:
        raise KeyError(f"unknown workload {name!r}; available: {list(SETTINGS[setting])}")
    return SETTINGS[setting][name]


def workload_of_record(cell: str) -> Workload:
    """The workload with the given name in the records of the paper's experiments."""
    for setting in SETTINGS.values():
        for w in setting.values():
            if w.record_cell == cell:
                return w
    raise KeyError(f"no workload has the record name {cell!r}")
