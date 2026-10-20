"""Built-in task families; imports perform registry registration.

Frozen suite (target-design.md §5, spec ``csfrbench-targets/freeze-1.0``):
node_anova, flat_anova, tree_anova, tree_relu, dag_anova,
tree_tri, dense_anova, tree_fourier. Generator freeze: no target-definition
changes past this point absent a documented protocol failure.

Extension families (graph-generalization.md, spec ``EXT_SPEC``; separately
versioned, not part of the frozen core suite): rand_anova, rand_fourier.

Finalization candidates (v1 certification campaign; design parameters
calibrate in its Stage A): dense_fourier, dag_fourier.
"""
from . import (  # noqa: F401
    dag_anova,
    dag_fourier,
    dense_anova,
    dense_fourier,
    flat_anova,
    functional_anova,
    node_anova,
    rand,
    tree_anova,
    tree_fourier,
    tree_relu,
    tree_tri,
)

#: Frozen target-specification id; recorded in manifests, never in fingerprints
#: (the freeze changes roles and disclosures, not target bits).
FROZEN_SPEC = "csfrbench-targets/freeze-1.0"

#: Graph-generalization extension spec id (rand_anova, rand_fourier).
EXT_SPEC = "csfrbench-targets/ext-graphgen-1.0"

__all__ = [
    "FROZEN_SPEC", "EXT_SPEC",
    "dag_anova", "dense_anova", "flat_anova", "node_anova", "rand",
    "tree_anova", "tree_fourier", "tree_relu", "tree_tri",
]
