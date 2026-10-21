"""CSFRBench: a controlled benchmark for compositionally sparse function regression.

    import csfrbench
    task = csfrbench.task("depth4_tree", instance=6016, n=1024)     # a development instance; examples only
    ...                                                             # fit any model on task.X_train, task.y_train
    task.evaluate(model)                                            # test NMSE, as reported in the paper

See README.md beside this file.
"""
__version__ = "1.6.1"
#: The benchmark definition this package implements: targets, data, protocol and records of the paper (docs/SPEC.md).
BENCHMARK_VERSION = "1.5"

from .benchmark import LearningCurve, Task, compare, learning_curve, task  # noqa: E402

__all__ = ["Task", "LearningCurve", "task", "learning_curve", "compare", "__version__", "BENCHMARK_VERSION"]
