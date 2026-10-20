"""CSFRBench: a controlled benchmark for compositionally sparse function regression.

    import csfrbench
    task = csfrbench.task("depth4_tree", instance=6000, n=1024)     # examples only; the graph stays hidden
    ...                                                             # fit any model on task.X_train, task.y_train
    task.evaluate(model)                                            # test NMSE, as reported in the paper

See README.md beside this file.
"""
__version__ = "1.6.0"

from .benchmark import LearningCurve, Task, compare, learning_curve, task  # noqa: E402

__all__ = ["Task", "LearningCurve", "task", "learning_curve", "compare", "__version__"]
