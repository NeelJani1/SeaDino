"""High-throughput benchmark runner with backwards-compatible re-exports."""

from eval_suite.benchmarks.single_label import evaluate_knn, evaluate_single_label

__all__ = ["evaluate_knn", "evaluate_single_label"]
