"""External benchmarks (no training, no tuning): data adapters, official prompts and scoring, judge client, tables.

Everything here is plain Python (no torch import) so it can be tested without a GPU. The GPU phases live in
``scripts/run_external_benchmark.py``; the control centre is ``notebooks/VLM_Dental_Colab_External_Benchmarks.ipynb``.
"""
