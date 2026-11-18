"""Segmentation foundation: output contract, mask utilities and metrics. Isolated on purpose: it imports only numpy / PIL /
scipy, never ``dental_agent.evaluation`` (whose ``__init__`` pulls the training stack) and never the hashed agent folders, so
adding it does not change any existing result identity. It is not wired into the agent loop; later work builds on it."""
