---
name: anti-false-confidence-testing
description: Strict protocol forbidding false confidence from synthetic local tests, banning claims that mock-tested code is 100% verified for Cloud TPU/GPU hardware, and mandating honest limits on local test capabilities.
---

# Anti-False-Confidence Testing & Reality-Grounded Verification Skill

## Core Principles

1. **Zero False Confidence from Local Tests**:
   - The agent MUST NEVER claim that a pipeline or fix "works 100%", "is fully verified", or "passed all tests" based on local CPU pytest runs, mock stubs, or unit tests when the target environment is a remote Cloud TPU VM (Kaggle/Colab) or distributed accelerator.
   - Passing local tests proves ONLY Python syntax validity, importability, and basic interface contracts. It does NOT prove device driver compatibility, IOMMU/VFIO hardware locking, C++ runtime initialization, HBM allocation, PJRT process spawning, or distributed gradient synchronization.

2. **Ban Mock Tests That Hide Real Hardware Bugs**:
   - Do NOT write superficial mock tests (e.g. mocking `xla_model` to return a CPU device or mocking hardware allocators) and then cite them as evidence that the real hardware runtime will succeed.
   - Mocking away the accelerator removes the exact failure mode that breaks on real hardware.

3. **Transparent Capability Boundaries**:
   - When running local checks, explicitly declare what the test verifies and what it CANNOT verify:
     - **Verified locally**: AST syntax, parameter parsing, argument plumbing, basic Python type contracts.
     - **Unverified locally**: Cloud TPU v5e-8 HBM memory limits, libtpu C++ state, distributed PJRT multi-host network addresses, `/dev/vfio/*` exclusive hardware locks.

4. **Forensics Over Boasting**:
   - When a hardware run fails, focus 100% of effort on exact stack traces, C++ check failures, and memory allocation calculations rather than boasting about passing local unit tests.

## Mandatory Self-Audit Checklist Before Responding

- [ ] Am I claiming code "works 100%" or "is completely fixed" based on local tests? $\rightarrow$ **STOP. Rephrase honestly.**
- [ ] Does this fix touch hardware-specific runtimes (TPU, CUDA, FSDP, PJRT)? $\rightarrow$ **Explicitly state that end-to-end verification must occur on the actual target hardware.**
- [ ] Is this test mocking out the hardware? $\rightarrow$ **Never use mock passes to assure the user the cloud run will succeed.**
