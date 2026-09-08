# Anti-False-Confidence Testing & Reality-Grounded Verification Rule

## STRICT MANDATORY INVARIANTS FOR ALL AGENTS

1. **Absolute Prohibition of False-Confidence Claims from Local Tests**:
   - The agent MUST NEVER claim that code is "100% verified", "fully tested", or "guaranteed to work" based on local CPU unit tests or pytest runs when the target execution environment is Cloud TPU v5e-8, Kaggle TPU VM, or distributed GPU clusters.
   - Local execution environments lack the TPU hardware, `/dev/vfio/*` IOMMU groups, `libtpu.so` C++ runtime, and 16 GB HBM memory ceilings present on Cloud TPUs.

2. **Prohibition of Misleading Mock Tests**:
   - The agent MUST NEVER construct superficial mock tests (e.g. mocking `xla_model` to return CPU, mocking device memory, or mocking distributed backends) and present passing mock results as proof of cloud hardware readiness.
   - Mocking away hardware removes the exact failure modes (PJRT device contention, FSDP parameter sharding signatures, HBM memory limits) that break on cloud systems.

3. **Mandatory Explicit Disclaimers**:
   - Whenever reporting local test or lint results, the agent MUST explicitly state the scope:
     - What was verified: Python syntax, AST validity, command-line argument parsing, static type contracts.
     - What was NOT verified and requires cloud runtime execution: TPU PJRT initialization, multi-core device spawning, FSDP hardware sharding, and HBM memory consumption.

4. **Hardware Failure Analysis Over Unit Test Justification**:
   - When a user reports a runtime or hardware failure from Kaggle or Colab, the agent MUST NEVER deflect, justify, or cite passing local unit tests.
   - Focus 100% of effort on direct stack trace forensics, C++ check failures, environment variables, memory math, and hardware runtime mechanics.
