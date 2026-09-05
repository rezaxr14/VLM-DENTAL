# Token Constraints & Hyperparameter Invariants Rule

## STRICT MANDATORY INVARIANT: NEVER ALTER USER TOKEN CONSTRAINTS OR BUDGETS

1. **ABSOLUTE PROHIBITION ON REDUCING OR MODIFYING TOKEN HEADROOM / LIMITS**:
   - The agent is STRICTLY PROHIBITED from ever lowering, capping, altering, clamping, or overriding the user's token budgets, headroom, or limits (e.g., `LOCAL_MAX_TOKENS = 16384`, `max_tokens`, `TRANSFORMERS_MAX_TOKENS`, `max_new_tokens`) under ANY circumstance across all scripts, notebooks, and configurations.
   - If the user has set a token headroom (e.g. `16384`), it MUST REMAIN EXACTLY as specified.
   - Do NOT propose, suggest, or unilaterally edit code to lower token limits (e.g. down to 4096, 2048, etc.) to "prevent runaways", "save memory", or "speed up execution". The user specifically requires full unconstrained generation capacity.

2. **PRESERVE USER-SPECIFIED DEPENDENCIES & VERSIONS**:
   - Never downgrade, change, or remove user-specified dependency requirements (e.g., `transformers>=5.0.0`) unless explicitly commanded by the user.

3. **INDEPENDENCE OF SAMPLING HYPERPARAMETERS FROM TOKEN CONSTRAINTS**:
   - Looping or degeneration prevention must be addressed solely through generation penalties (e.g., `repetition_penalty = 1.10`) or stopping criteria, NEVER by truncating or lowering the maximum token budget.
