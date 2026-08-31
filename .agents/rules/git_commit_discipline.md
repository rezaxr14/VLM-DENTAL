# Git Commit & Version Control Discipline Rule

## STRICT MANDATORY INVARIANTS FOR ALL AGENTS

1. **ABSOLUTE PROHIBITION OF AUTOMATIC GIT COMMITS**:
   - The agent MUST NEVER execute `git commit` autonomously under ANY circumstance.
   - Passing tests, completing tasks, generating plans, or creating files does NOT grant permission to commit.
   - A `git commit` command may ONLY be run if the USER explicitly instructs "git commit" or "commit this" in their current prompt.

2. **PROHIBITION OF PIECEMEAL / MICRO-COMMITS**:
   - The agent MUST NEVER create multiple sequential micro-commits during an iteration or task.
   - When a commit is explicitly commanded by the user, all staged and verified changes MUST be combined/squashed into **ONE single clean commit** with a clear, concise conventional commit message.

3. **STRICT PROHIBITION OF GIT PUSH**:
   - The agent MUST NEVER execute `git push` under ANY circumstance. All remote pushes are reserved exclusively for the user.

4. **CLEAN WORKING TREE & DIFF TRANSPARENCY**:
   - Keep working changes clean and uncommitted in the local working directory so the user can inspect diffs directly before deciding whether to commit or push.
