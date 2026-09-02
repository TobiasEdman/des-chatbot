<!-- agentic-task:coordination:start -->
## Cross-runtime coordination mechanics

@AGENTS.md

Shared policy lives in `AGENTS.md`. Claude-specific hooks may enforce it but
must not weaken or duplicate that policy. Use a Claude worktree for every
writing session. Invoke task and continuity commands only through the external,
non-editable operator wheel runtime described in `.agents/README.md`; never
execute a verifier from the repository or caller `PATH`. Commit only validated task-scoped
`.agents/handoffs/` records for another machine.
<!-- agentic-task:coordination:end -->
