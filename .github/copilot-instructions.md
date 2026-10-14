# HOOPS AI Tutorials Rules

## Git Safety

- Never push directly to `main` in this repository, even when credentials allow a
  branch-protection bypass. A general request to commit or push is not an exception.
- Use a feature branch and a pull request targeting `main`. Check the current branch,
  push destination and unpushed commits before any push. If currently on `main`,
  move the intended work to a feature branch before committing or pushing it.
- Never update or delete remote `refs/heads/main` through any refspec, force push,
  GitHub API, MCP tool or other direct-write route. Changes to `main` go through PRs.
- Never bypass or disable the local pre-push guard, use `--no-verify`, override
  `core.hooksPath`, or bypass GitHub branch protections to complete a push.
- Preserve unrelated working-tree and staged changes. Stage and commit only the
  files authorized for the task.

## Local Protection

This PC's tutorial clone has a `.git/hooks/pre-push` guard rejecting every push
whose destination is `refs/heads/main`, including deletion and feature-to-main
refspecs. Feature-branch pushes remain allowed. The hook is local and is not
installed automatically in new clones. It is an accident-prevention guard, not
a security boundary; GitHub must enforce PR requirements without bypass rights
for protection across machines and API clients.