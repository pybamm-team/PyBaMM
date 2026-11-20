# Changelog fragments

Each pull request with a user-visible change adds one file here instead of editing
`CHANGELOG.md`. At release time, `towncrier build` collects the files into a dated
block of `CHANGELOG.md` and deletes them.

Name the file `<PR number>.<type>.md`, e.g. `1234.bugfix.md`, where `<type>` is one of:

| Type | Section |
| --- | --- |
| `breaking` | Breaking changes (include a one-line migration note) |
| `deprecated` | Deprecated (say when removal is planned) |
| `feature` | Features |
| `bugfix` | Bug fixes |
| `optimization` | Optimizations |

A PR with several entries of the same type numbers the extras: `1234.bugfix.1.md`,
`1234.bugfix.2.md`.

The file holds the entry text only, without a leading `- ` or a PR link; the link is added
from the file name. Nested bullets go on the following lines, unindented:

```markdown
`IDAKLUSolver.jaxify` now returns correct derivatives at any requested time.
- Inputs are looked up by name.
- Missing inputs raise a `SolverError`.
```

Preview the next release's notes with:

```bash
uvx towncrier build --draft --version Unreleased
```
