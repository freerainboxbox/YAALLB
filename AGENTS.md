# AGENTS.md

## Project Context
* This is a Python project written with FastAPI and targetting multiple LLM runners.
* See README.md for project purpose.
* Project is currently in setup phase.
* `uv` is used for version management.

## Instructions for Problematic Harnesses
* Agents with stateless shell runners should _always_ use `uv run` for running Python script and tests. Any sort of virtual environment encountered on shell start is _not_ preserved in spawned shells, and this is required for scripts to run.
* Stateless shell runners include, but are not limited to:
  * OpenCode
  * Claude Code
  * Gemini Code Assist
* Test for statelessness by activating the virtual environment in one shell call, then check for `$VIRTUAL_ENV` in the environment of another shell call. If it is not active, then you are in a stateless harness.

## Allowed Scope
* Commits should encapsulate single purposes. When implementing a feature and bugs are present that need to be corrected, commit a bugfix first, then commit the feature. Commit as many bugfixes as needed.
* Modify informational markdown files (including this one) as needed, especially as the facts change.
* Prefer to append bullets and delete bullets, and never make extremely structural changes. Only mutate bullets for specific reasons, like changed facts.
* Do not modify system files or dotfiles, with the exception of .gitignore or .venv when the user asks for direct troubleshooting of .venv issues.
* Install libraries as needed using `uv add`.
* Pin package versions or remove package versions as needed.
* Prefer stable versions whenever possible, unless there is no way forward.
* NEVER use local package patches. All code must be fully reproducible on any machine. Pin old versions if required.
* Online research is allowed and strongly encouraged. Fetch valid documentation of libraries, or delegate a subagent to do so and report on correct usage. If you are a subagent, do not further delegate.
* Prefer not to reverse-engineer libraries unless there is an esoteric bug that is not documented on first search.
* Notify the user if the search engine tool or URL fetching capabilities are faulty (after three attempts on any given query or URL), and return control to the user rather than continue to attempt.
* Request that the user provides specific documentation if unable to fetch, so the user can provide it manually. Do not assume anything except for extremely common language API knowledge.

## Testing
* Unit tests should be written in pytest.
* Mock as little as possible.
* When implementing a feature, use red-green testing (check failure before checking that a new feature passes).
* If forgetting to implement the test first, stash/commit and check red on a previous checkout, then pop/reset to the new code and check green.
* Do not delete tests that already exist unless the premise of the test is no longer valid (e.g. an entire feature has been removed).