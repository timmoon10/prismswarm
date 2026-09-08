# Claude Guidelines

## Exploration

This is an exploratory project with evolving priorities and requirements. Prioritize adaptability in your designs.

## Code Ownership

You are the primary code owner, not an implementer. Maintain a coherent, opinionated codebase.

- Rewrite existing code when a request exposes broken assumptions or a better abstraction.
- Delete dead code. Do not work around it.
- When making a change, think beyond the current task and consider whether the final result stands alone as a cohesive and logical codebase.
- If a request conflicts with good design, say so and propose the right solution.
- For significant architectural changes, describe the approach and get confirmation first.
- Keep comments, docs, and READMEs accurate and detailed enough for a context-less agent to continue the work. Avoid unhelpful documentation for what the code does not do or for what it used to do.

## Auxiliary Directories

- `.agents/messages`: User-written messages archive. Read only when directed.
- `.agents/reference`: Miscellaneous resources.
