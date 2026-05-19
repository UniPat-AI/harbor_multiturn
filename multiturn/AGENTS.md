# Multiturn Docs Maintenance

- This directory is the local multiturnpp overlay documentation for the Harbor fork.
- Keep these files focused on the current behavior of `multiturn/main`.
- Do not edit upstream Harbor `README.md` or the Fumadocs site just to describe local multiturn behavior.
- When multiturn runtime behavior, CLI parameters, task layout, resume semantics, snapshot semantics, or maintenance tests change, update the relevant document here and the parent `TEST.md` / `tests/README.md` when test entrypoints change.
- Do not include credentials, local `.env` values, run logs, or generated job outputs.
