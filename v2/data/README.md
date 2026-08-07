# Data layout

This directory defines local development data boundaries. Production data should live in durable storage, not Git.

- `raw/`: source-faithful local samples or generated fixtures; do not commit licensed/full source corpora.
- `fixtures/`: tiny deterministic test inputs safe to version.
- `gold/`: hand-labeled evaluation examples safe to version when licensing permits.

Never commit API keys, private user data, or large downloaded datasets.
