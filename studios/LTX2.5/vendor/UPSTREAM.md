# Vendored LTX-2 packages

The directories `ltx-core/` and `ltx-pipelines/` are copied without source-code
changes from the official Lightricks LTX-2 repository:

- Repository: https://github.com/Lightricks/LTX-2
- Commit: `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`
- Source paths: `packages/ltx-core`, `packages/ltx-pipelines`
- License: see `LICENSE.LTX-2.md` in this directory

They are vendored so a ModelScope Studio deployment does not need to clone
GitHub while installing Python dependencies. The pure-Python wheels in
`wheels/` were built from these exact source directories:

- `ltx_core-1.2.0-py3-none-any.whl` SHA-256:
  `f97e5cc46c2a177167074af3c0bc0523652ee5c2f9cee4c7d90cccd54c2b696f`
- `ltx_pipelines-1.2.0-py3-none-any.whl` SHA-256:
  `92bf52ea2223e76d6eb4f752e47468a79c65fd98d33870bf3fd377c88b10d374`

Model weights are not included.
