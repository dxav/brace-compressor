# Contributing

## Development setup

Use Python 3.11 or newer. A Rust toolchain is optional and is used only for
the accelerated scan.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[test]"
```

## Before opening a pull request

Run the full test suite and check the patch:

```bash
PYTHONPATH=. pytest tests -q
git diff --check
```

Changes to prediction arithmetic, quantization, entropy coding, container
flags, or model versions must update the paired encode/decode path, tests, and
algorithm documentation. Do not commit generated build directories, virtual
environments, local datasets, or credentials.

## Pull requests

Describe the motivation, behavior change, validation performed, and any
compatibility impact. Keep unrelated refactors out of focused changes.
