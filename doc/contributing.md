# Contributing

Install development dependencies:

```
conda env create  # or `mamba env create`
```


## Development install

```
pip install -e .
```


## Testing the build

```
rm -rf dist
python -m build
pip install dist/*.whl  # or `dist/*.tar.gz`
```


## Code formatting and linting

This codebase uses [black](https://black.readthedocs.io/en/stable/) and
[ruff](https://github.com/charliermarsh/ruff) to automatically format and lint the code.

[`pre-commit`](https://pre-commit.com/) is configured to run them automatically. You can
trigger this manually with `pre-commit run --all-files`.

Thanks to pre-commit, all commits should be formatted. In cases where formatting needs
to be fixed (e.g. changing config of a linter), a format-only commit should be created,
and then another commit should immediately follow which updates
`.git-blame-ignore-revs`. For example:
[1fec42d](https://github.com/matplotlib/viscm/pull/64/commits/1fec42d0baf90e00d510efd76cb6006fa0c70dc4),
[8aa7bb0](https://github.com/matplotlib/viscm/pull/64/commits/8aa7bb01440aeca6f8bbcefe0671c28f2ce284c6).
