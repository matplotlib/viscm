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
