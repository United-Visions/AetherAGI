# Publishing to PyPI

## Prerequisites

1. Create account on [PyPI](https://pypi.org/account/register/) and [TestPyPI](https://test.pypi.org/account/register/)
2. Install build tools:

```bash
pip install build twine
```

## Build Package

```bash
cd sdk/python
python -m build
```

This creates:
- `dist/aethermind-1.0.0.tar.gz` (source distribution)
- `dist/aethermind-1.0.0-py3-none-any.whl` (wheel)

## Test on TestPyPI First

```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Test install
pip install --index-url https://test.pypi.org/simple/ aethermind

# Test it works
python -c "from aethermind import AetherMindClient; print('Success!')"
```

## Publish to PyPI

```bash
# Upload to production PyPI
twine upload dist/*

# Enter your PyPI credentials
# Username: __token__
# Password: pypi-xxxxxxxxxxxxxxxxxxxxx
```

## Install from PyPI

Users can now install with:

```bash
pip install aethermind
```

## Version Management

Update version in:
1. `aethermind/__init__.py` - `__version__ = "1.0.1"`
2. `pyproject.toml` - `version = "1.0.1"`
3. `setup.py` - Reads from `__init__.py`

Then rebuild and republish.

## API Token Setup

Create `.pypirc` file in home directory:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-AgEIcHlwaS5vcmcC...your-token-here

[testpypi]
username = __token__
password = pypi-AgENdGVzdC5weXBpLm9y...your-token-here
```

Then upload without entering credentials:

```bash
twine upload dist/*
```

## Automated Publishing (GitHub Actions)

Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install build twine
      - name: Build package
        run: python -m build
      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: twine upload dist/*
```

Add `PYPI_API_TOKEN` to GitHub repository secrets.

## Verification

After publishing, verify:

1. Package appears on [pypi.org/project/aethermind](https://pypi.org/project/aethermind)
2. Install works: `pip install aethermind`
3. Import works: `python -c "import aethermind"`
4. README displays correctly on PyPI page
5. Version badge shows latest: [![PyPI](https://img.shields.io/pypi/v/aethermind.svg)](https://pypi.org/project/aethermind/)

## Distribution Checklist

Before publishing:

- [ ] Update version number in all files
- [ ] Update CHANGELOG.md
- [ ] Run tests: `pytest`
- [ ] Build package: `python -m build`
- [ ] Test install locally: `pip install dist/aethermind-*.whl`
- [ ] Upload to TestPyPI first
- [ ] Test TestPyPI install
- [ ] Upload to production PyPI
- [ ] Create GitHub release with tag
- [ ] Update documentation
- [ ] Announce on Discord/Twitter
