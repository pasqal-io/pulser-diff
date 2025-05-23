# PulserDiff

`PulserDiff` is a differentiable backend for pulse-level quantum simulation framework [Pulser](https://github.com/pasqal-io/Pulser).

It aims at providing completely differentiable quantum states or expectation values through use of automatic differentiation capabilities of underlying [PyTorch](https://pytorch.org/) package.


[![Unit tests](https://github.com/pasqal-io/pulser-diff/actions/workflows/test.yml/badge.svg)](https://github.com/pasqal-io/pulser-diff/actions/workflows/test.yml)
[![Notebook tests](https://github.com/pasqal-io/pulser-diff/actions/workflows/test_notebooks.yml/badge.svg)](https://github.com/pasqal-io/pulser-diff/actions/workflows/test_notebooks.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


## Installation guide

`PulserDiff` can be installed from source by entering your preferred virtual python environment and running the following commands:

```bash
git clone https://github.com/pasqal-io/pulser-diff.git
cd pulser-diff
pip install .
```

## Develop

When developing the package, the recommended way is to create a virtual environment with `hatch` as shown below:

```bash
python -m pip install hatch
python -m hatch -v shell
```

When inside the shell with development dependencies, install first the pre-commit hook:
```
pre-commit install
```

In this way, you will get automatic linting and formatting every time you commit new code. Do not
forget to run the unit test suite by simply running the `pytest` command.

If you do not want to get into the Hatch shell, you can alternatively do the following:

```bash
python -m pip install hatch
python -m hatch -v shell

# install the pre-commit
python -m hatch run pre-commit install

# commit some code
python -m hatch run git commit -m "My awesome commit"

# run the unit tests suite
python -m hatch run pytest

```

## Document

You can improve the documentation of the package by editing this file for the landing page or adding new
markdown or Jupyter notebooks to the `docs/` folder in the root of the project. In order to modify the
table of contents, edit the `mkdocs.yml` file in the root of the project.

In order to build and serve the documentation locally, you can use `hatch` with the right environment:

```bash
python -m hatch -v run docs:build
python -m hatch -v run docs:serve
```
