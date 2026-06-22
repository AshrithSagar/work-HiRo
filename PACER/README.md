# PACER: Progress-Aligned Curation for Error-Resilient Imitation Learning

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

<https://openreview.net/forum?id=gaYyBvP2Rz>

> [!WARNING]
> This uses my experimental [`typingkit`](https://github.com/AshrithSagar/typingkit) package.

## Setup

<details>

<summary>Clone the repo</summary>

```shell
git clone https://github.com/AshrithSagar/work-Hiro.git
cd work-Hiro
cd PACER
```

</details>

<details>

<summary>Install uv</summary>

Install [`uv`](https://docs.astral.sh/uv/), if not already.
Check [here](https://docs.astral.sh/uv/getting-started/installation/) for installation instructions.

It is recommended to use `uv`, as it will automatically install the dependencies in a virtual environment.
If you don't want to use `uv`, skip to the next step.

TL;DR: Just run

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

</details>

The dependencies are listed in the [pyproject.toml](pyproject.toml) file.

Install the package in editable mode (recommended):

```shell
# Using uv
uv sync --all-groups --all-extras

# Or with pip
pip install -e .
```

## License

This project falls under the [MIT License](../../LICENSE).
