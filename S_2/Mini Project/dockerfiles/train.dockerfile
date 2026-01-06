FROM ghcr.io/astral-sh/uv:python3.12.3-alpine AS base

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

RUN uv sync --frozen --no-install-project

COPY src src/

RUN uv sync --frozen

ENTRYPOINT ["uv", "run", "src/mlops_mini_project/train.py"]
