FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        swig \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "uv==0.11.6"

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN uv sync --frozen --no-dev --no-editable


FROM python:3.12-slim AS core

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app /app

CMD ["python", "-c", "import emsaplibraries; print(emsaplibraries.__version__)"]


FROM core AS tools

RUN apt-get update \
    && for package in mafft pdb2pqr propka apbs; do \
        if apt-cache show "$package" >/dev/null 2>&1; then \
            DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "$package"; \
        else \
            echo "Package $package is not available from this Debian base image"; \
        fi; \
    done \
    && rm -rf /var/lib/apt/lists/*

CMD ["python", "-c", "import emsaplibraries.external; print('emsaplibraries tools image ready')"]
