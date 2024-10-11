# The base layer will contain all the shared dependencies for the builder and
# runtime layers
FROM python:3.11-slim AS base

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

RUN apt-get update && apt-get install -y build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# The builder layer is where we install poetry and all it’s build dependencies.
FROM base AS builder

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

RUN pip install poetry==1.8.3

COPY pyproject.toml poetry.lock ./

RUN poetry install --without dev --no-root && rm -rf $POETRY_CACHE_DIR;

# Finally we can setup our final runtime layer.
FROM base AS runtime

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

# Copy the rest of the application code
COPY . /app

WORKDIR /app

RUN pip install -e . --no-cache-dir --no-deps

EXPOSE 8000

ENTRYPOINT ["/app/entrypoints/services/run_rest.sh"]
