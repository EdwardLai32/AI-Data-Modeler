# AutoML Architect — API image.
#
# Multi-stage: the build stage owns compilers and pip caches, the runtime stage
# gets only a populated virtualenv. That matters more than usual here because the
# dependency set (scipy, scikit-learn, xgboost, lightgbm, shap, reportlab) pulls
# in a large build toolchain that has no business shipping to production.
#
# Build:
#   docker build -t automl-architect .
#   docker build --build-arg EXTRAS=".[api,boost,tuning,explain,reports]" -t automl-architect:lean .
# Run:
#   docker run --rm -p 8000:8000 \
#     -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
#     -v automl-workspace:/var/lib/automl \
#     automl-architect

# ---------------------------------------------------------------------------
# Stage 1 — build
# ---------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS build

# Which optional extras to install. `all` includes cloud, kaggle, and SQL drivers;
# override to keep the image smaller.
ARG EXTRAS=".[api,boost,tuning,explain,charts,reports,sql]"

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# build-essential and gfortran for any wheel that has to compile; libpq-dev for
# psycopg. All of it stays in this stage.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        libpq-dev \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /src

# Copy only the metadata first so a source-only change does not invalidate the
# dependency layer — that layer takes minutes to rebuild.
COPY pyproject.toml README.md ./
COPY automl_architect/__init__.py automl_architect/__init__.py

RUN pip install --upgrade pip setuptools wheel \
    && pip install "${EXTRAS}"

COPY automl_architect/ automl_architect/
COPY examples/ examples/

# A *non-editable* reinstall, deliberately: an editable install records a path
# into /src, which does not exist in the runtime stage, so every import would
# fail there. This puts the real package inside the venv that gets copied.
RUN pip install --no-deps --force-reinstall --no-build-isolation . \
    && python -c "import automl_architect; print('built', automl_architect.__version__)" \
    && python -c "from automl_architect.api.app import create_app; create_app()" \
    && python -c "import automl_architect.cli"

# ---------------------------------------------------------------------------
# Stage 2 — runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="AutoML Architect" \
      org.opencontainers.image.description="Autonomous multi-agent AI data scientist" \
      org.opencontainers.image.licenses="MIT"

# libgomp is required at runtime by xgboost, lightgbm, and sklearn's OpenMP paths;
# libpq5 by psycopg. Neither needs the -dev package here.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libpq5 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root. A pipeline that executes model-authored plans against caller-supplied
# data should not be running as uid 0.
RUN groupadd --system --gid 1001 automl \
    && useradd --system --uid 1001 --gid automl --create-home --home-dir /home/automl automl

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # 0.0.0.0 because 127.0.0.1 inside a container is unreachable from outside it.
    AUTOML_API_HOST=0.0.0.0 \
    AUTOML_API_PORT=8000 \
    AUTOML_WORKSPACE=/var/lib/automl \
    AUTOML_LOG_LEVEL=INFO \
    # A CPU-limited container with n_jobs=-1 oversubscribes and runs slower than
    # with a sane cap. Override to -1 if the container owns the host.
    AUTOML_N_JOBS=4

# The package itself lives inside the venv. Only the example datasets are copied
# out, so `uri: /app/examples/churn.csv` works in a fresh container.
COPY --from=build /opt/venv /opt/venv
COPY --from=build /src/examples /app/examples

# The workspace is a volume in practice; create and chown it so the container
# still starts writably when no volume is mounted.
RUN mkdir -p /var/lib/automl/runs && chown -R automl:automl /var/lib/automl /app

WORKDIR /app
USER automl
VOLUME ["/var/lib/automl"]
EXPOSE 8000

# /api/health reports "degraded" rather than failing when credentials or the
# database are missing, so grep for "ok" instead of trusting the status code.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${AUTOML_API_PORT}/api/health" | grep -q '"status":"ok"' || exit 1

# --factory because create_app() reads settings at construction time.
ENTRYPOINT ["uvicorn", "automl_architect.api.app:create_app", "--factory"]
CMD ["--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
