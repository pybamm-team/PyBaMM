# Benchmark image for Bencher Bare Metal. Build context is the repository root:
#   docker build -f packages/pybamm/tests/benchmarks/Dockerfile -t pybamm-benchmarks .
#
# Bencher's shared bare metal runners have no network access, so everything the
# benchmarks touch must already be in this image. Nothing here may resolve a
# dependency, download parameter data, or phone home at run time.
#
# Pushes and every job's pull share a daily registry bandwidth quota: ship only
# what the benchmarks run, with source last so cached dependency layers dedupe.

# Pinned by digest (Dependabot bumps it)
FROM python:3.13-slim-bookworm@sha256:2325bb286ec344af3e5898cc224b5844e2707ac6e26b1632516fd3edc84a5e26 AS base

# libgomp is the OpenMP runtime the IDAKLU solver links against; the manylinux
# wheel bundles everything else. gcc is needed at *run* time: the solver's
# "compile" option builds CasADi-generated C, and silently falls back to the
# uncompiled path without it, making the compiled benchmarks measure nothing.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# No telemetry (the runner has no network) and no baked-in bytecode
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONIOENCODING=utf-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    PYBAMM_DISABLE_TELEMETRY=true \
    UV_NO_CACHE=1

# the final stage takes only the locked requirements and solver
FROM base AS inputs
COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /bin/uv
WORKDIR /src
COPY . .
RUN mkdir -p /inputs/wheel \
    && uv export --frozen --package pybamm --no-default-groups --group bench \
         --no-emit-workspace --no-hashes --output-file /inputs/requirements.txt \
    && tag=$(python -c 'import sys; print("cp%d%d" % sys.version_info[:2])') \
    && for wheel in solver-wheels/*-"$tag"-*.whl; do \
         if [ -e "$wheel" ]; then cp "$wheel" /inputs/wheel/; break; fi; \
       done

FROM base

COPY --from=inputs /inputs/requirements.txt /tmp/requirements.txt
RUN --mount=from=ghcr.io/astral-sh/uv:0.9.7,source=/uv,target=/bin/uv \
    uv venv "$VIRTUAL_ENV" \
    && uv pip install --no-deps --requirement /tmp/requirements.txt

# The in-repo solver wheel CI drops in solver-wheels/, as noxfile.install_locked
# does, so the benchmarks measure the solver this commit ships.
RUN --mount=from=ghcr.io/astral-sh/uv:0.9.7,source=/uv,target=/bin/uv \
    --mount=type=bind,from=inputs,source=/inputs/wheel,target=/tmp/wheel \
    for wheel in /tmp/wheel/*.whl; do \
      if [ -e "$wheel" ]; then uv pip install --no-deps "$wheel"; fi; \
    done

COPY packages/pybamm /pybamm/packages/pybamm
WORKDIR /pybamm/packages/pybamm

# Local builds without the wheel fall back to the PyPI solver (the submodules are
# outside the build context); `uv pip check` rejects a wheel pybamm's pin excludes.
RUN --mount=from=ghcr.io/astral-sh/uv:0.9.7,source=/uv,target=/bin/uv \
    if python -c "import importlib.metadata as m; m.version('pybammsolvers')" 2>/dev/null; then \
      uv pip install --no-deps --editable . && uv pip check; \
    else \
      uv pip install --editable .; \
    fi

WORKDIR /pybamm/packages/pybamm/tests/benchmarks

# Fail the build, not the benchmark run, if the image can't import or collect.
RUN pytest -m time_bench --collect-only -q -o addopts= -p no:cacheprovider > /dev/null
