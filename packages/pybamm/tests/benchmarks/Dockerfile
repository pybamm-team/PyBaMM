# Benchmark image for Bencher Bare Metal. Build context is the repository root:
#   docker build -f packages/pybamm/tests/benchmarks/Dockerfile -t pybamm-benchmarks .
#
# Bencher's shared bare metal runners have no network access, so everything the
# benchmarks touch must already be in this image. Nothing here may resolve a
# dependency, download parameter data, or phone home at run time.
FROM python:3.13-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /bin/uv

# libgomp is the OpenMP runtime the IDAKLU solver links against; the manylinux
# wheel bundles everything else. gcc is needed at *run* time: the solver's
# "compile" option builds CasADi-generated C, and silently falls back to the
# uncompiled path without it, making the compiled benchmarks measure nothing.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Telemetry is disabled because there is no network on the runner to send it to.
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PYTHONIOENCODING=utf-8 \
    MPLBACKEND=Agg \
    PYBAMM_DISABLE_TELEMETRY=true

WORKDIR /pybamm
COPY . .

# Mirrors noxfile.install_locked: install the prebuilt in-repo solver wheel that CI
# drops in solver-wheels/, so the benchmarks measure the solver this commit ships.
# Without that directory (local builds) fall back to the released PyPI wheel — the
# solver submodules are excluded from the build context, so it cannot be compiled
# here. Benchmarks need core pybamm only, so the extras (jax, plotting) stay out.
RUN uv venv "$VIRTUAL_ENV" \
    && set -- -e ./packages/pybamm --group packages/pybamm/pyproject.toml:dev \
    && for wheel in solver-wheels/*-cp313-*.whl; do \
         if [ -e "$wheel" ]; then set -- "$wheel" "$@"; break; fi; \
       done \
    && uv pip install --no-sources "$@"

WORKDIR /pybamm/packages/pybamm/tests/benchmarks

# Fail the build, not the benchmark run, if the image can't import or collect.
RUN pytest -m time_bench --collect-only -q -o addopts= > /dev/null
