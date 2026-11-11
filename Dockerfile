# syntax=docker/dockerfile:1
#
# open-rlcd server image. Guide: docs/deployment.md
#
#   docker build -t open-rlcd .                                   core only (CPU, no torch, no ONNX)
#   docker build -t open-rlcd --build-arg RLCD_EXTRAS=documents . + onnxruntime, faiss, tokenizers, pypdf
#   docker run --rm -p 127.0.0.1:8095:8095 -e RLCD_API_KEY=... open-rlcd
#
# With nothing mounted it serves the bundled synthetic demo model, like `uv run rlcd serve` after a clone.
# A real deployment mounts a project directory (rlcd.yaml) read-only and a volume for RLCD_HOME:
#
#   docker run -d -p 127.0.0.1:8095:8095 -e RLCD_API_KEY -e RLCD_ADMIN_KEY \
#     -e RLCD_PROJECT=/project -v /srv/acme:/project:ro -v rlcd-home:/var/lib/rlcd open-rlcd
#
# The image holds the code, its locked dependencies and the synthetic demo bundle. No keys, no trained
# model of yours, no data: those arrive at run time through the environment and the two mounts.

ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.10.12

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# ---------------------------------------------------------------------------- build: a venv pinned by uv.lock
FROM python:${PYTHON_VERSION}-slim-bookworm AS build
COPY --from=uv /uv /usr/local/bin/uv

# Space-separated extras of pyproject.toml, e.g. "documents" or "documents qdrant". Empty: core only.
ARG RLCD_EXTRAS=""

ENV UV_PROJECT_ENVIRONMENT=/opt/rlcd \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_NO_DEV=1

WORKDIR /src

# Dependencies first, exactly as locked: this layer is rebuilt only when uv.lock or pyproject.toml changes.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    set -eu; flags=""; for e in ${RLCD_EXTRAS}; do flags="${flags} --extra ${e}"; done; \
    uv sync --locked --no-install-project ${flags}

# The project itself, installed as a wheel (not editable): the wheel build copies bundles/demo-documents to
# open_rlcd/_demo/demo-documents (pyproject.toml force-include), which is where `rlcd serve` looks for the demo.
COPY pyproject.toml uv.lock README.md LICENSE NOTICE ./
COPY open_rlcd ./open_rlcd
COPY bundles/demo-documents ./bundles/demo-documents
RUN --mount=type=cache,target=/root/.cache/uv \
    set -eu; flags=""; for e in ${RLCD_EXTRAS}; do flags="${flags} --extra ${e}"; done; \
    uv sync --locked --no-editable ${flags}; \
    test -f "$(cd / && /opt/rlcd/bin/python -I -c 'import open_rlcd, pathlib; print(pathlib.Path(open_rlcd.__file__).parent)')/_demo/demo-documents/model.json"

# ---------------------------------------------------------------------------- runtime
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ARG RLCD_EXTRAS=""
LABEL org.opencontainers.image.title="open-rlcd" \
      org.opencontainers.image.description="Open RLCD decision-model service (rlcd serve)" \
      org.opencontainers.image.source="https://github.com/f8fvwgzc/open-rlcd" \
      org.opencontainers.image.licenses="MIT" \
      io.open-rlcd.extras="${RLCD_EXTRAS}"

# libgomp1: the OpenMP runtime LightGBM loads. uid/gid 10001: a fixed number, so a bind-mounted home can be
# chown'ed to it on the host. /var/lib/rlcd is created owned by that user so a fresh named volume inherits it.
RUN set -eu; \
    apt-get update; \
    apt-get install -y --no-install-recommends libgomp1; \
    rm -rf /var/lib/apt/lists/*; \
    groupadd --system --gid 10001 rlcd; \
    useradd --system --uid 10001 --gid 10001 --home-dir /var/lib/rlcd --no-create-home --shell /usr/sbin/nologin rlcd; \
    install -d -o 10001 -g 10001 -m 0750 /var/lib/rlcd; \
    install -d -o root -g root -m 0755 /srv/rlcd

COPY --from=build /opt/rlcd /opt/rlcd
COPY --chmod=0755 deploy/entrypoint.sh /usr/local/bin/rlcd-entrypoint

# RLCD_HOME: everything the service persists. HF_HOME: keeps the Hugging Face client's own files (documents
# extra) inside the volume too, so the root filesystem can be read-only.
ENV PATH=/opt/rlcd/bin:$PATH \
    RLCD_HOME=/var/lib/rlcd \
    HF_HOME=/var/lib/rlcd/cache/huggingface \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER 10001:10001
# An empty, root-owned directory with no rlcd.yaml above it: "no project" unless RLCD_PROJECT says otherwise.
WORKDIR /srv/rlcd
VOLUME ["/var/lib/rlcd"]
EXPOSE 8095

# /health needs no key. It answers as soon as the registry is readable; it does not load the encoder.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --start-interval=2s --retries=3 \
    CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:8095/health', timeout=4).read()"]

# `docker run IMAGE` serves; `docker run IMAGE receipts verify` (or any other rlcd subcommand) runs the CLI.
ENTRYPOINT ["rlcd-entrypoint"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8095"]
