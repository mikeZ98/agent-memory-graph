# agent-memory-graph: hardened, unprivileged, multi-stage image.
#
#   docker build -t agent-memory-graph:local .
#   docker run --rm --read-only --tmpfs /tmp:rw,noexec,nosuid,nodev \
#     --cap-drop ALL --security-opt no-new-privileges:true -p 127.0.0.1:8000:8000 \
#     -e AMG_NEO4J_PASSWORD=... agent-memory-graph:local
#
# Provider SDKs for AMG_LLM_MODEL are opt-in extras: --build-arg AMG_EXTRAS=anthropic
# Every base image is pinned by digest; bump tag and digest together.

ARG PYTHON_IMAGE=python:3.12.13-slim-trixie@sha256:229a2c5bfa27522db7815ea81f9bed70af17ccb9de9fc7ad142b1877b5830d36
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.7@sha256:240fb85ab0f263ef12f492d8476aa3a2e4e1e333f7d67fbdd923d00a506a516a

# --------------------------------------------------------------------------
# uv binary only (never shipped in the runtime image)
# --------------------------------------------------------------------------
FROM ${UV_IMAGE} AS uv

# --------------------------------------------------------------------------
# Builder: resolve the locked dependency set into an isolated virtualenv
# --------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS builder

ARG AMG_EXTRAS=""
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3.12 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_FROZEN=1

COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /src

# Dependencies first (cached layer), then the project itself; never editable.
COPY pyproject.toml uv.lock README.md LICENSE ./
# ${extras} is a deliberately word-split list of "--extra NAME" flags.
# hadolint ignore=SC2086
RUN set -eu; \
    extras=""; \
    for extra in ${AMG_EXTRAS}; do extras="${extras} --extra ${extra}"; done; \
    uv sync --no-dev --no-install-project --no-editable ${extras}

COPY src ./src
# hadolint ignore=SC2086
RUN set -eu; \
    extras=""; \
    for extra in ${AMG_EXTRAS}; do extras="${extras} --extra ${extra}"; done; \
    uv sync --no-dev --no-editable ${extras}; \
    /opt/venv/bin/python -c "import agent_memory_graph, uvicorn"

# --------------------------------------------------------------------------
# Runtime: the same CPython, minus every way to install or escalate
# --------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS runtime

ARG VERSION=0.1.0
ARG REVISION=unknown
ARG CREATED=unknown

LABEL org.opencontainers.image.title="agent-memory-graph" \
      org.opencontainers.image.description="Asynchronous GraphRAG long-term memory engine for AI agents (Neo4j + Qdrant)." \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/mikeZ98/agent-memory-graph" \
      org.opencontainers.image.url="https://github.com/mikeZ98/agent-memory-graph" \
      org.opencontainers.image.authors="mikeZ98" \
      org.opencontainers.image.vendor="mikeZ98" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}" \
      org.opencontainers.image.created="${CREATED}" \
      org.opencontainers.image.base.name="docker.io/library/python:3.12.13-slim-trixie"

# 1. static unprivileged identity; 2. strip setuid/setgid bits; 3. remove pip and the
#    apt/dpkg toolchain (with their state) so nothing can be installed at runtime.
RUN set -eu; \
    groupadd --system --gid 10001 appuser; \
    useradd --system --uid 10001 --gid 10001 --no-create-home \
        --home-dir /nonexistent --shell /usr/sbin/nologin appuser; \
    find / -xdev -type f -perm /6000 -exec chmod a-s {} +; \
    rm -rf /usr/local/lib/python3.12/site-packages/pip \
           /usr/local/lib/python3.12/site-packages/pip-* \
           /usr/local/bin/pip /usr/local/bin/pip3 /usr/local/bin/pip3.12 \
           /usr/local/lib/python3.12/ensurepip \
           /usr/share/doc /usr/share/man /usr/share/info; \
    rm -rf /etc/apt /var/lib/apt /var/cache/apt /var/log/apt \
           /usr/lib/apt /usr/share/keyrings \
           /usr/bin/apt /usr/bin/apt-* /usr/sbin/apt-* \
           /usr/bin/dpkg /usr/bin/dpkg-* /usr/sbin/dpkg-* \
           /var/lib/dpkg /var/cache/debconf /etc/dpkg /usr/share/dpkg

# Root-owned and therefore immutable for appuser; bytecode precompiled at build time.
COPY --from=builder --chown=root:root /opt/venv /opt/venv

ENV PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp \
    TMPDIR=/tmp \
    XDG_CACHE_HOME=/tmp/.cache

WORKDIR /tmp
USER 10001:10001
EXPOSE 8000
STOPSIGNAL SIGTERM

HEALTHCHECK --interval=15s --timeout=3s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"]

# Exec form, no shell: python is PID 1 and receives SIGTERM directly.
ENTRYPOINT ["python", "-m", "agent_memory_graph"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
