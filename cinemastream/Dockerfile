# cinemastream/Dockerfile
# Production container for the CinemaStream data pipeline
# Upgraded to multi-stage in Ch 021a (was single-stage in Ch 021)
# Before: ~870MB single-stage  |  After: ~183MB multi-stage runtime
#
# Build: docker build -t cinemastream-pipeline:latest .
# Run:   docker run --rm -e OPENEXCHANGE_APP_ID=key cinemastream-pipeline:latest


# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# gcc needed for psycopg2 (PostgreSQL driver with C extension)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file first (layer cache optimisation from Ch 021)
COPY pyproject.toml .

# Install into /install — not system Python — so we can COPY it cleanly next stage
RUN pip install --no-cache-dir --prefix=/install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -e .


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="data@cinemastream.example.com" \
      version="0.2.0" \
      description="CinemaStream data pipeline — multi-stage, non-root"

WORKDIR /app

# Copy ONLY the installed packages from the builder stage
# gcc and libpq-dev are NOT included — they stayed in the builder
COPY --from=builder /install /usr/local

# Security: non-root user — required by enterprise K8s security policies
RUN groupadd -r cinemastream && useradd -r -g cinemastream cinemastream

# Copy application code
COPY cinemastream/ ./cinemastream/
COPY scripts/ ./scripts/

# Create data and log directories with correct ownership
RUN mkdir -p /app/data /app/logs && \
    chown -R cinemastream:cinemastream /app

# Switch to non-root user
USER cinemastream

# NEVER put secrets here — pass them at runtime via -e or K8s secretKeyRef
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python3 -c "import cinemastream; print('OK')" || exit 1

# Default command
CMD ["python3", "scripts/load_orders.py", \
     "--input", "/app/data/raw_orders.csv", \
     "--output", "/app/data/clean_orders.csv"]
