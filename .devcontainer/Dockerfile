# Use Python 3.12 base image (NautilusTrader supports 3.11-3.13)
FROM python:3.12-slim

# Install system dependencies required for NautilusTrader
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    clang \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain (required for NautilusTrader)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install uv (NautilusTrader's preferred package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

# Create a non-root user
RUN useradd -m -s /bin/bash vscode

# Set working directory
WORKDIR /workspaces/nt

# Switch to non-root user
USER vscode

# Set environment variables for uv and Rust
ENV UV_CACHE_DIR=/tmp/uv-cache
ENV UV_PYTHON_PREFERENCE=only-managed
ENV PATH="/home/vscode/.cargo/bin:${PATH}"

# Install Rust toolchain for non-root user
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y 