FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

RUN apt-get update && apt-get install -y \
    clang \
    clang-format \
    cmake \
    ninja-build \
    wget \
    python3-pip \
    python3-venv

RUN     wget -nd -P /tmp/ https://github.com/watchexec/watchexec/releases/download/v1.23.0/watchexec-1.23.0-$(uname -m)-unknown-linux-gnu.deb \
    &&  dpkg -i /tmp/watchexec-1.23.0-$(uname -m)-unknown-linux-gnu.deb \
    &&  rm -f /tmp/watchexec-1.23.0-$(uname -m)-unknown-linux-gnu.deb
