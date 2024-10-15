FROM ubuntu:24.10

RUN apt-get update && apt-get install -y \
    bash-completion \
    build-essential \
    clang-format \
    clangd \
    cmake \
    ninja-build \
    python3-pip \
    python3-venv \
    wget

RUN     wget -nd -P /tmp/ https://github.com/watchexec/watchexec/releases/download/v2.2.0/watchexec-2.2.0-$(uname -m)-unknown-linux-gnu.deb \
    &&  dpkg -i /tmp/watchexec-2.2.0-$(uname -m)-unknown-linux-gnu.deb \
    &&  rm -f /tmp/watchexec-2.2.0-$(uname -m)-unknown-linux-gnu.deb
