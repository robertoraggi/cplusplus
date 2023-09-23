FROM mcr.microsoft.com/devcontainers/universal:2

RUN wget -nd -P /tmp/ https://github.com/watchexec/watchexec/releases/download/v1.23.0/watchexec-1.23.0-x86_64-unknown-linux-musl.deb \
    && dpkg -i /tmp/watchexec-1.23.0-x86_64-unknown-linux-musl.deb \
    && rm -f /tmp/watchexec-1.23.0-x86_64-unknown-linux-musl.deb

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends g++-10 ninja-build \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

ENV CC=/usr/bin/gcc-10
ENV CXX=/usr/bin/g++-10
