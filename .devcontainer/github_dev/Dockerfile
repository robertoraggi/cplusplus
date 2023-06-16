FROM mcr.microsoft.com/devcontainers/universal:2

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends g++-10 ninja-build \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

ENV CC=/usr/bin/gcc-10
ENV CXX=/usr/bin/g++-10
