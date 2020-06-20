ARG VARIANT=20.04
FROM mcr.microsoft.com/vscode/devcontainers/base:ubuntu${VARIANT}

RUN apt-get update \
    && export DEBIAN_FRONTEND=noninteractive \
    #
    # install packages
    && apt-get -y install \
        build-essential \
        cmake cppcheck \
        valgrind \
        libgflags-dev \
        libgtest-dev \
        libfmt-dev \
        libboost-dev \
        ninja-build \
    #
    # clean up
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*
