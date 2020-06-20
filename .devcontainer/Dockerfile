ARG VARIANT=20.04
FROM mcr.microsoft.com/vscode/devcontainers/base:ubuntu${VARIANT}

ARG USERNAME=workspace
ARG USER_UID=1000
ARG USER_GID=$USER_UID

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
    # update UID/GID
    && if [ "$USER_GID" != "1000" ] || [ "$USER_UID" != "1000" ]; then \
        groupmod --gid $USER_GID $USERNAME \
        && usermod --uid $USER_UID --gid $USER_GID $USERNAME \
        && chown -R $USER_UID:$USER_GID /home/$USERNAME; \
    fi \
    #
    # clean up
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*
