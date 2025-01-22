FROM ubuntu:24.10

RUN userdel -f -r ubuntu

RUN apt-get update && apt-get install -y \
    bash-completion \
    build-essential \
    clang-format \
    clangd \
    cmake \
    lldb \
    ninja-build \
    pipx \
    python3-pip \
    python3-venv \
    valgrind \
    wget

RUN     wget -nd -P /tmp/ https://github.com/watchexec/watchexec/releases/download/v2.2.0/watchexec-2.2.0-$(uname -m)-unknown-linux-gnu.deb \
    &&  dpkg -i /tmp/watchexec-2.2.0-$(uname -m)-unknown-linux-gnu.deb \
    &&  rm -f /tmp/watchexec-2.2.0-$(uname -m)-unknown-linux-gnu.deb

RUN    wget -nd -P /tmp/ https://github.com/bytecodealliance/wasmtime/releases/download/v25.0.2/wasmtime-v25.0.2-$(uname -m)-linux.tar.xz \
    && tar -C /usr/local/bin -xvf /tmp/wasmtime-v25.0.2-$(uname -m)-linux.tar.xz --strip-components=1 --wildcards 'wasmtime-*-linux/wasmtime' \
    && rm -f /tmp/wasmtime-v25.0.2-$(uname -m)-linux.tar.xz
