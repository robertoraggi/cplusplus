FROM ubuntu:23.10

RUN apt-get update && apt-get install -y \
    bash-completion \
    build-essential \
    clang-17 \
    clang-format-17 \
    cmake \
    ninja-build \
    wget \
    python3-pip

RUN     wget -nd -P /tmp/ https://github.com/watchexec/watchexec/releases/download/v1.23.0/watchexec-1.23.0-$(uname -m)-unknown-linux-gnu.deb \
    &&  dpkg -i /tmp/watchexec-1.23.0-$(uname -m)-unknown-linux-gnu.deb \
    &&  rm -f /tmp/watchexec-1.23.0-$(uname -m)-unknown-linux-gnu.deb

RUN     update-alternatives --install /usr/bin/clang-format clang-format /usr/bin/clang-format-17 100 \
    &&  update-alternatives --install /usr/bin/clang clang /usr/bin/clang-17 100
