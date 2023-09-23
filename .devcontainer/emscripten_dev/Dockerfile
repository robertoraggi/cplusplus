FROM emscripten/emsdk:3.1.46

RUN wget -nd -P /tmp/ https://github.com/watchexec/watchexec/releases/download/v1.23.0/watchexec-1.23.0-x86_64-unknown-linux-gnu.deb \
    && dpkg -i /tmp/watchexec-1.23.0-x86_64-unknown-linux-gnu.deb \
    && rm -f /tmp/watchexec-1.23.0-x86_64-unknown-linux-gnu.deb

RUN apt-get update && apt-get install -y \
    bash-completion \
    ninja-build \
    wget \
    unzip

RUN echo emscripten ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/emscripten && \
    chmod 0440 /etc/sudoers.d/emscripten

RUN wget https://github.com/google/flatbuffers/releases/download/v23.5.26/Linux.flatc.binary.g++-10.zip && \
    unzip Linux.flatc.binary.g++-10.zip && \
    mv flatc /usr/bin/flatc && \
    rm Linux.flatc.binary.g++-10.zip

USER emscripten

RUN wget -q -O - https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.5/install.sh | bash  && \
    . /home/emscripten/.nvm/nvm.sh && \
    nvm install node && \
    nvm use node
