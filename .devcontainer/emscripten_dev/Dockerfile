FROM emscripten/emsdk:3.1.42

RUN apt-get update && apt-get install -y \
    ninja-build \
    wget \
    unzip

RUN echo emscripten ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/emscripten && \
    chmod 0440 /etc/sudoers.d/emscripten

RUN wget https://github.com/google/flatbuffers/releases/download/v23.5.26/Linux.flatc.binary.g++-10.zip && \
    unzip Linux.flatc.binary.g++-10.zip && \
    mv flatc /usr/bin/flatc && \
    rm Linux.flatc.binary.g++-10.zip
