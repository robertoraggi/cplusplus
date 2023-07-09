#!/bin/sh

set -e

me=$(dirname "$0")

project_root=$(cd "$me/.." && pwd)

DOCKER_EXTRA_OPTS="--rm -t -v ${project_root}:/code -w /code -u $(id -u) cxx-wasi"

docker build -t cxx-wasi -f ${project_root}/Dockerfile.wasi ${project_root}

docker run ${DOCKER_EXTRA_OPTS} \
    cmake -G Ninja \
    -S . \
    -B build.wasi \
    -DCMAKE_INSTALL_PREFIX=build.wasi/install/usr \
    -DCMAKE_TOOLCHAIN_FILE=/usr/share/cmake/wasi-sdk.cmake \
    -DCMAKE_BUILD_TYPE=MinSizeRel \
    -DCXX_INTERPROCEDURAL_OPTIMIZATION=1 \
    -DKWGEN_EXECUTABLE=/usr/bin/kwgen \
    -DFLATBUFFERS_FLATC_EXECUTABLE=/usr/bin/flatc \
    -DCXX_INSTALL_WASI_SYSROOT=ON \

docker run ${DOCKER_EXTRA_OPTS} \
    cmake --build build.wasi --target install


