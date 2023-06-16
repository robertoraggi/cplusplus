#!/bin/sh

set -e

me=$(dirname "$0")

project_root=$(cd "$me/.." && pwd)

DOCKER_EXTRA_OPTS="--rm -t -v ${project_root}:/code -w /code -u $(id -u) cxx-emsdk"

docker build -t cxx-emsdk -f $project_root/Dockerfile.emsdk ${project_root}

docker run ${DOCKER_EXTRA_OPTS} \
    emcmake cmake -G Ninja \
    -S . \
    -B build.em \
    -DCMAKE_INSTALL_PREFIX=build.em/install/usr \
    -DCMAKE_BUILD_TYPE=MinSizeRel \
    -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=1 \
    -DKWGEN_EXECUTABLE=/usr/bin/kwgen

docker run ${DOCKER_EXTRA_OPTS} \
    cmake --build build.em



