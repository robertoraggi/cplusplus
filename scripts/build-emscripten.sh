#!/bin/bash

set -e

me=$(dirname "$0")

project_root=$(cd "$me/.." && pwd)

CMAKE_CONFIGURE_OPTIONS="
-DCMAKE_INSTALL_PREFIX=build.em/install/usr \
-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-MinSizeRel} \
-DCXX_INTERPROCEDURAL_OPTIMIZATION=${CMAKE_INTERPROCEDURAL_OPTIMIZATION:-1} \
-DKWGEN_EXECUTABLE=${KWGEN_EXECUTABLE:-/usr/bin/kwgen} \
-DFLATBUFFERS_FLATC_EXECUTABLE=${FLATBUFFERS_FLATC_EXECUTABLE:-/usr/bin/flatc}"

if [ ! -z "${CODESPACES}" ] && [ ! -z "${EMSDK}" ]; then
    if [ ! -d "${project_root}/build.em" ]; then
        emcmake cmake -G Ninja ${CMAKE_CONFIGURE_OPTIONS} -S ${project_root} -B ${project_root}/build.em
    fi
    cmake --build $project_root/build.em
    exit 0
fi

# build cache
mkdir -p $HOME/.emscripten-cache

docker build -t cxx-emsdk -f $project_root/Dockerfile.emsdk ${project_root}

docker run -t --rm -u $(id -u) -v $HOME/.emscripten-cache:/emsdk/upstream/emscripten/cache/ cxx-emsdk embuilder.py build MINIMAL --lto=thin

DOCKER_EXTRA_OPTS="--rm -t -v $HOME/.emscripten-cache:/emsdk/upstream/emscripten/cache/ -v ${project_root}:/code -w /code -u $(id -u) cxx-emsdk"

docker run ${DOCKER_EXTRA_OPTS} \
    emcmake cmake -G Ninja ${CMAKE_CONFIGURE_OPTIONS} -S . -B build.em

docker run ${DOCKER_EXTRA_OPTS} \
    cmake --build build.em



