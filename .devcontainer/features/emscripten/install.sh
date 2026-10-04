#!/bin/sh

set -e

git clone --depth 1 http://github.com/emscripten-core/emsdk.git /opt/emsdk
cd /opt/emsdk
./emsdk install 6.0.10
./emsdk activate 6.0.10
