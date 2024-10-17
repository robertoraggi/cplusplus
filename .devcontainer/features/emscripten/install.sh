#!/bin/sh

set -e

git clone --depth 1 http://github.com/emscripten-core/emsdk.git /opt/emsdk
cd /opt/emsdk
./emsdk install latest
./emsdk activate latest
