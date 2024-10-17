#!/bin/bash

set -e

arch=$(uname -m)

if [ "$arch" == "aarch64" ]; then
    arch="arm64"
fi

wget -nd -P /tmp/ https://github.com/WebAssembly/wasi-sdk/releases/download/wasi-sdk-24/wasi-sdk-24.0-${arch}-linux.deb

dpkg -i /tmp/wasi-sdk-24.0-${arch}-linux.deb
