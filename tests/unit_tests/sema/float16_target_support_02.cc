// RUN: %cxx --target=aarch64-linux-gnu -verify -fsyntax-only %s
// RUN: %cxx --target=x86_64-linux-gnu -verify -fsyntax-only %s
// RUN: %cxx --target=arm64-apple-macosx -verify -fsyntax-only %s
// RUN: %cxx --target=x86_64-pc-windows-msvc -verify -fsyntax-only %s
// RUN: %cxx --target=spirv64-unknown-unknown -verify -fsyntax-only %s

// expected-no-diagnostics

_Float16 value;
auto literal = 1.5f16;
