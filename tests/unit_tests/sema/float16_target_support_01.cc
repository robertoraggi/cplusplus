// RUN: %cxx --target=wasm32-wasip1 -verify -fsyntax-only %s
// RUN: %cxx --target=wasm32-unknown-emscripten -verify -fsyntax-only %s

// expected-error@+1 {{the target provides no _Float16 type}}
_Float16 value;

// expected-error@+1 {{the target provides no _Float16 type}}
auto literal = 1.5f16;
