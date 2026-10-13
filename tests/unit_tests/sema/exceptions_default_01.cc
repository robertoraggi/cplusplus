// RUN: %cxx -fsyntax-only %s
// RUN: %cxx -fno-exceptions -fsyntax-only %s
// RUN: %cxx -fexceptions -fno-exceptions -fsyntax-only %s
// RUN: %cxx -fexceptions -fsyntax-only -DEXCEPTIONS %s
// RUN: %cxx -fno-exceptions -fexceptions -fsyntax-only -DEXCEPTIONS %s

// RUN: %cxx -toolchain wasm32 -fsyntax-only %s
// RUN: %cxx --target=wasm32-wasip1 -fsyntax-only %s
// RUN: %cxx --target=wasm32-wasip2 -fsyntax-only %s
// RUN: %cxx --target=wasm32-unknown-emscripten -fsyntax-only %s
// RUN: %cxx --target=aarch64-linux-gnu -fsyntax-only -DEXCEPTIONS %s
// RUN: %cxx --target=x86_64-linux-gnu -fsyntax-only -DEXCEPTIONS %s
// RUN: %cxx --target=arm64-apple-macosx -fsyntax-only -DEXCEPTIONS %s
// RUN: %cxx --target=aarch64-linux-gnu -fno-exceptions -fsyntax-only %s
// RUN: %cxx --target=arm64-apple-macosx -fno-exceptions -fsyntax-only %s

#ifdef EXCEPTIONS
#ifndef __cpp_exceptions
#error exceptions must be enabled
#endif
#else
#ifdef __cpp_exceptions
#error exceptions must be disabled
#endif
#endif
