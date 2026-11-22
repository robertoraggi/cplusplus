// RUN: %cxx --target=wasm32-wasip1 -verify -fsyntax-only %s

auto braced = _Float16{};  // expected-error {{the target provides no _Float16 type}}
auto parenthesized = _Float16();  // expected-error {{the target provides no _Float16 type}}
using Type = decltype(_Float16{});  // expected-error {{the target provides no _Float16 type}}
