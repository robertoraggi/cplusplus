// RUN: %cxx --target=spirv64-unknown-unknown -verify -fsyntax-only %s

auto braced = __int128{};  // expected-error {{the target provides no 128-bit extended integer type}}
auto parenthesized = __int128();  // expected-error {{the target provides no 128-bit extended integer type}}
using Type = decltype(__int128{});  // expected-error {{the target provides no 128-bit extended integer type}}
