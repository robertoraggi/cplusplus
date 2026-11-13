// RUN: %cxx -std=c++26 -verify -fsyntax-only %s

auto empty = []<>() {}; // expected-error {{expected a template parameter}}
