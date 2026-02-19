// RUN: %cxx -verify -fcheck %s

auto v = [:missing_value:]; // expected-error {{invalid splicer expression}}
