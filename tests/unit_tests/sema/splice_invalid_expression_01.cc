// RUN: %cxx -verify %s

auto v = [:missing_value:]; // expected-error {{invalid splicer expression}}
