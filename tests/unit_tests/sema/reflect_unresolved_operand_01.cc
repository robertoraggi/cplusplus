// RUN: %cxx -verify -fcheck %s

auto r = ^^missing_symbol; // expected-error {{use of undeclared identifier 'missing_symbol'}}
