// RUN: %cxx -verify -fsyntax-only %s

auto v = [:missing_value:]; // expected-error {{use of undeclared identifier 'missing_value'}}
