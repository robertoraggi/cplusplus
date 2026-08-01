// RUN: %cxx -verify -fcheck %s

auto v = [:missing_value:]; // expected-error {{use of undeclared identifier 'missing_value'}}
