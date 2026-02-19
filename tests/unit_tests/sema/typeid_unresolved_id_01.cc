// RUN: %cxx -verify -fcheck %s

auto m = typeid(missing_identifier); // expected-error {{use of undeclared identifier 'missing_identifier'}}
