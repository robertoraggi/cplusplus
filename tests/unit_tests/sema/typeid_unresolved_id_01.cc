// RUN: %cxx -verify %s

auto m = typeid(missing_identifier); // expected-error {{use of undeclared identifier 'missing_identifier'}}
