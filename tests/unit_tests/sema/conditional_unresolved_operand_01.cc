// RUN: %cxx -verify -fsyntax-only %s

int x = true ? missing_value : 1; // expected-error {{use of undeclared identifier 'missing_value'}}
