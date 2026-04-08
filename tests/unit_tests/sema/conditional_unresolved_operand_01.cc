// RUN: %cxx -verify %s

int x = true ? missing_value : 1; // expected-error {{use of undeclared identifier 'missing_value'}}
