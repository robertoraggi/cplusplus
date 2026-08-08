// RUN: %cxx -verify -fsyntax-only %s

int NotAScope = 0;

class NotAScope::Inner;  // expected-error {{nested name specifier must be a class or namespace}}
