// RUN: %cxx -verify -fsyntax-only %s

int NotAScope = 0;

class NotAScope::Inner;  // expected-error {{lookup for 'NotAScope' finds no declaration}}
