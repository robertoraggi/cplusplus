// RUN: %cxx -Wall -verify -fsyntax-only %s
// expected-no-diagnostics

_Noreturn void fail();
int _Noreturn abort_now();

void redeclared_later();
_Noreturn void redeclared_later();
void redeclared_later();

int after_fail() { fail(); }
int after_abort_now() { abort_now(); }
int after_redeclared_later() { redeclared_later(); }
