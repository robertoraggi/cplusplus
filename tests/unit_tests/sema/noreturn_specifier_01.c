// RUN: %cxx -Wall -verify -fsyntax-only %s
// expected-no-diagnostics

_Noreturn void fail(void);
int _Noreturn abort_now(void);

int after_fail(void) { fail(); }
int after_abort_now(void) { abort_now(); }
