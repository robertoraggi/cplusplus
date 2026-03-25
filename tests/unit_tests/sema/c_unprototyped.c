// RUN: %cxx -toolchain macos -fno-strict-prototypes -verify %s
// expected-no-diagnostics

void foo();
void foo(int x);

void bar();
void bar(int x, float y);

void baz(int x);
void baz(int x) {}
