// RUN: %cxx -toolchain macos -verify %s

void foo();       // expected-note {{previous declaration of 'foo' is here}}
void foo(int x);  // expected-error {{conflicting types for 'foo'}}

void bar(int x);    // expected-note {{previous declaration of 'bar' is here}}
void bar(float y);  // expected-error {{conflicting types for 'bar'}}

// Compatible redeclarations should be fine.
void baz(int x);
void baz(int x);
