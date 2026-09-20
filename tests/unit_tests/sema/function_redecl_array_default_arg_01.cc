// RUN: %cxx -verify -fsyntax-only %s

void g(int a[4] = nullptr);

// expected-error@2 {{redefinition of default argument}}
// expected-note@-3 {{previous definition is here}}
void g(int a[] = nullptr);
