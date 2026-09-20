// RUN: %cxx -verify -fsyntax-only %s

void f(int x = 1);

// expected-error@2 {{redefinition of default argument}}
// expected-note@-3 {{previous definition is here}}
void f(int x = 2);

void g(int y = 3);
void g(int y);

int main() { return 0; }
