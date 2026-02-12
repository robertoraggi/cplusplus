// RUN: %cxx -verify -fcheck %s

void f(int x = 1);

// expected-error@1 {{redefinition of default argument}}
void f(int x = 2);

void g(int y = 3);
void g(int y);

int main() { return 0; }
