// RUN: %cxx -verify -fcheck %s

void g(int a[4] = nullptr);

// expected-error@1 {{redefinition of default argument}}
void g(int a[] = nullptr);
