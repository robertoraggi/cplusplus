// RUN: %cxx -verify -fcheck %s

int x;

// expected-error@1 {{conflicting declaration of 'x'}}
double x;

int main() { return 0; }
