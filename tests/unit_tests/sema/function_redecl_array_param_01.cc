// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

void f(int a[4]);
void f(int a[]);

int arr[4];

void test() { f(arr); }
