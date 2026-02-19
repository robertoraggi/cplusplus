// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

struct S {
  float f;
};

float a{0};
float b{16777216};
float arr[2] = {0, 1};
S s{1};
