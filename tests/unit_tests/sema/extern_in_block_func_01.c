// RUN: %cxx -verify %s
// expected-no-diagnostics

void caller() {
  extern int helper(int x);
  int r = helper(1);
  (void)r;
}

int helper(int x) { return x + 1; }

void caller2() {
  extern int helper(int x);
  int r = helper(2);
  (void)r;
}
