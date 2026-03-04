// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

int compute() { return 2; }

int test_switch_condition_decl() {
  switch (int x = compute()) {
    case 1:
      return x + 1;
    case 2:
      return x + 2;
    default:
      return x;
  }
}
