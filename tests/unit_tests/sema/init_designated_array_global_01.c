// RUN: %cxx -verify -toolchain macos %s

int global_basic[] = {[2] = 1, [4] = 2, [6] = 3};
_Static_assert(sizeof(global_basic) == 7 * sizeof(int),
               "global size deduction");

int global_mixed[] = {1, 2, [5] = 3};
_Static_assert(sizeof(global_mixed) == 6 * sizeof(int),
               "global mixed deduction");

int global_fixed[10] = {[9] = 1};

void test_static(void) {
  static int s[] = {[2] = 1, [4] = 2, [6] = 3};
  _Static_assert(sizeof(s) == 7 * sizeof(int), "static size deduction");
}
