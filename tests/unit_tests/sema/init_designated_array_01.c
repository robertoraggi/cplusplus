// RUN: %cxx -verify -toolchain macos %s

void test_basic(void) {
  int x[] = {[2] = 1, [4] = 2, [6] = 3};
  _Static_assert(sizeof(x) == 7 * sizeof(int));
}

void test_mixed(void) {
  int y[] = {1, 2, [5] = 3};
  _Static_assert(sizeof(y) == 6 * sizeof(int));
}

void test_fixed(void) { int z[10] = {[9] = 1}; }

struct Point {
  int x;
  int y;
};

void test_struct(void) { struct Point p = {.x = 1, .y = 2}; }
