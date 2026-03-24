// RUN: %cxx -emit-llvm %s

void test_basic(int n) {
  int arr[n];
  arr[0] = 42;
}

void test_loop(int n) {
  for (int i = 0; i < n; i++) {
    int tmp[i + 1];
    tmp[0] = i;
  }
}

struct Point {
  int x, y;
};

void test_struct_vla(int n) {
  Point pts[n];
  pts[0].x = 1;
}

int test_sizeof(int n) {
  int arr[n];
  return sizeof(arr);
}

void test_param_1d(int n, int arr[n]) { arr[0] = 0; }

void test_param_2d(int m, int n, int arr[m][n]) { arr[0][0] = 0; }

void test_pass_vla(int n) {
  int arr[n];
  test_param_1d(n, arr);
}

void test_local_2d(int m, int n) {
  int arr[m][n];
  arr[0][0] = 1;
}
