// RUN: %cxx -emit-mlir -c %s

int sum_vla(int n) {
  int arr[n];
  for (int i = 0; i < n; i++) arr[i] = i;
  int s = 0;
  for (int i = 0; i < n; i++) s += arr[i];
  return s;
}

typedef struct {
  int x;
  int y;
} Point;

void vla_structs(int n) {
  Point pts[n];
  pts[0].x = 1;
  pts[0].y = 2;
}

void multi_vla(int a, int b) {
  int x[a];
  int y[b];
  x[0] = 1;
  y[0] = 2;
}

void fill_1d(int n, int arr[n]) {
  for (int i = 0; i < n; i++) arr[i] = i;
}

void fill_2d(int m, int n, int arr[m][n]) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      arr[i][j] = i * n + j;
    }
  }
}

void local_2d(int m, int n) {
  int arr[m][n];
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      arr[i][j] = i * n + j;
    }
  }
}
