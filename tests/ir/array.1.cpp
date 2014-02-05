const int N = 10;

void foo(double a[N][N], int index[N], double value[N]) {
  int i; int j;
  double x;
  i = 0;
  while (i < N) {
    index[i] = 0;
    x = a[i][0];
    if (x < 0)
      x = x * -1;
    value[i] = x;
    j = 1;
    while (j < N) {
      x = a[i][j];
      if (x < 0)
        x = x * -1;
      if (x > value[i]) {
        index[i] = j;
        value[i] = x;
      }
      j = j + 1;
    }
    i = i + 1;
  }
}
