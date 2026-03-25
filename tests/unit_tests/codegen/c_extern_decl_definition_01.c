// RUN: %cxx -emit-llvm %

typedef struct {
  int x;
  int y;
} point_t;

extern point_t points[];

point_t points[] = {{1, 2}, {3, 4}, {5, 6}};

int main(void) { return points[0].x; }
