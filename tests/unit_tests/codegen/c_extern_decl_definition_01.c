// RUN: %cxx -emit-llvm %s -o - | %filecheck %s

typedef struct {
  int x;
  int y;
} point_t;

extern point_t points[];

point_t points[] = {{1, 2}, {3, 4}, {5, 6}};

int main(void) { return points[0].x; }

// CHECK:      @points = global [3 x %point_t]
// CHECK-SAME:   [%point_t { i32 1, i32 2 },
// CHECK-SAME:    %point_t { i32 3, i32 4 },
// CHECK-SAME:    %point_t { i32 5, i32 6 }]