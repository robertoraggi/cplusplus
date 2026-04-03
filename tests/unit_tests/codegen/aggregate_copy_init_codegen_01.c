// RUN: %cxx -emit-llvm %s -o - | %filecheck %s

struct Point {
  int x;
  int y;
};

int copy(struct Point p) {
  struct Point q = p;
  return q.x + q.y;
}

// CHECK: call void @llvm.memcpy
