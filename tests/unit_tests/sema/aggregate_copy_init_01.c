// RUN: %cxx -verify %s
// expected-no-diagnostics

struct Point {
  int x;
  int y;
};

int main(void) {
  struct Point p = {1, 2};
  struct Point q = p;
  return q.x + q.y - 3;
}
