// RUN: %cxx -verify -fcheck %s

struct A {
  struct {
    int x;
  };
  union {
    int y;
    struct {
      int z;
    };
  };
};

int main() {
  struct A a;
  (void)_Generic(a.x, int: 123);
  (void)_Generic(a.y, int: 321);
  (void)_Generic(a.z, int: 444);
}