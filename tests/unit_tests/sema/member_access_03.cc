// RUN: %cxx -verify -fcheck -freport-missing-types %s

struct X {
  int a[1];
};

struct S {
  struct X x[2];
};

int main() {
  struct S s;
  return s.x->a[0];
}