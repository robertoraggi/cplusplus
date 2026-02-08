// RUN: %cxx -verify -fcheck %s

struct K {
  auto begin() const -> int { return 0; }
  auto begin() -> int { return 1; }
  auto end() const -> int { return 10; }
};

void test_non_const() {
  K k;
  int a = k.begin();
  int b = k.end();
}

void test_const() {
  const K ck;
  int a = ck.begin();
  int b = ck.end();
}

struct V {
  auto get() volatile -> int { return 1; }
  auto get() -> int { return 2; }
};

void test_volatile() {
  V v;
  int a = v.get();
  volatile V vv;
  int b = vv.get();
}
