// RUN: %cxx -verify -fcheck %s

struct B {
  int x;
};

struct D : B {
  int y;
};

void test_c_style_as_static_cast() {
  int i = (int)3.14;
  (void)i;

  float f = (float)42;
  (void)f;

  // Cast to void
  (void)42;
  (void)3.14;

  // Enum conversions
  enum Color { Red, Green, Blue };
  int c = (int)Red;
  (void)c;
  Color col = (Color)1;
  (void)col;
}

void test_c_style_as_const_cast() {
  const int ci = 42;
  const int* pci = &ci;

  int* pi = (int*)pci;
  (void)pi;
}

void test_c_style_as_reinterpret_cast() {
  int i = 42;
  int* pi = &i;

  long addr = (long)pi;
  (void)addr;

  int* p2 = (int*)addr;
  (void)p2;

  float* pf = (float*)pi;
  (void)pf;
}

void test_c_style_pointer_casts() {
  D d;
  B* pb = &d;

  D* pd = (D*)pb;
  (void)pd;

  void* pv = pb;
  B* pb2 = (B*)pv;
  (void)pb2;
}

void test_c_style_reference_casts() {
  int i = 42;

  const int& cri = i;
  int& ri = (int&)cri;
  (void)ri;
}

void test_c_style_value_categories() {
  int i = 42;
  B b;

  static_assert(__is_reference(decltype((int)3.14)) == false);
  static_assert(__is_lvalue_reference(decltype((int&)i)));

  using F = int();
  int f();
  static_assert(__is_lvalue_reference(decltype((F&&)f)));
}

auto main() -> int {
  test_c_style_as_static_cast();
  test_c_style_as_const_cast();
  test_c_style_as_reinterpret_cast();
  test_c_style_pointer_casts();
  test_c_style_reference_casts();
  test_c_style_value_categories();
  return 0;
}
