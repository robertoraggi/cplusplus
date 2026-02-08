// RUN: %cxx -verify -fcheck %s

struct A {
  int x;
};

struct B {
  int y;
};

void test_pointer_to_pointer() {
  A a;
  A* pa = &a;

  B* pb = reinterpret_cast<B*>(pa);
  (void)pb;

  void* pv = reinterpret_cast<void*>(pa);
  (void)pv;

  int i = 42;
  int* pi = &i;
  char* pc = reinterpret_cast<char*>(pi);
  (void)pc;
}

void test_pointer_to_integral() {
  int i = 42;
  int* pi = &i;

  long addr = reinterpret_cast<long>(pi);
  (void)addr;

  int* p2 = reinterpret_cast<int*>(addr);
  (void)p2;
}

void test_reference_reinterpret() {
  int i = 42;

  float& rf = reinterpret_cast<float&>(i);
  (void)rf;
}

void test_identity() {
  int i = 42;
  int* pi = &i;

  int* p2 = reinterpret_cast<int*>(pi);
  (void)p2;
}

void test_value_categories() {
  int i = 42;

  static_assert(__is_reference(decltype(reinterpret_cast<float*>(&i))) ==
                false);

  static_assert(__is_lvalue_reference(decltype(reinterpret_cast<float&>(i))));
}

void test_function_pointers() {
  using F1 = int (*)();
  using F2 = void (*)(int);

  int (*fp1)() = 0;

  F2 fp2 = reinterpret_cast<F2>(fp1);
  (void)fp2;
}
