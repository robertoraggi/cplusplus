// RUN: %cxx -verify %s

// clang-format off

// expected-no-diagnostics

struct Base {
  int x;
  void f();
};

struct Derived : Base {
  int y;
};

void test_nullptr_to_data_member_ptr() {
  int Base::* p = nullptr;
}

void test_zero_to_data_member_ptr() {
  int Base::* p = 0;
}

void test_nullptr_to_member_fn_ptr() {
  void (Base::* fp)() = nullptr;
}

void test_base_data_member_to_derived() {
  int Base::* pb = &Base::x;
  int Derived::* pd = pb;
}
