// RUN: %cxx -verify -fsyntax-only %s

// clang-format off

// expected-no-diagnostics

struct Base {
  int x;
  void f();
  void g() noexcept;
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

void test_base_member_fn_to_derived() {
  void (Base::* pb)() = &Base::f;
  void (Derived::* pd)() = pb;
}

void test_noexcept_member_fn_to_derived() {
  void (Base::* pb)() noexcept = &Base::g;
  void (Derived::* pd)() = pb;
}

void test_added_pointee_const() {
  int Base::* pb = &Base::x;
  const int Derived::* pd = pb;
}
