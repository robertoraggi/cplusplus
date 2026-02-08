// RUN: %cxx -verify -fcheck %s

struct B {
  int x;
};

struct D : B {
  int y;
};

enum UnscopedEnum { A, B_val, C };
enum class ScopedEnum { X, Y, Z };
enum class ScopedEnum2 : int { P = 1, Q = 2 };

void test_scoped_enum_to_integral() {
  int i = static_cast<int>(ScopedEnum::X);
  (void)i;

  long l = static_cast<long>(ScopedEnum::Y);
  (void)l;

  unsigned u = static_cast<unsigned>(ScopedEnum2::Q);
  (void)u;
}

void test_scoped_enum_to_floating() {
  double d = static_cast<double>(ScopedEnum2::P);
  (void)d;

  float f = static_cast<float>(ScopedEnum::Z);
  (void)f;
}

void test_integral_to_enum() {
  UnscopedEnum e1 = static_cast<UnscopedEnum>(1);
  (void)e1;

  ScopedEnum e2 = static_cast<ScopedEnum>(2);
  (void)e2;

  ScopedEnum2 e3 = static_cast<ScopedEnum2>(1);
  (void)e3;
}

void test_floating_to_enum() {
  ScopedEnum2 e = static_cast<ScopedEnum2>(1.5);
  (void)e;
}

void test_float_to_float() {
  double d = 3.14;
  float f = static_cast<float>(d);
  (void)f;

  long double ld = static_cast<long double>(f);
  (void)ld;
}

void test_pointer_downcast() {
  D d;
  B* pb = &d;
  D* pd = static_cast<D*>(pb);
  (void)pd;

  // With cv-qualifiers
  const B* cpb = &d;
  const D* cpd = static_cast<const D*>(cpb);
  (void)cpd;
}

void test_void_ptr_roundtrip() {
  int i = 42;
  void* pv = &i;
  int* pi = static_cast<int*>(pv);
  (void)pi;

  const void* cpv = &i;
  const int* cpi = static_cast<const int*>(cpv);
  (void)cpi;
}

void test_reference_downcast() {
  D d;
  B& rb = d;
  D& rd = static_cast<D&>(rb);
  (void)rd;

  const B& crb = d;
  const D& crd = static_cast<const D&>(crb);
  (void)crd;
}

void test_cast_to_void() {
  static_cast<void>(42);
  static_cast<void>(3.14);
  static_cast<const void>(42);
  D d;
  static_cast<void>(d);
}

void test_standard_conversions() {
  int i = static_cast<int>(3.14);
  (void)i;

  double d = static_cast<double>(42);
  (void)d;

  bool b = static_cast<bool>(42);
  (void)b;
}
