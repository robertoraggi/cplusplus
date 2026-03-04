// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

struct Base {
  int kind;
};

struct Derived : Base {
  int value;
  static Derived* from(Base* b) {
    if (b && b->kind == 1) return static_cast<Derived*>(b);
    return nullptr;
  }
};

int test_if_condition_decl(Base* b) {
  if (Derived* d = Derived::from(b)) {
    return d->value;
  }
  return 0;
}

int test_if_condition_decl_else(Base* b) {
  if (Derived* d = Derived::from(b)) {
    return d->value;
  } else {
    // d should also be visible in the else branch
    (void)d;  // d is null here
    return -1;
  }
}

int test_if_condition_bool() {
  if (int x = 42) {
    return x;
  }
  return 0;
}
