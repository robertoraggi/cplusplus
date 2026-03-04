// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

template <class T>
T deref(const T* p) {
  return *p;
}

const int ci = 10;
int i = 20;
long l = 30;

static_assert(__is_same(decltype(deref(&ci)), int));
static_assert(__is_same(decltype(deref(&i)), int));
static_assert(__is_same(decltype(deref(&l)), long));

template <class T>
T deref_volatile(volatile T* p) {
  return *p;
}

volatile int vi = 5;
static_assert(__is_same(decltype(deref_volatile(&vi)), int));
