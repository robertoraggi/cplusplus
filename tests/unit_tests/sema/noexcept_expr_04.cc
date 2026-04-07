// RUN: %cxx -verify %s

struct S {
  S& operator+=(const S&) noexcept;
};

S s1, s2;
static_assert(noexcept(s1 += s2));

struct SThrow {
  SThrow& operator+=(const SThrow&);
};

SThrow st1, st2;
static_assert(!noexcept(st1 += st2));

struct V {
  int& operator[](int) noexcept;
};

V v;
static_assert(noexcept(v[0]));

struct VThrow {
  int& operator[](int);
};

VThrow vt;
static_assert(!noexcept(vt[0]));

struct WithThrowingDtor {
  ~WithThrowingDtor() noexcept(false);
};

static_assert(!noexcept(delete(WithThrowingDtor*)nullptr));

static_assert(noexcept(static_cast<int>(1.0)));

struct Base {
  virtual ~Base();
};

struct Derived : Base {};
Base* bp = nullptr;

// expected-warning@+1 {{dynamic_cast is not supported yet}}
static_assert(noexcept(dynamic_cast<Derived*>(bp)));

static_assert(noexcept((int)1.0));

int i = 0;
static_assert(noexcept(i++));
