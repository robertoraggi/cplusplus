// RUN: %cxx -verify %s
// expected-no-diagnostics

struct Any {
  template <class T>
  operator T() const;
};

template <class T>
Any::operator T() const { return T(); }

void test() {
  Any a;
  int i = a;
  double d = a;
}
