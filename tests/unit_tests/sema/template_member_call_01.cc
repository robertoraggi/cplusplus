// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

template <class T>
struct Box {
  template <class U>
  U get() const { return U{}; }
};

template <class T, class U>
U extract(const Box<T>& b) {
  return b.template get<U>();
}

void test() {
  Box<float> b;
  int i = extract<float, int>(b);
}
