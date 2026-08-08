// RUN: not %cxx -fsyntax-only %s

template <class T>
void crref(const T&&) {}

void test() {
  int i = 0;
  crref(i);
}
