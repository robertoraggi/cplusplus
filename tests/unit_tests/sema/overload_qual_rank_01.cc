// RUN: %cxx -verify %s
// expected-no-diagnostics

namespace prefer_exact_over_qual {

void f(int*);
void f(const int*);

void test() {
  int x = 0;
  f(&x);
}

}  // namespace prefer_exact_over_qual

namespace qual_conv_only_candidate {

void g(const int*);

void test() {
  int x = 0;
  g(&x);
}

}  // namespace qual_conv_only_candidate
