// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

int main() {
  int x = 0;
  int& a = x;
  const int& b = 1;
  int&& c = 2;

  (void)a;
  (void)b;
  (void)c;
}
