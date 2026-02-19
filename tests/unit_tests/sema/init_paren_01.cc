// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

void test_paren_initialization() {
  auto a(1);
  int check_auto_size[(sizeof(a) == sizeof(int)) ? 1 : -1];
  (void)check_auto_size;

  int b(2);
  int c((3));
  int* p(0);
  int* q((0));

  (void)b;
  (void)c;
  (void)p;
  (void)q;
}
