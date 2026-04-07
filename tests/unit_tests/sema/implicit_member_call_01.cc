// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

struct Counter {
  int n = 0;

  void increment() { n++; }
  void inc_twice() {
    increment();
    increment();
  }

  void get() const { (void)n; }
  void check() const { get(); }
};

struct Chain {
  void a() {}
  void b() { a(); }
  void c() { b(); }
};

void test() {
  Counter cnt;
  cnt.inc_twice();
  cnt.check();

  Chain ch;
  ch.c();
}
