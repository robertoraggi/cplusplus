// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

struct Stream {};

Stream& operator<<(Stream& s, int) { return s; }
Stream& operator>>(Stream& s, int&) { return s; }

int test_shift() {
  Stream s;
  int x = 0;
  s << 1;
  s >> x;
  return x;
}
