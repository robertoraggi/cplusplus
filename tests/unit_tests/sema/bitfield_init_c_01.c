// RUN: %cxx -fsyntax-only -verify %s
// expected-no-diagnostics

// Test non-designated struct init with bitfields.
// Bitfields sharing a storage unit must not cause out-of-bounds insertvalue.

struct flags {
  unsigned int a : 1;
  unsigned int b : 1;
  unsigned int c : 2;
};

struct mixed {
  const char* name;
  int fd;
  unsigned int initialized : 1;
  unsigned int need_close : 1;
};

static struct flags f = {1, 0, 3};
static struct mixed m = {"key", 42, 1, 0};

int main(void) {
  struct flags g = {0, 1, 2};
  struct mixed n = {"val", 0, 0, 1};
  return (int)f.a + (int)g.b + (int)m.initialized + (int)n.need_close - 3;
}
