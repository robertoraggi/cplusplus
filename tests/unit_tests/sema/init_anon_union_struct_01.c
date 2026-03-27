// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

struct S1 {
  int flags;
  double num;
  union {
    void* ptr;
    int* iptr;
  };
};

void test_s1_nested(void) {
  struct S1 a = {0, 0.0, {0}};
  struct S1 b = {1, 2.0, {(void*)0}};
}

void test_s1_flat(void) { struct S1 c = {0, 0.0, 0}; }

struct S2 {
  int tag;
  union {
    char c;
    long long ll;
  };
};

void test_s2_nested(void) {
  struct S2 a = {0, {0}};
  struct S2 b = {1, {'A'}};
}

void test_s2_flat(void) { struct S2 c = {0, 0}; }

struct S1 g_s1_zero = {0, 0.0, {0}};
struct S1 g_s1_flat = {0, 0.0, 0};
struct S2 g_s2_zero = {0, {0}};
struct S2 g_s2_char = {1, {'A'}};
