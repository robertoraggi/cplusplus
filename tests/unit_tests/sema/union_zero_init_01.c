// RUN: %cxx -toolchain macos -verify %s
// expected-no-diagnostics

typedef union {
  unsigned long long x[2];
  double d;
  float f;
} Val;

void test_fpval(void) {
  Val fval = {0};
  (void)fval.d;
}

typedef struct {
  int tag;
  union {
    int i[4];
    double d;
  } data;
} Value;

void test_tagged(void) {
  Value t = {0};
  (void)t.tag;
}

typedef union {
  int arr[3];
  long l;
} ArrUnion;

void test_arr_union(void) {
  ArrUnion u = {0};
  (void)u.l;
}
