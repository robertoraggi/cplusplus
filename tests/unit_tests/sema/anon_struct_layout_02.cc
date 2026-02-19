// clang-format off
// RUN: %cxx -toolchain macos -fcheck -dump-record-layouts %s | %filecheck %s

// Anonymous union/struct with regular fields, verifying field ordering
// and layout when anonymous members are interleaved with named fields.

struct S1 {
  union {
    float f;
    int i;
  };
  int n;
};

// CHECK: *** Dumping AST Record Layout
// CHECK:         0 | struct S1
// CHECK:         0 |  union (anonymous)
// CHECK:         0 |    float f
// CHECK:         0 |    int i
// CHECK:         4 |  int n
// CHECK:           | [sizeof=8, dsize=8, align=4,
// CHECK:           |  nvsize=8, nvalign=4]

// Skip standalone anonymous layout
// CHECK: *** Dumping AST Record Layout

struct S2 {
  struct {
    int a;
    int b;
  };
  union {
    long l;
    double d;
  };
  int c;
};

// CHECK: *** Dumping AST Record Layout
// CHECK:         0 | struct S2
// CHECK:         0 |  struct (anonymous)
// CHECK:         0 |    int a
// CHECK:         4 |    int b
// CHECK:         8 |  union (anonymous)
// CHECK:         8 |    long l
// CHECK:         8 |    double d
// CHECK:        16 |  int c
// CHECK:           | [sizeof=24, dsize=24, align=8,
// CHECK:           |  nvsize=24, nvalign=8]

// Skip standalone layouts
// CHECK: *** Dumping AST Record Layout
// CHECK: *** Dumping AST Record Layout

struct S3 {
  union {
    struct {
      short x;
      short y;
    };
    int xy;
  };
};

// CHECK: *** Dumping AST Record Layout
// CHECK:         0 | struct S3
// CHECK:         0 |  union (anonymous)
// CHECK:         0 |    int xy
// CHECK:         0 |    struct (anonymous)
// CHECK:         0 |      short x
// CHECK:         2 |      short y
// CHECK:           | [sizeof=4, dsize=4, align=4,
// CHECK:           |  nvsize=4, nvalign=4]
