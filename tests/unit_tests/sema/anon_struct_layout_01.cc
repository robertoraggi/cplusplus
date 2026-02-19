// clang-format off
// RUN: %cxx -toolchain macos -fcheck -dump-record-layouts %s | %filecheck %s

// Anonymous union with bitfields and regular fields interleaved with
// anonymous structs. Verifies layout sizes, offsets, and nesting.

struct X {
  union {
    int i;
    struct {
      int a : 4;
      int b : 2;
      int c : 6;
    };
  };

  struct {
    struct {
      int x;
      int y;
    };
  };
};

// CHECK: *** Dumping AST Record Layout
// CHECK:         0 | struct X
// CHECK:         0 |  union (anonymous)
// CHECK:         0 |    int i
// CHECK:         0 |    struct (anonymous)
// CHECK:     0:0-3 |      int a
// CHECK:     0:4-5 |      int b
// CHECK:    0:6-11 |      int c
// CHECK:         4 |  struct (anonymous)
// CHECK:         4 |    struct (anonymous)
// CHECK:         4 |      int x
// CHECK:         8 |      int y
// CHECK:           | [sizeof=12, dsize=12, align=4,
// CHECK:           |  nvsize=12, nvalign=4]

struct D : X {
  long m;
};

// Skip the standalone anonymous class layouts
// CHECK: *** Dumping AST Record Layout
// CHECK: *** Dumping AST Record Layout
// CHECK: *** Dumping AST Record Layout
// CHECK: *** Dumping AST Record Layout

// CHECK: *** Dumping AST Record Layout
// CHECK:         0 | struct D
// CHECK:         0 |  struct X (base)
// CHECK:         0 |    union (anonymous)
// CHECK:         0 |      int i
// CHECK:         0 |      struct (anonymous)
// CHECK:     0:0-3 |        int a
// CHECK:     0:4-5 |        int b
// CHECK:    0:6-11 |        int c
// CHECK:         4 |    struct (anonymous)
// CHECK:         4 |      struct (anonymous)
// CHECK:         4 |        int x
// CHECK:         8 |        int y
// CHECK:        16 |  long m
// CHECK:           | [sizeof=24, dsize=24, align=8,
// CHECK:           |  nvsize=24, nvalign=8]
