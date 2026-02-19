// clang-format off
// RUN: %cxx -toolchain macos -fcheck -dump-record-layouts %s | %filecheck %s

// Cross-type packing (Itanium ABI: all fit contiguously)
struct X {
  int i;
  int b : 2;
  int c : 10;
  int d : 4;
  long e : 2;
  unsigned long f : 3;
  int g : 1;
};

// CHECK: *** Dumping AST Record Layout
// CHECK:         0 | struct X
// CHECK:         0 |   int i
// CHECK:     4:0-1 |   int b
// CHECK:    4:2-11 |   int c
// CHECK:     5:4-7 |   int d
// CHECK:     6:0-1 |   long e
// CHECK:     6:2-4 |   unsigned long f
// CHECK:     6:5-5 |   int g
// CHECK:           | [sizeof=8, dsize=8, align=8,
// CHECK:           |  nvsize=8, nvalign=8]

// Same type, fits in one unit
struct S1 { int a : 3; int b : 29; };

// CHECK: *** Dumping AST Record Layout
// CHECK:         0 | struct S1
// CHECK:     0:0-2 |   int a
// CHECK:    0:3-31 |   int b
// CHECK:           | [sizeof=4, dsize=4, align=4,
// CHECK:           |  nvsize=4, nvalign=4]

// Same type, overflow -> new unit
struct S2 { int a : 3; int b : 30; };

// CHECK: *** Dumping AST Record Layout
// CHECK:         0 | struct S2
// CHECK:     0:0-2 |   int a
// CHECK:    4:0-29 |   int b
// CHECK:           | [sizeof=8, dsize=8, align=4,
// CHECK:           |  nvsize=8, nvalign=4]

// Different types: small to large
struct S3 { char a : 3; int b : 2; };

// CHECK: *** Dumping AST Record Layout
// CHECK:         0 | struct S3
// CHECK:     0:0-2 |   char a
// CHECK:     0:3-4 |   int b
// CHECK:           | [sizeof=4, dsize=4, align=4,
// CHECK:           |  nvsize=4, nvalign=4]

// Different types: large to small
struct S4 { int a : 3; char b : 2; };

// CHECK: *** Dumping AST Record Layout
// CHECK:         0 | struct S4
// CHECK:     0:0-2 |   int a
// CHECK:     0:3-4 |   char b
// CHECK:           | [sizeof=4, dsize=4, align=4,
// CHECK:           |  nvsize=4, nvalign=4]
