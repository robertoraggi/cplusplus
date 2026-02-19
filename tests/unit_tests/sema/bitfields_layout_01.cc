// RUN: %cxx -fcheck -dump-record-layouts %s | %filecheck %s

struct A {
  int a : 3;
  int b : 29;
};

// CHECK: *** Dumping AST Record Layout
// CHECK:         0 | struct A
// CHECK:     0:0-2 |   int a
// CHECK:    0:3-31 |   int b
// CHECK:           | [sizeof=4, dsize=4, align=4,
// CHECK:           |  nvsize=4, nvalign=4]

struct B {
  long long a : 3;
  long long b : 60;
  long long c : 1;
};

// CHECK: *** Dumping AST Record Layout
// CHECK:         0 | struct B
// CHECK:     0:0-2 |   long long a
// CHECK:    0:3-62 |   long long b
// CHECK:     7:7-7 |   long long c
// CHECK:           | [sizeof=8, dsize=8, align=8,
// CHECK:           |  nvsize=8, nvalign=8]

struct C {
  char a : 3;
  char b : 6;  // overflow char (8 bits) -> new unit
};

// CHECK: *** Dumping AST Record Layout
// CHECK:         0 | struct C
// CHECK:     0:0-2 |   char a
// CHECK:     1:0-5 |   char b
// CHECK:           | [sizeof=2, dsize=2, align=1,
// CHECK:           |  nvsize=2, nvalign=1]
