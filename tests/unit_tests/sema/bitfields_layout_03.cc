// clang-format off
// RUN: %cxx -toolchain macos -fcheck -dump-record-layouts %s | %filecheck %s

// Bitfield then regular field then bitfield (Itanium ABI packing)
struct T1 { int a : 2; char b; int c : 3; };

// CHECK: *** Dumping AST Record Layout
// CHECK:         0 | struct T1
// CHECK:     0:0-1 |   int a
// CHECK:         1 |   char b
// CHECK:     2:0-2 |   int c
// CHECK:           | [sizeof=4, dsize=4, align=4,
// CHECK:           |  nvsize=4, nvalign=4]

// Short char then int bitfield — should pack into 4 bytes
struct T2 { char a; int b : 3; };

// CHECK: *** Dumping AST Record Layout
// CHECK:         0 | struct T2
// CHECK:         0 |   char a
// CHECK:     1:0-2 |   int b
// CHECK:           | [sizeof=4, dsize=4, align=4,
// CHECK:           |  nvsize=4, nvalign=4]

// Bitfield fills beyond type boundary — second alloc unit
struct T3 { int a : 17; char b; };

// CHECK: *** Dumping AST Record Layout
// CHECK:         0 | struct T3
// CHECK:    0:0-16 |   int a
// CHECK:         3 |   char b
// CHECK:           | [sizeof=4, dsize=4, align=4,
// CHECK:           |  nvsize=4, nvalign=4]

// Zero-width bitfield separator
struct T4 { int a : 3; int : 0; int b : 5; };

// CHECK: *** Dumping AST Record Layout
// CHECK:         0 | struct T4
// CHECK:     0:0-2 |   int a
// CHECK:    4:0-4 |   int b
// CHECK:           | [sizeof=8, dsize=8, align=4,
// CHECK:           |  nvsize=8, nvalign=4]

// Single zero-width bitfield (empty struct)
struct T5 { int : 0; };

// CHECK: *** Dumping AST Record Layout
// CHECK:         0 | struct T5
// CHECK:           | [sizeof=1, dsize=1, align=1,
// CHECK:           |  nvsize=1, nvalign=1]

// Aligned large bitfield
struct T6 { long a : 40; int b : 3; };

// CHECK: *** Dumping AST Record Layout
// CHECK:         0 | struct T6
// CHECK:    0:0-39 |   long a
// CHECK:     5:0-2 |   int b
// CHECK:           | [sizeof=8, dsize=8, align=8,
// CHECK:           |  nvsize=8, nvalign=8]

// Single bit
struct T7 { int a : 1; };

// CHECK: *** Dumping AST Record Layout
// CHECK:         0 | struct T7
// CHECK:     0:0-0 |   int a
// CHECK:           | [sizeof=4, dsize=4, align=4,
// CHECK:           |  nvsize=4, nvalign=4]

// Full width
struct T8 { int a : 32; };

// CHECK: *** Dumping AST Record Layout
// CHECK:         0 | struct T8
// CHECK:    0:0-31 |   int a
// CHECK:           | [sizeof=4, dsize=4, align=4,
// CHECK:           |  nvsize=4, nvalign=4]

// Unsigned long long
struct T9 { unsigned long long a : 40; unsigned long long b : 24; };

// CHECK: *** Dumping AST Record Layout
// CHECK:         0 | struct T9
// CHECK:    0:0-39 |   unsigned long long a
// CHECK:   5:0-23 |   unsigned long long b
// CHECK:           | [sizeof=8, dsize=8, align=8,
// CHECK:           |  nvsize=8, nvalign=8]

// Union with bitfields
union U1 { int a : 3; int b : 5; };

// CHECK: *** Dumping AST Record Layout
// CHECK:         0 | union U1
// CHECK:     0:0-2 |   int a
// CHECK:     0:0-4 |   int b
// CHECK:           | [sizeof=4, dsize=4, align=4,
// CHECK:           |  nvsize=4, nvalign=4]
