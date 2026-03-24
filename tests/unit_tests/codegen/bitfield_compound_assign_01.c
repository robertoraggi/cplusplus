// RUN: %cxx -toolchain macos -emit-ir %s -o - | %filecheck %s

typedef struct {
  unsigned flags : 4;
} Bits;

void or_assign(Bits *b, int v) { b->flags |= v; }
void and_assign(Bits *b, int v) { b->flags &= v; }
void xor_assign(Bits *b, int v) { b->flags ^= v; }
void add_assign(Bits *b, int v) { b->flags += v; }

// CHECK: cxx.bitfield.load
// CHECK: cxx.bitfield.store
// CHECK: cxx.bitfield.load
// CHECK: cxx.bitfield.store
// CHECK: cxx.bitfield.load
// CHECK: cxx.bitfield.store
// CHECK: cxx.bitfield.load
// CHECK: cxx.bitfield.store
