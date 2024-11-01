// RUN: %cxx -verify -E %s -o - | %filecheck %s

#define FOR_EACH(V) \
  V(aa)             \
  V(bb)             \
  V(cc)

#define DECLARE_VAR(v) int v;

int main() { FOR_EACH(DECLARE_VAR) }

// CHECK: int aa ; int bb ; int cc ;
