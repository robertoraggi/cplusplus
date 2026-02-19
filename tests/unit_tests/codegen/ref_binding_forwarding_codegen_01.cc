// RUN: %cxx -verify -emit-mlir %s -o %t.cxir
// RUN: %cxx -verify -emit-mlir %s -o %t.mlir
// RUN: %cxx -verify -emit-llvm %s -o %t.ll

struct L {};
struct CL {};
struct R {};

L pick(int&);
CL pick(const int&);
R pick(int&&);

template <typename T>
auto forward_pick(T&& t) -> decltype(pick(static_cast<T&&>(t))) {
  return pick(static_cast<T&&>(t));
}

int main() {
  int x = 0;
  (void)forward_pick(x);
  (void)forward_pick(0);
  return 0;
}
