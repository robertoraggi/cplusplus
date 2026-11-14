// RUN: %cxx -emit-llvm %s -o %t.ll

template <class T>
struct Allocator {
  Allocator() = default;
  int i{};
};

template <class A>
struct Vector {
  explicit Vector(int n, A const& a = A()) {}
};

template <class T>
void g() {
  Vector<Allocator<int>> v(4);
}

void h() { g<int>(); }
