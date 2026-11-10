// RUN: %cxx -std=c++26 -toolchain macos -emit-llvm %s -o - | %filecheck %s

// CHECK: define{{.*}}@_Z1mITk8HasValue1XENT_10value_typeES1_(

template <class T>
concept HasValue = requires { typename T::value_type; };

struct X {
  using value_type = int;
};

template <HasValue T>
typename T::value_type m(T) {
  return 0;
}

int n() { return m(X{}); }
