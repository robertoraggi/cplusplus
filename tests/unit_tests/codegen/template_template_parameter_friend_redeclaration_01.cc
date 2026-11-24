// RUN: %cxx -toolchain wasm32 -std=c++26 -emit-llvm %s -o - | %filecheck %s

template <class T>
struct Alloc {};

template <class T, class V>
struct Vec {
  T* p;
};

template <class T, template <class> class A>
class Mat {
  template <class U, template <class> class M, class N>
  friend void mv(Vec<U, N>& y, const Mat<U, M>& a, const Vec<U, N>& x);
  int n = 1;
};

template <class T, class Index = long>
void mv(long n, T* y, const T* x) {
  for (long i = 0; i < n; ++i) y[i] = x[i];
}

template <class T, template <class> class M, class N>
void mv(Vec<T, N>& y, const Mat<T, M>& a, const Vec<T, N>& x) {
  mv<T, int>(a.n, y.p, x.p);
}

void use(Vec<double, int>& y, const Vec<double, int>& x) {
  Mat<double, Alloc> m;
  mv(y, m, x);
}

// CHECK-LABEL: define linkonce_odr void @_Z2mvIdiEvlPT_PKS0_(
// CHECK-LABEL: define linkonce_odr void @_Z2mvId5AllociEvR3VecIT_T1_ERK3MatIS2_T0_ERKS4_(
// CHECK: call void @_Z2mvIdiEvlPT_PKS0_(
// CHECK-LABEL: define void @_Z3useR3VecIdiERKS0_(
// CHECK: call void @_Z2mvId5AllociEvR3VecIT_T1_ERK3MatIS2_T0_ERKS4_(
