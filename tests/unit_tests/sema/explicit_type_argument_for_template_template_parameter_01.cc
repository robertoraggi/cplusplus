// RUN: %cxx -verify -fsyntax-only -std=c++26 %s
// expected-no-diagnostics

template <class T>
struct Alloc {};

template <class T, template <class> class A>
struct Mat {
  template <class U, template <class> class M>
  friend constexpr int mv(const Mat<U, M>&);
};

template <class T, class Index = long>
constexpr int mv(long) {
  return sizeof(Index);
}

template <class T, template <class> class M>
constexpr int mv(const Mat<T, M>&) {
  return mv<T, int>(1) + mv<T, char>(1);
}

static_assert(mv(Mat<double, Alloc>{}) == sizeof(int) + sizeof(char));

template <class T, template <class> class M>
constexpr int declaredFirst(const Mat<T, M>&);

template <class T, class Index = long>
constexpr int declaredFirst(long) {
  return sizeof(Index);
}

template <class T, template <class> class M>
constexpr int declaredFirst(const Mat<T, M>&) {
  return declaredFirst<T, char>(1);
}

static_assert(declaredFirst(Mat<double, Alloc>{}) == sizeof(char));

template <template <class> class M, class T>
constexpr int first(M<T>) {
  return 1;
}

template <class T, class U>
constexpr int first(U) {
  return 2;
}

template <class T>
struct Holder {
  static constexpr int value = first<int, T*>(nullptr) + first<Alloc>(Alloc<T>{});
};

static_assert(Holder<char>::value == 3);
