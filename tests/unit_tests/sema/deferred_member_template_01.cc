// RUN: %cxx -verify %s
// expected-no-diagnostics

typedef decltype(sizeof(int)) size_t;
typedef decltype(nullptr) nullptr_t;

inline void* operator new(size_t, void* __p) { return __p; }

template <class _Tp, class... _Args>
struct allocator_traits {
  template <class _Up, class... _A>
  static void construct(_Tp& __a, _Up* __p, _A&&... __args) {
    ::new ((void*)__p) _Up(__args...);
  }
};

template <class _Tp>
struct wrapper {
  template <class _Up>
  _Up cast(_Tp& __t) {
    return static_cast<_Up>(__t);
  }
};
