// RUN: %cxx -verify -fcheck %s

template <class... _Types>
struct __type_list {};

template <class _TL>
struct test_template;

template <class _Head>
struct test_template<__type_list<_Head>> {
  enum { value = 1 };
};

template <class _Head, class... _Tail>
struct test_template<__type_list<_Head, _Tail...>> {
  enum { value = 2 };
};

static_assert(test_template<__type_list<int>>::value == 1, "single element");

static_assert(test_template<__type_list<int, int>>::value == 2, "two elements");

static_assert(test_template<__type_list<int, float, double>>::value == 2,
              "three elements");

static_assert(test_template<__type_list<float>>::value == 1, "single float");
static_assert(test_template<__type_list<double>>::value == 1, "single double");

static_assert(test_template<__type_list<__type_list<int>>>::value == 1,
              "nested type_list as single element");

static_assert(test_template<__type_list<int, double, char>>::value == 2,
              "three mixed elements");
