// RUN: %cxx -std=c++20 -verify -fsyntax-only %s
// expected-no-diagnostics

#if !__has_builtin(__array_rank)
#error "__array_rank"
#endif

struct Incomplete;
struct Object {
  int field;
  void method();
};
union Union;
enum class Enum;
using Matrix = const volatile int[3][4];
using Unknown = Incomplete[][4];
using Function = void(int);
using QualifiedFunction = void() const;

template <class T>
struct Uninstantiated {
  static_assert(sizeof(T) == 0);
};

static_assert(__array_rank(Uninstantiated<int>) == 0);
static_assert(__array_rank(Uninstantiated<int>[2][3]) == 2);
static_assert(__array_rank(int) == 0);
static_assert(__array_rank(const volatile int) == 0);
static_assert(__array_rank(void) == 0);
static_assert(__array_rank(const void) == 0);
static_assert(__array_rank(Incomplete) == 0);
static_assert(__array_rank(Union) == 0);
static_assert(__array_rank(Enum) == 0);
static_assert(__array_rank(Object) == 0);
static_assert(__array_rank(Function) == 0);
static_assert(__array_rank(QualifiedFunction) == 0);
static_assert(__array_rank(decltype(nullptr)) == 0);
static_assert(__array_rank(int Object::*) == 0);
static_assert(__array_rank(void (Object::*)()) == 0);
static_assert(__array_rank(int*) == 0);
static_assert(__array_rank(int (*)[3][4]) == 0);
static_assert(__array_rank(Matrix&) == 0);
static_assert(__array_rank(Matrix&&) == 0);
static_assert(__array_rank(int[]) == 1);
static_assert(__array_rank(int[3]) == 1);
static_assert(__array_rank(int[0]) == 1);
static_assert(__array_rank(int[3][4]) == 2);
static_assert(__array_rank(Matrix) == 2);
static_assert(__array_rank(Unknown) == 2);
static_assert(__array_rank(int[][2][3][4][5]) == 5);
static_assert(__array_rank(int* [2][3]) == 2);
static_assert(__is_same(decltype(__array_rank(int)), __SIZE_TYPE__));
static_assert(__is_same(decltype(__array_rank(Matrix)), __SIZE_TYPE__));
static_assert(__is_same(decltype(__is_array(Matrix)), bool));
static_assert(__array_rank(Matrix) + __array_rank(int[2]) == 3);
static_assert(__array_rank(Matrix) - 3 > 0);

template <class T>
inline constexpr unsigned long rank_v = __array_rank(T);

template <class T>
constexpr auto rank() {
  static_assert(__is_same(decltype(__array_rank(T)), __SIZE_TYPE__));
  return __array_rank(T);
}

template <class T, int N>
struct Nested {
  using Array = T[N];
  static constexpr auto value = __array_rank(Array);
};

template <__SIZE_TYPE__ N>
struct Constant {
  static constexpr auto value = N;
};

template <class T>
using Rank = Constant<__array_rank(T)>;

template <class... Ts>
constexpr auto ranks() {
  return (__array_rank(Ts) + ... + 0);
}

static_assert(rank_v<int> == 0);
static_assert(rank_v<int[2]> == 1);
static_assert(rank_v<Matrix> == 2);
static_assert(rank<Unknown>() == 2);
static_assert(rank<Matrix&>() == 0);
static_assert(Nested<int[4][5], 3>::value == 3);
static_assert(Nested<int, 3>::value == 1);
static_assert(Rank<int>::value == 0);
static_assert(Rank<Matrix>::value == 2);
static_assert(ranks<int, int[2], Matrix, Unknown>() == 5);

void variable_bound(int n) {
  using Array = int[n][2];
  static_assert(__array_rank(Array) == 2);
  static_assert(__array_rank(Array*) == 0);
}
