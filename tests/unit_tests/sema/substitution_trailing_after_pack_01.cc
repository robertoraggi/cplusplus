// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

// Variadic function template with deduced trailing parameter.
// The pack is deduced from explicit template arguments, the trailing
// parameter U is deduced from the call.
template <int N, class... Ts>
auto pick(int (&)[N], Ts...) -> char;

int arr3[3];

int trailing_1[sizeof(pick<3>(arr3)) == sizeof(char) ? 1 : -1];
int trailing_2[sizeof(pick<3, int>(arr3, 42)) == sizeof(char) ? 1 : -1];

// Class template with pack last and a defaulted parameter before it.
template <class T = int, class U = double>
struct DefaultPair {
  enum { size = sizeof(T) + sizeof(U) };
};

static_assert(DefaultPair<>::size == sizeof(int) + sizeof(double),
              "both defaults");
static_assert(DefaultPair<char>::size == sizeof(char) + sizeof(double),
              "one default");
static_assert(DefaultPair<char, float>::size == sizeof(char) + sizeof(float),
              "no defaults");
