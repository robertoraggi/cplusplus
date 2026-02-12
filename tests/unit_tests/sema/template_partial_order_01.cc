// RUN: %cxx -verify -fcheck %s

template <typename T, typename U>
struct S {
  enum { value = 0 };
};

template <typename T>
struct S<T, int> {
  enum { value = 1 };
};

template <typename T>
struct S<T, T> {
  enum { value = 2 };
};

static_assert(S<char, int>::value == 1);
static_assert(S<char, char>::value == 2);

int main() { return 0; }
