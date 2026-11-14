// RUN: %cxx -std=c++26 -verify -fsyntax-only %s

struct Table {
  static constexpr int values[2] = {1, 2};
  static constexpr int scalar = 3;
  static int mutableValue;
  struct Value {
    int values[3];
  };
  static constexpr Value value{{2, 4, 6}};
};

constexpr int writeArray() {
  const_cast<int&>(Table::values[0]) = 5;
  return Table::values[0];
}
static_assert(writeArray() == 5); // expected-error {{static assertion expression is not an integral constant expression}}

constexpr int writeScalar() {
  const_cast<int&>(Table::scalar) = 4;
  return Table::scalar;
}
static_assert(writeScalar() == 4); // expected-error {{static assertion expression is not an integral constant expression}}

constexpr int writeMutable() {
  Table::mutableValue = 1;
  return 1;
}
static_assert(writeMutable() == 1); // expected-error {{static assertion expression is not an integral constant expression}}

constexpr int copies() {
  auto first = Table::value;
  auto second = Table::value;
  first.values[1] = 30;
  return first.values[1] + second.values[1] * 100 + Table::value.values[1];
}
static_assert(copies() == 434);

constexpr int reads() {
  int sum = 0;
  for (int i = 0; i < 2; ++i) sum += Table::values[i];
  return sum + Table::scalar + Table::value.values[2];
}
static_assert(reads() == 12);
